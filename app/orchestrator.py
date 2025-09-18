from typing import Optional, Dict, Any, List, Tuple
import logging
import os
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from fuzzywuzzy import fuzz
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY, MIN_SIMILARITY_THRESHOLD
from .retrievers import isbn_exact, hybrid_with_rerank
from .utils_isbn import extract_first_isbn
from .mixed_filters_parser import MixedFiltersParser
from .debug_reporter import (start_debug, set_debug_intent, add_debug_step, add_debug_issue, finalize_debug,
                           set_debug_query_processing, set_debug_search_queries, add_debug_filtered_result, 
                           set_debug_final_response, set_debug_final_books, set_performance_metrics, set_search_metrics)

logger = logging.getLogger(__name__)

# Ensure LangSmith tracing is enabled
if os.getenv("LANGSMITH_TRACING") == "true":
    logger.info("🔍 LangSmith tracing enabled")
else:
    logger.warning("⚠️ LangSmith tracing not enabled")

class ChatState(BaseModel):
    session_id: str
    message: str
    intent: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    exclude_filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarify: bool = False
    clarify_question: Optional[str] = None
    # Mixed filters support
    mixed_filters_result: Optional[Dict[str, Any]] = None
    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    # Detailed search metrics
    search_metrics: Dict[str, float] = Field(default_factory=dict)

# Global storage for mixed filters functions (outside Pydantic)
_mixed_filters_functions: Dict[str, Tuple[Optional[callable], Optional[callable]]] = {}

llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)
mixed_filters_parser = MixedFiltersParser()

def _filter_results_by_relevance(query: str, docs: List, threshold: float = MIN_SIMILARITY_THRESHOLD) -> List:
    """Filter documents by semantic similarity to query"""
    if not docs:
        return []
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from .settings import OPENAI_MODEL_EMBED
        import math
        
        embeddings_func = OpenAIEmbeddings(model=OPENAI_MODEL_EMBED, api_key=OPENAI_API_KEY)
        query_embedding = embeddings_func.embed_query(query)
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            if magnitude1 == 0 or magnitude2 == 0:
                return 0
            return dot_product / (magnitude1 * magnitude2)
        
        filtered_docs = []
        for doc in docs:
            # Create text for embedding from document content/metadata
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            elif hasattr(doc, 'metadata') and doc.metadata:
                # Create searchable text from metadata
                meta = doc.metadata
                text_parts = []
                if meta.get('title'): text_parts.append(meta['title'])
                if meta.get('author'): text_parts.append(meta['author'])
                if meta.get('summary'): text_parts.append(meta['summary'])
                if meta.get('primary_genre'): text_parts.append(meta['primary_genre'])
                text = ' '.join(text_parts)
            else:
                text = str(doc)
            
            if not text.strip():
                continue
                
            doc_embedding = embeddings_func.embed_query(text)
            similarity = cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= threshold:
                filtered_docs.append(doc)
                
        return filtered_docs
    except Exception:
        # If filtering fails, return original docs
        return docs

def _is_simple_query(query: str) -> bool:
    """Check if query is simple enough to skip mixed_filters parsing"""
    words = query.strip().split()
    
    # Very short queries (1-3 words) without explicit DSL syntax
    if len(words) <= 3 and ':' not in query and '"' not in query and ' by ' not in query.lower():
        # Check if it's likely a simple topic/genre query
        # Exclude obvious author+title patterns
        capitalized_words = [w for w in words if w[0].isupper()]
        
        # If all words are lowercase or there's only one proper noun, likely simple topic
        if len(capitalized_words) <= 1:
            return True
            
        # If 2-3 words and no obvious book title indicators, could be simple
        has_common_title_words = any(word.lower() in ['the', 'a', 'an', 'of', 'and', 'or'] for word in words)
        if not has_common_title_words and len(words) <= 3:
            return True
    
    return False

def _get_pattern_analysis_details(query: str) -> Dict[str, Any]:
    """Detailed pattern analysis for debug purposes"""
    import re
    from .utils_isbn import extract_first_isbn
    
    details = {
        "patterns_found": [],
        "pattern_count": 0,
        "query_features": {
            "word_count": len(query.strip().split()),
            "has_quotes": '"' in query or "'" in query,
            "has_colons": ":" in query,
            "has_caps_words": bool([w for w in query.split() if w.isupper() and len(w) > 1]),
            "has_by_pattern": " by " in query.lower(),
            "has_isbn": bool(extract_first_isbn(query))
        }
    }
    
    # Check various patterns
    patterns = [
        ("ISBN pattern", r'isbn[:\s]*\d'),
        ("Title DSL", r'title[:\s]*["\']'),
        ("Author DSL", r'author[:\s]*["\']'),
        ("Genre DSL", r'genre[:\s]*\w+'),
        ("Topics DSL", r'topics?[:\s]*[\(\[]'),
        ("Year DSL", r'year[:\s]*\d{4}'),
        ("Language DSL", r'lang(?:uage)?[:\s]*[a-z]{2}'),
        ("NOT pattern", r'NOT\s+\w+'),
        ("Negative pattern", r'-\w+'),
        ("Quote+by pattern", r'"[^"]+"\s+by\s+'),
        ("ISBN+author", r'isbn\s+[\d-]+\s+by\s+'),
        ("Year range", r'\d{4}\.\.\d{4}'),
        ("Published pattern", r'published\s+(after|before)'),
        ("Exclusion words", r'\b(?:but\s+not|except|excluding|without|но\s+не|кроме|исключая)\s+\w+'),  # Russian: но не, кроме, исключая
    ]
    
    for pattern_name, pattern in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            details["patterns_found"].append(pattern_name)
    
    details["pattern_count"] = len(details["patterns_found"])
    
    # Determine likely intent based on patterns
    if details["query_features"]["has_isbn"]:
        details["likely_intent"] = "ISBN"
    elif "Quote+by pattern" in details["patterns_found"]:
        details["likely_intent"] = "author_title"
    elif len(details["patterns_found"]) >= 2:
        details["likely_intent"] = "mixed_filters"
    elif "Exclusion words" in details["patterns_found"]:
        details["likely_intent"] = "negative_filter"
    else:
        details["likely_intent"] = "simple/topic"
    
    return details

INTENT_SYS = """You are an intent router for a book search assistant.
Decide one intent: isbn | author_title | author | title | genre | topic | negative_filter | year_range | mixed_filters | free_text | clarify.

CRITICAL: Use negative_filter for ANY query with exclusion words like NOT, but, except, excluding, without, не про, но не, кроме, исключая! (Russian exclusion words)

Examples:
- "SomeBook Author Name" → author_title (NOT mixed_filters)
- "Author Name SomeBook" → author_title (NOT mixed_filters) 
- "Author books" → author
- "SomeTitle" → title
- "fantasy books" → genre  
- "leadership" → topic
- "self-help NOT spirituality" → negative_filter (has NOT)
- "books about love but not romance" → negative_filter (has "but not")
- "science books except fiction" → negative_filter (has "except")
- "книги про медицину но не про хирургию" → negative_filter (has "но не")  # Russian: books about medicine but not surgery
- "Дай книгу по самодисциплине, но не про духовность" → negative_filter (has "но не про")  # Russian: Give me a book on self-discipline but not about spirituality
- "A book about Big Brother and censorship but not science fiction" → negative_filter (has "but not")
- "books from 2020-2024" → year_range
- "title:\"Book\" author:\"Name\"" → mixed_filters (DSL syntax)
- "ISBN 123 by Different Author" → mixed_filters (conflict)

negative_filter is for ANY query with exclusion/negation patterns:
- English: NOT, but not, except, without, excluding, minus
- Russian: не про, но не, кроме, исключая, минус  # Russian exclusion words
- Czech: ale ne, kromě, bez

mixed_filters is ONLY for:
1. DSL syntax with colons: title:"Book" author:"Name"  
2. Explicit conflicts: ISBN + wrong author

Return compact JSON: {"intent":"...", "filters":{...}, "exclude_filters":{...}, "clarify":""}.

For negative_filter intent:
- Put positive terms in "filters" (topics, genres, etc.)
- Put excluded terms in "exclude_filters": {"topics": ["spirituality"], "genres": ["science fiction"]}

For year_range intent, use filters like: {"year_from": 2020, "year_to": 2024} or {"year_after": 2010} or {"year_before": 2000}.

Be strict: do not invent values."""

def node_detect_intent(state: ChatState) -> ChatState:
    import uuid
    node_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    logger.info(f"🧠 [orchestrator.py] [{node_id}] node_detect_intent - Analyzing user intent...")
    logger.info(f"    Purpose: Determine what user wants (ISBN, author/title, genre, etc.)")
    logger.info(f"    Input: '{state.message}'")
    
    # Start debug session
    start_debug(state.message, state.session_id)
    
    # Quick pre-check: if it's a simple query, don't use mixed_filters
    simple_query_check = _is_simple_query(state.message)
    
    # Check if query needs mixed_filters parsing first (including author+title)
    use_mixed_filters = not simple_query_check and mixed_filters_parser.should_use_mixed_filters(state.message)
    
    # Record query processing flags for debug
    set_debug_query_processing(simple_query=simple_query_check, mixed_filters=use_mixed_filters)
    
    if use_mixed_filters:
        logger.info("🔧 [orchestrator.py] Complex query detected, using mixed_filters parser...")
        
        try:
            mixed_result = mixed_filters_parser.parse_query(state.message, use_llm=True)
            logger.info(f"🔧 [orchestrator.py] Mixed result received: ISBN={mixed_result.filters.isbn}, Author={mixed_result.filters.author}")
            
            # Store mixed filters result
            state.mixed_filters_result = mixed_result.model_dump()
            logger.info(f"🔧 [orchestrator.py] Stored to state: {state.mixed_filters_result}")
            
            # Check if we should downgrade from mixed_filters to simpler intent
            non_empty_filters = {k: v for k, v in mixed_result.filters.model_dump().items() 
                               if v is not None and v != [] and v != {}}
            
            # Determine appropriate intent based on detected filters
            if len(non_empty_filters) == 1:
                filter_name = list(non_empty_filters.keys())[0]
                logger.info(f"🔧 [orchestrator.py] Single filter detected: {filter_name}, downgrading intent")
                
                if filter_name == 'author':
                    state.intent = "author"
                elif filter_name == 'title':
                    state.intent = "title"
                elif filter_name == 'topics':
                    state.intent = "topic"
                elif filter_name == 'primary_genre':
                    state.intent = "genre"
                else:
                    state.intent = "mixed_filters"
            elif len(non_empty_filters) == 2 and 'title' in non_empty_filters and 'author' in non_empty_filters:
                # Special case: title + author = author_title intent
                logger.info(f"🔧 [orchestrator.py] Author+Title detected, using author_title intent")
                state.intent = "author_title"
            else:
                state.intent = "mixed_filters"
            
            # Convert to legacy format for compatibility
            filters_data = mixed_result.filters
            state.filters = {}
            state.exclude_filters = {}
            
            if filters_data.isbn:
                state.filters["isbn13"] = filters_data.isbn
            if filters_data.title:
                state.filters["title"] = filters_data.title
            if filters_data.author:
                state.filters["author"] = filters_data.author
            if filters_data.primary_genre:
                state.filters["primary_genre"] = filters_data.primary_genre
            if filters_data.topics:
                state.filters["topics"] = filters_data.topics
            if filters_data.year:
                if isinstance(filters_data.year, dict):
                    state.filters.update(filters_data.year)
                else:
                    state.filters["year"] = filters_data.year
            if filters_data.lang_filter:
                state.filters["lang_filter"] = filters_data.lang_filter
                
            # Handle exclusions
            if filters_data.exclude_topics:
                state.exclude_filters["topics"] = filters_data.exclude_topics
            if filters_data.exclude_genres:
                state.exclude_filters["genres"] = filters_data.exclude_genres
            
            # Handle conflicts and clarification
            if mixed_result.conflict.has_conflict:
                state.need_clarify = True
                state.clarify_question = mixed_result.clarify.question or mixed_result.conflict.reason
                logger.info(f"⚠️ [orchestrator.py] Conflict detected: {state.clarify_question}")
            
            # Compile post-filters for later use
            predicate_filter, booster_function = mixed_filters_parser.compile_post_filters(mixed_result)
            _mixed_filters_functions[state.session_id] = (predicate_filter, booster_function)
            
            logger.info(f"✅ [orchestrator.py] Mixed filters parsing successful:")
            logger.info(f"    Intent: '{state.intent}'")
            logger.info(f"    Filters: {state.filters}")
            logger.info(f"    Exclude filters: {state.exclude_filters}")
            logger.info(f"    Need clarify: {state.need_clarify}")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ [orchestrator.py] Mixed filters parsing failed: {e}")
            logger.info("    Falling back to legacy intent detection...")
    
    # Use LLM for intent detection for non-mixed queries
    logger.info("    LLM Call: gpt-4o-mini to analyze query intent")
    
    out = llm.invoke([("system", INTENT_SYS), ("user", state.message)]).content
    
    logger.info(f"    LLM Response: {out}")
    
    import json, re
    try:
        j = json.loads(out)
    except Exception:
        logger.warning("⚠️  [orchestrator.py] LLM response not valid JSON, trying regex...")
        m = re.search(r"\{.*\}", out, re.S)
        j = json.loads(m.group(0)) if m else {"intent":"free_text","filters":{}}
    
    state.intent = j.get("intent") or "free_text"
    state.filters = j.get("filters") or {}
    state.exclude_filters = j.get("exclude_filters") or {}
    clar = j.get("clarify") or ""
    
    if state.intent == "clarify" and clar.strip():
        state.need_clarify = True
        state.clarify_question = clar.strip()
        logger.info(f"❓ [orchestrator.py] Need clarification: {clar}")
    
    # Handle simple ISBN override if detected
    isbn = extract_first_isbn(state.message)
    if isbn and state.intent not in ["mixed_filters"]:
        logger.info(f"📖 [orchestrator.py] ISBN override: {isbn}")
        state.intent = "isbn"
        state.filters = {"isbn": isbn.get("isbn13") or isbn.get("isbn10") or isbn.get("isbn")}
    
    logger.info(f"✅ [orchestrator.py] Final intent detected:")
    logger.info(f"    Intent: '{state.intent}'")
    logger.info(f"    Filters: {state.filters}")
    logger.info(f"    Exclude filters: {state.exclude_filters}")
    logger.info(f"    Need clarify: {state.need_clarify}")
    
    # Record intent detection for debug with detailed info
    confidence = "high" if state.intent in ["isbn", "mixed_filters", "author_title"] else "medium"
    
    # Determine how intent was detected
    intent_method = "function"
    intent_details = {}
    
    if state.intent == "isbn":
        intent_method = "ISBN regex"
        intent_details["extracted_isbn"] = state.filters.get("isbn")
    elif hasattr(state, 'mixed_filters_result') and state.mixed_filters_result:
        intent_method = "mixed_filters + LLM"
        intent_details["used_llm"] = "Yes"
        intent_details["original_mixed_intent"] = "mixed_filters"
        intent_details["downgraded_to"] = state.intent
        intent_details["non_empty_filters"] = len([k for k, v in state.filters.items() if v and v != [] and v != {}])
    elif state.intent == "free_text":
        intent_method = "LLM fallback"
        intent_details["reason"] = "No specific patterns detected"
    else:
        intent_method = "pattern analysis"
        intent_details["simple_query_check"] = simple_query_check
        intent_details["mixed_filters_used"] = use_mixed_filters
        intent_details["pattern_analysis"] = _get_pattern_analysis_details(state.message)
        
    set_debug_intent(state.intent, state.filters, confidence, intent_method, intent_details)
    
    # Record performance metrics
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["detect_intent_ms"] = execution_time
    logger.info(f"⏱️ [orchestrator.py] Intent detection completed in {execution_time:.1f}ms")
    
    return state

def _docs_to_payload(docs):
    out = []
    for d in docs:
        m = d.metadata or {}
        out.append({
            "title": m.get("title"),
            "author": m.get("author"),
            "isbn13": m.get("isbn13"),
            "isbn10": m.get("isbn10"),
            "language": m.get("language"),
            "year": m.get("year"),
            "primary_genre": m.get("primary_genre"),
            "secondary_genres": m.get("secondary_genres"),
            "summary": m.get("summary"),
            "document_id": m.get("document_id"),
        })
    return out

def _check_author_title_conflicts(state: ChatState, results: List[Dict]) -> ChatState:
    """Check for author-title conflicts and suggest clarification"""
    if not state.filters or not results:
        return state
    
    query_author = (state.filters.get("author") or "").lower().strip()
    query_title = (state.filters.get("title") or "").lower().strip()
    
    if not (query_author and query_title):
        return state
    
    logger.info("🔍 [orchestrator.py] Checking for author-title conflicts...")
    
    for result in results[:3]:  # Check top 3 results
        db_author = (result.get("author") or "").lower().strip()
        db_title = (result.get("title") or "").lower().strip()
        
        # Check if title matches but author doesn't
        title_similarity = fuzz.ratio(query_title, db_title)
        author_similarity = fuzz.ratio(query_author, db_author)
        
        logger.info(f"    Comparing: '{query_title}' vs '{db_title}' (similarity: {title_similarity}%)")
        logger.info(f"    Authors: '{query_author}' vs '{db_author}' (similarity: {author_similarity}%)")
        
        # If title matches well but author doesn't
        if title_similarity > 70 and author_similarity < 60:
            logger.info("⚠️ [orchestrator.py] CONFLICT DETECTED: Title match but author mismatch")
            state.need_clarify = True
            state.clarify_question = f"Did you mean '{result.get('title')}' by {result.get('author')}? (Author specified in query: {state.filters.get('author')})"
            # Still include the result but mark for clarification
            state.results = [result]
            return state
    
    return state

def _apply_negative_filters(state: ChatState, payload: List[Dict]) -> List[Dict]:
    """Apply negative filtering based on exclude_filters"""
    if not state.exclude_filters or state.intent != "negative_filter":
        return payload
    
    logger.info("🚫 [orchestrator.py] Applying negative filters...")
    logger.info(f"    Exclude filters: {state.exclude_filters}")
    
    filtered_results = []
    
    for result in payload:
        should_exclude = False
        
        # Check exclude topics
        exclude_topics = state.exclude_filters.get("topics", [])
        if exclude_topics:
            result_topics = []
            if result.get("primary_genre"):
                result_topics.append(result["primary_genre"].lower())
            if result.get("secondary_genres"):
                result_topics.extend([g.lower() for g in result["secondary_genres"]])
            
            for exclude_topic in exclude_topics:
                if any(exclude_topic.lower() in topic for topic in result_topics):
                    logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (matches excluded topic: {exclude_topic})")
                    should_exclude = True
                    break
        
        # Check exclude genres
        exclude_genres = state.exclude_filters.get("genres", [])
        if exclude_genres and result.get("primary_genre"):
            primary_genre = result["primary_genre"].lower()
            if any(genre.lower() in primary_genre for genre in exclude_genres):
                logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (matches excluded genre)")
                should_exclude = True
        
        if not should_exclude:
            filtered_results.append(result)
            logger.info(f"    ✅ KEEPING: '{result.get('title')}'")
    
    logger.info(f"    📋 Filtered results: {len(filtered_results)} from {len(payload)}")
    return filtered_results

def _check_mixed_filters_consistency(state: ChatState, payload: List[Dict]) -> ChatState:
    """Check that all parameters in mixed_filters query belong to the same book"""
    if state.intent != "mixed_filters" or not payload:
        return state
        
    logger.info("🔍 [orchestrator.py] Checking mixed_filters parameter consistency...")
    
    # Get all the parameters from mixed_filters_result
    mixed_data = state.mixed_filters_result or {}
    filters = mixed_data.get('filters', {})
    
    # Extract the parameters we need to verify
    isbn = filters.get('isbn')
    title = filters.get('title') 
    author = filters.get('author')
    primary_genre = filters.get('primary_genre')
    year = filters.get('year')
    lang_filter = filters.get('lang_filter')
    
    logger.info(f"    Parameters to verify:")
    if isbn: logger.info(f"      ISBN: {isbn}")
    if title: logger.info(f"      Title: {title}")
    if author: logger.info(f"      Author: {author}")
    if primary_genre: logger.info(f"      Genre: {primary_genre}")
    if year: logger.info(f"      Year: {year}")
    if lang_filter: logger.info(f"      Language: {lang_filter}")
    
    # Check each result against all parameters
    for i, result in enumerate(payload[:3]):  # Check top 3 results
        book_title = result.get('title', 'Unknown')
        book_author = result.get('author', 'Unknown') 
        book_isbn = result.get('isbn13')
        book_genre = result.get('primary_genre')
        book_year = result.get('year')
        book_lang = result.get('language')
        
        logger.info(f"    Checking result #{i+1}: '{book_title}' by {book_author}")
        
        conflicts = []
        
        # ISBN conflict check
        if isbn and book_isbn and isbn != book_isbn:
            conflicts.append(f"ISBN mismatch: expected {isbn}, found {book_isbn}")
        
        # Author conflict check (fuzzy)
        if author and book_author:
            author_similarity = fuzz.ratio(author.lower(), book_author.lower())
            if author_similarity < 70:  # Low similarity = conflict
                conflicts.append(f"Author mismatch: expected '{author}', found '{book_author}' (similarity: {author_similarity}%)")
        
        # Title conflict check (fuzzy)
        if title and book_title:
            title_similarity = fuzz.ratio(title.lower(), book_title.lower())
            if title_similarity < 60:  # Lower threshold for titles
                conflicts.append(f"Title mismatch: expected '{title}', found '{book_title}' (similarity: {title_similarity}%)")
        
        # Genre conflict check
        if primary_genre and book_genre:
            genre_similarity = fuzz.ratio(primary_genre.lower(), book_genre.lower())
            if genre_similarity < 80:  # Strict for genres
                conflicts.append(f"Genre mismatch: expected '{primary_genre}', found '{book_genre}'")
        
        # Year conflict check
        if year and book_year:
            try:
                expected_year = int(year) if isinstance(year, (str, int)) else None
                actual_year = int(book_year)
                if expected_year and abs(expected_year - actual_year) > 1:  # Allow 1 year difference
                    conflicts.append(f"Year mismatch: expected {expected_year}, found {actual_year}")
            except:
                pass
        
        # Language conflict check
        if lang_filter and book_lang and lang_filter.lower() != book_lang.lower():
            conflicts.append(f"Language mismatch: expected '{lang_filter}', found '{book_lang}'")
        
        if conflicts:
            logger.info(f"    ⚠️ CONFLICTS DETECTED for '{book_title}':")
            for conflict in conflicts:
                logger.info(f"      - {conflict}")
            
            # Generate clarification question
            state.need_clarify = True
            primary_conflict = conflicts[0]
            
            if "ISBN" in primary_conflict:
                state.clarify_question = f"ISBN {isbn} belongs to book '{book_title}' by {book_author}. Did you mean this book, not the author {author}?"
            elif "Author" in primary_conflict:
                state.clarify_question = f"Found book '{book_title}' by {book_author}. Did you mean this author, not {author}?"
            elif "Genre" in primary_conflict:
                state.clarify_question = f"Book '{book_title}' belongs to genre '{book_genre}', not '{primary_genre}'. Were you looking for this specific book?"
            else:
                state.clarify_question = f"Found discrepancy: {primary_conflict}. Please clarify your query."
            
            # Return only the conflicting result for user to confirm
            state.results = [result]
            logger.info(f"    🚨 Conflict detected, asking for clarification")
            return state
        else:
            logger.info(f"    ✅ No conflicts found for '{book_title}'")
    
    logger.info(f"    📋 All parameters consistent across top results")
    return state

def _apply_year_filters(state: ChatState, payload: List[Dict]) -> List[Dict]:
    """Apply year range filtering"""
    if state.intent != "year_range" or not state.filters:
        return payload
    
    logger.info("📅 [orchestrator.py] Applying year range filters...")
    logger.info(f"    Year filters: {state.filters}")
    
    filtered_results = []
    
    year_from = state.filters.get("year_from")
    year_to = state.filters.get("year_to") 
    year_after = state.filters.get("year_after")
    year_before = state.filters.get("year_before")
    
    for result in payload:
        book_year = result.get("year")
        if not book_year:
            logger.info(f"    ⚠️ SKIPPING: '{result.get('title')}' (no year information)")
            continue
            
        try:
            book_year = int(book_year)
        except (ValueError, TypeError):
            logger.info(f"    ⚠️ SKIPPING: '{result.get('title')}' (invalid year: {book_year})")
            continue
        
        should_include = True
        
        if year_from is not None and book_year < year_from:
            should_include = False
            logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (year {book_year} < {year_from})")
            
        if year_to is not None and book_year > year_to:
            should_include = False
            logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (year {book_year} > {year_to})")
            
        if year_after is not None and book_year <= year_after:
            should_include = False
            logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (year {book_year} <= {year_after})")
            
        if year_before is not None and book_year >= year_before:
            should_include = False
            logger.info(f"    ❌ EXCLUDING: '{result.get('title')}' (year {book_year} >= {year_before})")
        
        if should_include:
            filtered_results.append(result)
            logger.info(f"    ✅ KEEPING: '{result.get('title')}' (year {book_year})")
    
    logger.info(f"    📋 Year filtered results: {len(filtered_results)} from {len(payload)}")
    return filtered_results

def node_route_search(state: ChatState) -> ChatState:
    import uuid
    node_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    logger.info(f"🔍 [orchestrator.py] [{node_id}] node_route_search - Searching for relevant documents...")
    logger.info(f"    Purpose: Find books based on detected intent '{state.intent}'")
    logger.info(f"    Filters: {state.filters}")
    logger.info(f"    Exclude filters: {state.exclude_filters}")
    
    if state.intent == "isbn":
        logger.info("📚 [orchestrator.py] ISBN search path:")
        logger.info("    Step 1: Trying exact ISBN match...")
        
        docs = isbn_exact(state.message)
        if docs:
            logger.info(f"    ✅ Found {len(docs)} exact ISBN matches")
            add_debug_step("ISBN Exact Match", state.message, True, len(docs), 
                          details={"search_type": "isbn_exact"})
            
            # Check for author conflict in ISBN search
            payload = _docs_to_payload(docs[:1])
            if state.filters.get("author") and payload:
                result = payload[0]
                query_author = state.filters["author"].lower()
                db_author = (result.get("author") or "").lower()
                
                author_similarity = fuzz.ratio(query_author, db_author)
                
                if author_similarity < 60:
                    logger.info("⚠️ [orchestrator.py] ISBN CONFLICT: Author mismatch detected")
                    add_debug_issue(f"ISBN author conflict: expected '{query_author}', found '{db_author}' (similarity: {author_similarity}%)")
                    state.need_clarify = True
                    state.clarify_question = f"ISBN {state.filters.get('isbn13')} belongs to book '{result.get('title')}' by {result.get('author')}, but query specifies {state.filters['author']}. Did you mean this book?"
            
            state.results = payload
        else:
            logger.info("    ⚠️  No exact ISBN match, trying semantic search...")
            add_debug_step("ISBN Exact Match", state.message, False, 0, "No ISBN found in database")
            
            retr = hybrid_with_rerank(intent=state.intent)
            docs = retr.invoke(state.message, intent=state.intent)
            logger.info(f"    📊 Semantic search returned {len(docs)} documents")
            add_debug_step("ISBN Fallback Search", state.message, len(docs) > 0, len(docs), 
                          details={"search_type": "semantic_fallback"})
            state.results = _docs_to_payload(docs[:1])
            
        logger.info(f"    📋 Final ISBN results: {len(state.results)} books")
        return state

    logger.info("🔎 [orchestrator.py] Semantic search path:")
    logger.info("    Using OptimizedThresholdRetriever with similarity filtering")
    
    # Use hybrid search with threshold filtering for all other intents
    retr = hybrid_with_rerank(intent=state.intent) 
    
    # Record search queries for debug
    set_debug_search_queries(bm25_query=state.message, vector_query=state.message)
    
    # Measure core search time
    search_start_time = time.time()
    try:
        docs = retr.invoke(state.message, intent=state.intent)
        add_debug_step("Hybrid Search (BM25 + Vector)", state.message, len(docs) > 0, len(docs),
                      details={"search_type": "hybrid", "intent": state.intent})
        
        # Get search performance metrics
        if hasattr(retr, 'get_last_search_metrics'):
            search_metrics = retr.get_last_search_metrics()
            state.search_metrics.update(search_metrics)
            logger.info(f"📊 [orchestrator.py] Search metrics collected: {search_metrics}")
            
            # Pass metrics to debug reporter
            set_search_metrics(search_metrics)
    except Exception as search_error:
        logger.error(f"❌ [orchestrator.py] Hybrid search failed: {search_error}")
        add_debug_step("Hybrid Search (BM25 + Vector)", state.message, False, 0,
                      f"Search failed: {str(search_error)}")
        add_debug_issue(f"Hybrid search error: {str(search_error)}")
        docs = []
    
    search_time = (time.time() - search_start_time) * 1000
    logger.info(f"⏱️ [orchestrator.py] Core search completed in {search_time:.1f}ms")
    
    # Measure post-search processing time
    post_search_start_time = time.time()
    
    # Apply predicate filter for mixed_filters queries
    predicate_filter, booster_function = _mixed_filters_functions.get(state.session_id, (None, None))
    predicate_used = state.intent == "mixed_filters" and predicate_filter is not None
    
    # Update debug info about predicate filter usage
    set_debug_query_processing(predicate_filter=predicate_used)
    
    if predicate_used:
        logger.info("📎 [orchestrator.py] Applying mixed_filters predicate filter...")
        logger.info(f"📎 [orchestrator.py] Query: '{state.message}'")
        logger.info(f"📎 [orchestrator.py] Original docs count: {len(docs)}")
        
        filtered_docs = []
        for i, doc in enumerate(docs):
            if hasattr(doc, 'metadata') and doc.metadata:
                filter_result = predicate_filter(doc.metadata)
                title = doc.metadata.get('title', 'Unknown')
                author = doc.metadata.get('author', 'Unknown')
                logger.info(f"    📋 Doc {i}: '{title}' -> predicate_filter: {filter_result}")
                
                # Record detailed filtering decision for debug
                action = "ACCEPTED" if filter_result else "REJECTED"
                reason = "Matches mixed_filters criteria" if filter_result else "Does not match mixed_filters criteria"
                add_debug_filtered_result({
                    "action": action,
                    "reason": reason,
                    "book_info": {"title": title, "author": author},
                    "score": "N/A",
                    "filter_type": "predicate"
                })
                
                if filter_result:
                    filtered_docs.append(doc)
                else:
                    logger.info(f"    ❌ Filtered out: '{title}' (predicate filter)")
        
        logger.info(f"    📋 Predicate filter: {len(filtered_docs)}/{len(docs)} documents passed")
        
        # Record predicate filtering step
        filter_criteria = list(state.filters.keys()) if state.filters else ["unknown"]
        add_debug_step("Predicate Filter", f"filter by {', '.join(filter_criteria)}", 
                      len(filtered_docs) > 0, len(filtered_docs),
                      f"Filtered from {len(docs)} to {len(filtered_docs)}")
        
        docs = filtered_docs
    
    logger.info(f"    📊 Retrieved {len(docs)} documents after similarity filtering")
    
    payload = _docs_to_payload(docs)
    
    # Apply booster for mixed_filters queries  
    if state.intent == "mixed_filters" and booster_function:
        logger.info("🚀 [orchestrator.py] Applying mixed_filters booster function...")
        
        # Add boost scores to payload
        for item in payload:
            boost_score = booster_function(item)
            item["_boost_score"] = boost_score
            if boost_score > 0:
                title = item.get('title', 'Unknown')
                logger.info(f"    🔥 Boost +{boost_score:.2f}: '{title}'")
        
        # Sort by boost score (higher first)
        payload.sort(key=lambda x: x.get("_boost_score", 0), reverse=True)

    logger.info("📋 [orchestrator.py] Converting documents to response format:")
    for i, doc in enumerate(payload):
        logger.info(f"    {i+1}. '{doc.get('title', 'Unknown')}' by {doc.get('author', 'Unknown')}")
        if doc.get('summary'):
            logger.info(f"       Summary: {doc['summary'][:80]}...")
        logger.info(f"       Genre: {doc.get('primary_genre', 'Unknown')}")

    # Apply filtering based on intent
    if state.intent == "negative_filter":
        payload = _apply_negative_filters(state, payload)
    elif state.intent == "year_range":
        payload = _apply_year_filters(state, payload)
    elif state.intent == "mixed_filters":
        # Additional filtering already applied via predicate_filter above
        # But we can still apply legacy filters if needed
        if state.exclude_filters.get("topics") or state.exclude_filters.get("genres"):
            logger.info("🚫 [orchestrator.py] Applying legacy negative filters for mixed_filters...")
            payload = _apply_negative_filters(state, payload)

    if state.intent == "author_title":
        logger.info("👤📖 [orchestrator.py] Enhanced Author-Title filtering with fuzzy matching and aliases:")
        
        f = state.filters or {}
        query_author = (f.get("author") or f.get("author_canonical") or "").lower().strip()
        query_title = (f.get("title") or f.get("title_canonical") or "").lower().strip()
        
        logger.info(f"    Looking for author: '{query_author}', title: '{query_title}'")
        
        # Enhanced matching with fuzzy logic
        matches = []
        potential_conflicts = []
        
        for p in payload:
            db_author = (p.get("author") or "").lower().strip()
            db_title = (p.get("title") or "").lower().strip()
            
            # Calculate similarities for main title
            author_similarity = fuzz.ratio(query_author, db_author) if query_author else 100
            title_similarity = fuzz.ratio(query_title, db_title) if query_title else 100
            
            # Also check partial matches
            author_partial = fuzz.partial_ratio(query_author, db_author) if query_author else 100
            title_partial = fuzz.partial_ratio(query_title, db_title) if query_title else 100
            
            # Check title_aliases for cross-language matching
            max_alias_similarity = 0
            max_alias_partial = 0
            if query_title and p.get("title_aliases"):
                try:
                    # Handle both JSON string and list formats
                    import json
                    aliases = p["title_aliases"]
                    if isinstance(aliases, str):
                        aliases = json.loads(aliases)
                    if isinstance(aliases, list):
                        for alias in aliases:
                            alias_lower = str(alias).lower().strip()
                            alias_sim = fuzz.ratio(query_title, alias_lower)
                            alias_partial = fuzz.partial_ratio(query_title, alias_lower)
                            max_alias_similarity = max(max_alias_similarity, alias_sim)
                            max_alias_partial = max(max_alias_partial, alias_partial)
                            logger.info(f"      Checking alias '{alias}': similarity {alias_sim}%, partial {alias_partial}%")
                except:
                    pass
            
            # Use best match between main title and aliases
            best_title_similarity = max(title_similarity, max_alias_similarity)
            best_title_partial = max(title_partial, max_alias_partial)
            
            logger.info(f"    Checking '{db_title}' by {db_author}:")
            logger.info(f"      Author similarity: {author_similarity}% (partial: {author_partial}%)")
            logger.info(f"      Title similarity: {title_similarity}% (partial: {title_partial}%)")
            logger.info(f"      Best title similarity (incl. aliases): {best_title_similarity}% (partial: {best_title_partial}%)")
            
            # Strong match criteria (using best title match including aliases)
            if (author_similarity > 80 or author_partial > 90) and (best_title_similarity > 70 or best_title_partial > 85):
                logger.info(f"    ✅ STRONG MATCH: '{db_title}' by {db_author} (aliases helped: {max_alias_similarity > title_similarity})")
                matches.append(p)
            # Title match but author mismatch - potential conflict (using best title match)
            elif (best_title_similarity > 70 or best_title_partial > 85) and author_similarity < 60:
                logger.info(f"    ⚠️ POTENTIAL CONFLICT: Title matches but author doesn't")
                potential_conflicts.append(p)
            # Decent author match with partial title match (using best title match)
            elif (author_similarity > 70 or author_partial > 85) and (best_title_similarity > 50 or best_title_partial > 65):
                logger.info(f"    🤔 PARTIAL MATCH: '{db_title}' by {db_author} (aliases helped: {max_alias_similarity > title_similarity})")
                matches.append(p)
        
        if matches:
            logger.info(f"    📋 Found {len(matches)} fuzzy matches")
            state.results = matches[:1]
        elif potential_conflicts:
            logger.info("⚠️ [orchestrator.py] CONFLICT DETECTED: Title match but author mismatch")
            conflict = potential_conflicts[0]
            state.need_clarify = True
            state.clarify_question = f"Did you mean '{conflict.get('title')}' by {conflict.get('author')}? (Author specified in query: {f.get('author')})"
            state.results = [conflict]
        else:
            logger.info("    ❌ No matches found with fuzzy matching")
            state.results = []
        
        return state

    # Check for conflicts in other intent types  
    state = _check_author_title_conflicts(state, payload)
    
    # Check mixed_filters parameter consistency
    if not state.need_clarify:
        state = _check_mixed_filters_consistency(state, payload)
    
    if not state.need_clarify:
        logger.info(f"📋 [orchestrator.py] General search results: {len(payload)} documents")
        state.results = payload[:10]
    
    post_search_time = (time.time() - post_search_start_time) * 1000
    logger.info(f"⏱️ [orchestrator.py] Post-search processing completed in {post_search_time:.1f}ms")
    
    # Record performance metrics
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["route_search_ms"] = execution_time
    state.performance_metrics["core_search_ms"] = search_time
    state.performance_metrics["post_search_processing_ms"] = post_search_time
    logger.info(f"⏱️ [orchestrator.py] Search routing completed in {execution_time:.1f}ms")
    
    return state

ANSWER_SYS = """You are a book assistant. Format responses as clear numbered lists.

For book recommendations, return a simple numbered list:
1. "Title" by Author - Brief description (1-2 sentences about why it matches the query)
2. "Title" by Author - Brief description

Guidelines:
- Use ONLY provided metadata, never invent information
- For genre/topic searches: focus on why each book matches the requested genre/topic  
- For author/title searches: list all matches found
- For ISBN searches: return exactly one book or "not found"
- Keep descriptions concise and relevant
- Always use Russian language for responses"""

def node_answer(state: ChatState) -> ChatState:
    start_time = time.time()
    
    logger.info("💬 [orchestrator.py] node_answer - Generating final response...")
    logger.info(f"    Purpose: Format search results into user-friendly answer")
    logger.info(f"    Input: {len(state.results)} results")
    
    if state.need_clarify and not state.results:
        logger.info("❓ [orchestrator.py] Returning clarification question")
        state.results = [{"clarify": state.clarify_question}]
        return state
    
    # Handle empty results
    if not state.results:
        logger.info("🚫 [orchestrator.py] No results found - generating 'not found' message")
        no_results_msg = f"Sorry, no books found matching your query '{state.message}'. Try modifying your query or uploading more books."
        state.results = [{"message": no_results_msg, "intent": state.intent}]
        logger.info("    📝 Generated: 'No results found' message")
        
        # Record final response for debug
        set_debug_final_response(no_results_msg)
        
        return state
    
    logger.info("🤖 [orchestrator.py] Using LLM to format response...")
    logger.info("    LLM Call: gpt-4o-mini to format search results")
    
    content = {"intent": state.intent, "filters": state.filters, "results": state.results}
    logger.info(f"    Sending to LLM: intent={state.intent}, {len(state.results)} results")
    
    # Log what we're sending to LLM for formatting
    logger.info("    📤 Data being sent to LLM for formatting:")
    for i, result in enumerate(state.results):
        if isinstance(result, dict):
            title = result.get('title', 'Unknown')
            author = result.get('author', 'Unknown')
            logger.info(f"      {i+1}. '{title}' by {author}")
    
    msg = llm.invoke([("system", ANSWER_SYS), ("user", f"{content}")]).content
    
    logger.info("    📝 LLM formatted the response successfully")
    logger.info(f"    📋 Final formatted response: {msg[:200]}..." if len(msg) > 200 else f"    📋 Final formatted response: {msg}")
    
    # Preserve original structured results AND add formatted message
    formatted_results = []
    logger.info(f"    🔍 Processing {len(state.results)} results for structured preservation:")
    for i, result in enumerate(state.results):
        logger.info(f"      Result {i}: keys={list(result.keys()) if isinstance(result, dict) else 'not dict'}")
        if isinstance(result, dict) and any(key in result for key in ['title', 'author', 'document_id']):
            # Keep structured result with all metadata
            enhanced_result = dict(result)
            enhanced_result["intent"] = state.intent
            formatted_results.append(enhanced_result)
            logger.info(f"        ✅ Preserved structured result: title={enhanced_result.get('title')}")
        else:
            logger.info(f"        ❌ Skipped: no structured keys")
    
    # Add formatted message as LAST result if we have structured results
    if formatted_results:
        formatted_results.append({"message": msg, "intent": state.intent})
        state.results = formatted_results
    else:
        # Fallback: only formatted message
        state.results = [{"message": msg, "intent": state.intent}]
    
    # Record final books and response for debug
    final_books = [result for result in (formatted_results or []) 
                   if isinstance(result, dict) and 'title' in result and 'author' in result]
    set_debug_final_books(final_books)
    set_debug_final_response(msg)
    
    # Record performance metrics
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["answer_generation_ms"] = execution_time
    logger.info(f"⏱️ [orchestrator.py] Answer generation completed in {execution_time:.1f}ms")
    
    # Pass performance metrics to debug reporter
    set_performance_metrics(state.performance_metrics)
    
    # Debug finalization will be handled in main.py at the end of request
    
    return state

builder = StateGraph(ChatState)
builder.add_node("detect_intent", node_detect_intent)
builder.add_node("route_search", node_route_search)
builder.add_node("answer", node_answer)
builder.set_entry_point("detect_intent")
builder.add_edge("detect_intent", "route_search")
builder.add_edge("route_search", "answer")
builder.add_edge("answer", END)
graph = builder.compile()
