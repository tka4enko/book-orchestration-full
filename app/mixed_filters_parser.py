"""
Mixed Filters Parser - Advanced query parsing with conflict detection
Handles complex queries with multiple slots and provides structured output
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .utils_isbn import normalize_isbn
from .metadata import canon

logger = logging.getLogger(__name__)

class ConflictInfo(BaseModel):
    has_conflict: bool = False
    reason: str = ""
    fields: List[str] = Field(default_factory=list)

class ClarifyInfo(BaseModel):
    should_ask: bool = False
    question: str = ""

class MixedFilters(BaseModel):
    isbn: Optional[str] = None
    title: Optional[str] = None
    title_alias: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    primary_genre: Optional[str] = None
    secondary_genres: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    exclude_topics: List[str] = Field(default_factory=list)
    exclude_genres: List[str] = Field(default_factory=list)
    year: Optional[Union[int, Dict[str, int]]] = None
    lang_filter: Optional[str] = None
    publisher_alias: Optional[str] = None
    request_type: Optional[str] = None  # "summary" | "biography" | None

class ParseResult(BaseModel):
    intent: str = "mixed_filters"
    filters: MixedFilters = Field(default_factory=MixedFilters)
    conflict: ConflictInfo = Field(default_factory=ConflictInfo)
    clarify: ClarifyInfo = Field(default_factory=ClarifyInfo)

class MixedFiltersParser:
    """Advanced parser for complex book queries with slot filling and conflict detection"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_CHAT,
            temperature=0,  # Deterministic
            top_p=0,
            max_tokens=512,
            api_key=OPENAI_API_KEY
        ) if OPENAI_API_KEY else None
        
        # JSON-deterministic LLM for strict parsing
        self.json_llm = ChatOpenAI(
            model=OPENAI_MODEL_CHAT,
            temperature=0,
            top_p=0,
            max_tokens=512,
            api_key=OPENAI_API_KEY,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        ) if OPENAI_API_KEY else None

    def should_use_mixed_filters(self, query: str) -> bool:
        """Determine if query needs mixed_filters parsing (≥2 slots OR author+title detected)"""
        
        # FIRST: Check if this is a simple negative_filter query that should NOT use mixed_filters
        simple_negative_patterns = [
            r'\b(?:but\s+not|except|excluding|without)\s+\w+',  # English
            r'\b(?:но\s+не|кроме|исключая|минус)\s+\w+',       # Russian  
            r'\b(?:ale\s+ne|kromě|bez)\s+\w+',                 # Czech
        ]
        
        has_simple_negation = any(re.search(pattern, query, re.IGNORECASE) for pattern in simple_negative_patterns)
        
        # Also check for simple NOT patterns (without complex DSL)
        has_not_pattern = re.search(r'\bNOT\s+\w+', query, re.IGNORECASE)
        has_dsl_syntax = ':' in query and any(keyword in query.lower() for keyword in ['title', 'author', 'genre', 'year', 'lang'])
        
        # If it's a simple negation without DSL syntax, let it go to negative_filter intent
        if (has_simple_negation or has_not_pattern) and not has_dsl_syntax:
            logger.info(f"    [should_use_mixed_filters] Query: '{query}'")
            logger.info(f"    [should_use_mixed_filters] Simple negative pattern detected - skipping mixed_filters")
            logger.info(f"    [should_use_mixed_filters] Result: False")
            return False
        
        # Continue with normal complex query detection
        slot_indicators = [
            r'isbn[:\s]*\d',                    # ISBN
            r'title[:\s]*["\']',               # title:"..."
            r'author[:\s]*["\']',              # author:"..."
            r'genre[:\s]*\w+',                 # genre:fantasy
            r'topics?[:\s]*[\(\[]',            # topics:(a,b,c)
            r'year[:\s]*\d{4}',                # year:2020
            r'lang(?:uage)?[:\s]*[a-z]{2}',    # lang:cs
            r'publisher[:\s]*\w+',             # publisher:penguin
            r'NOT\s+\w+',                      # NOT spirituality (only if has DSL)
            r'-\w+',                           # -topic
            r'"[^"]+"\s+by\s+',               # "Title" by Author
            r'isbn\s+[\d-]+\s+by\s+',         # ISBN 123 by Author
            r'\d{4}\.\.\d{4}',                # year range
            r'published\s+(after|before)',     # year indicators
        ]
        
        matches = sum(1 for pattern in slot_indicators if re.search(pattern, query, re.IGNORECASE))
        
        # Additional heuristics for complex queries
        has_quotes_and_by = '"' in query and ' by ' in query.lower()
        has_isbn_and_author = re.search(r'isbn\s*[:\s]*[\d-]+.*by\s+\w+', query, re.IGNORECASE)
        
        # NEW: Detect potential author+title patterns (natural language) - but be more specific
        words = query.strip().split()
        has_author_title_pattern = False
        if 2 <= len(words) <= 6:  # Reasonable length for author+title
            # Check for known author+title patterns (not just any capitalized words)
            # Must have clear indicators like "by", quotes, or explicit DSL syntax
            has_by_pattern = ' by ' in query.lower()
            has_quotes = '"' in query or "'" in query
            
            # Only trigger if we have explicit author/title indicators, not just multiple caps
            if has_by_pattern or has_quotes:
                capitalized_words = [w for w in words if w[0].isupper() and len(w) > 1 and not w.isdigit()]
                if len(capitalized_words) >= 2:
                    has_author_title_pattern = True
        
        logger.info(f"    [should_use_mixed_filters] Query: '{query}'")
        logger.info(f"    [should_use_mixed_filters] Pattern matches: {matches}")
        logger.info(f"    [should_use_mixed_filters] has_quotes_and_by: {has_quotes_and_by}")
        logger.info(f"    [should_use_mixed_filters] has_isbn_and_author: {has_isbn_and_author}")
        logger.info(f"    [should_use_mixed_filters] has_dsl_syntax: {has_dsl_syntax}")
        logger.info(f"    [should_use_mixed_filters] has_author_title_pattern: {has_author_title_pattern}")
        
        # More conservative: require explicit complexity indicators
        result = matches >= 2 or has_quotes_and_by or has_isbn_and_author or has_dsl_syntax or has_author_title_pattern
        logger.info(f"    [should_use_mixed_filters] Result: {result}")
        
        return result

    def preparse_query(self, query: str) -> Dict[str, Any]:
        """Pre-parse query using regex to extract obvious patterns"""
        result = {
            "isbn": None,
            "fields": {},
            "negatives": [],
            "quoted_phrases": [],
            "year_ranges": {},
            "dsl_detected": False
        }
        
        # Extract ISBN
        isbn_pattern = r'(?:isbn[:\s]*)?((?:\d{13}|\d{10}|\d{3}-\d{10}|\d{3}-\d{1}-\d{5}-\d{3}-\d{1})+)'
        isbn_match = re.search(isbn_pattern, query, re.IGNORECASE)
        if isbn_match:
            result["isbn"] = isbn_match.group(1).replace('-', '')
        
        # Extract field:value DSL patterns
        field_patterns = [
            (r'\btitle[:\s]*["\']([^"\']+)["\']', 'title'),
            (r'\bauthor[:\s]*["\']([^"\']+)["\']', 'author'),
            (r'genre[:\s]*([^\s,]+)', 'primary_genre'),
            (r'year[:\s]*(\d{4})', 'year'),
            (r'lang(?:uage)?[:\s]*([a-z]{2})', 'lang_filter'),
            (r'publisher[:\s]*([^\s,]+)', 'publisher_alias'),
        ]
        
        for pattern, field in field_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                result["fields"][field] = match.group(1).lower().strip()
                result["dsl_detected"] = True
        
        # Extract natural language patterns (e.g., "by Stephen King")
        natural_patterns = [
            (r'\bby\s+([A-Z][a-zA-Z\s]+(?:[A-Z][a-zA-Z]*)?)', 'author'),  # "by Stephen King"
            (r'"([^"]+)"\s+by\s+([A-Z][a-zA-Z\s]+)', 'title_and_author'),  # "Title" by Author
        ]
        
        for pattern, field in natural_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if field == 'author':
                    result["fields"]["author"] = match.group(1).strip().lower()
                elif field == 'title_and_author':
                    result["fields"]["title"] = match.group(1).strip().lower()
                    result["fields"]["author"] = match.group(2).strip().lower()
                result["dsl_detected"] = True
        
        # Extract topics with parentheses: topics:(feedback,growth)
        topics_match = re.search(r'topics?[:\s]*\(([^)]+)\)', query, re.IGNORECASE)
        if topics_match:
            topics = [t.strip().lower() for t in topics_match.group(1).split(',')]
            result["fields"]["topics"] = topics
            result["dsl_detected"] = True
        
        # Extract quoted phrases (likely titles)
        quoted_pattern = r'["\']([^"\']+)["\']'
        result["quoted_phrases"] = re.findall(quoted_pattern, query)
        
        # Extract NOT/negative tokens
        not_patterns = [
            r'NOT\s+([^\s,]+)',
            r'but\s+not\s+([^\s,]+)', 
            r'except\s+([^\s,]+)',
            r'exclude\s+([^\s,]+)',
            r'-([^\s,]+)',
            r'но\s+не\s+([^\s,]+)',  # Russian: но не
        ]
        
        for pattern in not_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            result["negatives"].extend([m.lower().strip() for m in matches])
        
        # Extract year ranges
        year_range_patterns = [
            (r'after\s+(\d{4})', 'gte'),
            (r'before\s+(\d{4})', 'lte'),  
            (r'from\s+(\d{4})\s+to\s+(\d{4})', 'range'),
            (r'(\d{4})\.\.(\d{4})', 'range'),
            (r'published\s+after\s+(\d{4})', 'gte'),
            (r'published\s+before\s+(\d{4})', 'lte'),
        ]
        
        for pattern, range_type in year_range_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if range_type == 'range':
                    result["year_ranges"]["gte"] = int(match.group(1))
                    result["year_ranges"]["lte"] = int(match.group(2))
                elif range_type == 'gte':
                    result["year_ranges"]["gte"] = int(match.group(1))
                elif range_type == 'lte':
                    result["year_ranges"]["lte"] = int(match.group(1))
        
        return result

    def parse_mixed_filters_rule_only(self, query: str) -> ParseResult:
        """Parse using only deterministic rules (no LLM)"""
        logger.info(f"🔧 [MixedFiltersParser] Rule-only parsing: '{query}'")
        
        preparsed = self.preparse_query(query)
        result = ParseResult()
        
        # Fill basic fields from regex
        if preparsed["isbn"]:
            result.filters.isbn = preparsed["isbn"]
            
        for field, value in preparsed["fields"].items():
            if field == "title":
                result.filters.title = value
            elif field == "author":  
                result.filters.author = value
            elif field == "primary_genre":
                result.filters.primary_genre = canon(value)
            elif field == "year":
                try:
                    result.filters.year = int(value)
                except:
                    pass
            elif field == "lang_filter":
                result.filters.lang_filter = value
            elif field == "publisher_alias":
                result.filters.publisher_alias = value
            elif field == "topics":
                result.filters.topics = value if isinstance(value, list) else [value]
        
        # Handle negatives
        if preparsed["negatives"]:
            result.filters.exclude_topics = preparsed["negatives"]
        
        # Handle year ranges
        if preparsed["year_ranges"]:
            if len(preparsed["year_ranges"]) == 1:
                if "gte" in preparsed["year_ranges"]:
                    result.filters.year = {"year_after": preparsed["year_ranges"]["gte"]}
                elif "lte" in preparsed["year_ranges"]:  
                    result.filters.year = {"year_before": preparsed["year_ranges"]["lte"]}
            else:
                result.filters.year = {
                    "year_from": preparsed["year_ranges"].get("gte"),
                    "year_to": preparsed["year_ranges"].get("lte")
                }
        
        # Handle quoted phrases as potential titles (only if not already used for other fields)
        if preparsed["quoted_phrases"] and not result.filters.title:
            # Don't use quoted phrases that are already assigned to other fields
            used_values = {
                result.filters.author,
                result.filters.primary_genre,
                result.filters.lang_filter,
                result.filters.publisher_alias
            }
            used_values = {v.lower() for v in used_values if v}
            
            available_phrases = [p for p in preparsed["quoted_phrases"] 
                               if p.lower() not in used_values]
            
            if available_phrases:
                result.filters.title = available_phrases[0].lower()
                result.filters.title_alias = available_phrases
        
        # Simple conflict detection
        result = self._detect_basic_conflicts(result, query)
        
        logger.info(f"    ✅ Rule-only parsing complete: {len([k for k, v in result.filters.model_dump().items() if v])} fields extracted")
        return result

    def parse_mixed_filters_with_llm(self, query: str) -> ParseResult:
        """Parse using LLM with strict JSON enforcement for deterministic slot filling"""
        if not self.json_llm:
            logger.warning("JSON LLM not available, falling back to rule-only parsing")
            return self.parse_mixed_filters_rule_only(query)
            
        logger.info(f"🤖 [MixedFiltersParser] Strict JSON LLM parsing: '{query}'")
        
        # Get rule-based baseline first
        rule_result = self.parse_mixed_filters_rule_only(query)
        preparsed = self.preparse_query(query)
        
        system_prompt = """You are a strict JSON slot-filling parser for book search queries.
IMPORTANT: You MUST return ONLY valid JSON matching the ParseResult schema.
No extra text, no explanations, ONLY JSON.

Extract fields from the query:
- isbn: ISBN number if found
- title: book title if mentioned  
- author: author name if mentioned
- primary_genre: main genre if specified
- topics: array of topic keywords
- exclude_topics: array of topics to exclude (NOT, -, but not)
- year: publication year or year range object
- lang_filter: language code (2 letters)
- has_conflict: true if parameters conflict
- should_ask: true if clarification needed

Rules:
- Normalize text to lowercase
- Extract ONLY explicitly mentioned information
- Never invent or guess data
- Use empty arrays [] for missing list fields
- Use null for missing single fields"""

        user_prompt = f"""Parse this query into JSON:

Query: "{query}"

Hints: {json.dumps(preparsed, indent=2)}

Return JSON only:"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use strict JSON LLM
            response = self.json_llm.invoke(messages)
            
            # Parse JSON with strict validation
            try:
                parsed_data = json.loads(response.content.strip())
                logger.info(f"    🔧 Raw JSON response: {json.dumps(parsed_data)[:200]}...")
            except json.JSONDecodeError as json_err:
                logger.error(f"    ❌ Invalid JSON from LLM: {json_err}")
                logger.error(f"    📄 Raw response: {response.content[:200]}...")
                raise
            
            # Create ParseResult with proper structure
            llm_result = ParseResult()
            
            # Map LLM fields to filters object
            if 'title' in parsed_data and parsed_data['title']:
                llm_result.filters.title = parsed_data['title']
            if 'author' in parsed_data and parsed_data['author']:
                llm_result.filters.author = parsed_data['author']
            if 'isbn' in parsed_data and parsed_data['isbn']:
                llm_result.filters.isbn = parsed_data['isbn']
            if 'primary_genre' in parsed_data and parsed_data['primary_genre']:
                llm_result.filters.primary_genre = parsed_data['primary_genre']
            if 'topics' in parsed_data and parsed_data['topics']:
                llm_result.filters.topics = parsed_data['topics']
            if 'exclude_topics' in parsed_data and parsed_data['exclude_topics']:
                llm_result.filters.exclude_topics = parsed_data['exclude_topics']
            if 'year' in parsed_data and parsed_data['year']:
                llm_result.filters.year = parsed_data['year']
            if 'lang_filter' in parsed_data and parsed_data['lang_filter']:
                llm_result.filters.lang_filter = parsed_data['lang_filter']
                
            # Handle conflict and clarification
            if parsed_data.get('has_conflict'):
                llm_result.conflict.has_conflict = True
            if parsed_data.get('should_ask'):
                llm_result.clarify.should_ask = True
            
            # Merge with rule-based result (LLM takes precedence)
            final_result = self._merge_parse_results(rule_result, llm_result)
            
            logger.info(f"    ✅ Strict JSON LLM parsing successful")
            return final_result
            
        except Exception as e:
            logger.error(f"    ❌ Strict JSON LLM parsing failed: {e}")
            logger.info(f"    🔄 Falling back to rule-only result")
            return rule_result

    def _merge_parse_results(self, rule_result: ParseResult, llm_result: ParseResult) -> ParseResult:
        """Merge rule-based and LLM results, with LLM taking precedence"""
        # Start with rule result as base
        merged = ParseResult(**rule_result.model_dump())
        
        # Override with LLM results where available
        llm_filters = llm_result.filters.model_dump()
        for field, value in llm_filters.items():
            if value is not None and value != [] and value != {}:
                setattr(merged.filters, field, value)
        
        # Use LLM conflict/clarify if detected
        if llm_result.conflict.has_conflict:
            merged.conflict = llm_result.conflict
        if llm_result.clarify.should_ask:
            merged.clarify = llm_result.clarify
            
        return merged

    def _detect_basic_conflicts(self, result: ParseResult, query: str) -> ParseResult:
        """Basic conflict detection using rules"""
        filters = result.filters
        
        # ISBN + different author conflict
        if filters.isbn and filters.author:
            # This would need actual DB lookup to detect conflict
            # For now, just flag as potential conflict
            result.conflict.has_conflict = True
            result.conflict.reason = "ISBN and author specified - need to verify match"
            result.conflict.fields = ["isbn", "author"]
            result.clarify.should_ask = True
            result.clarify.question = f"Please verify: does ISBN {filters.isbn} belong to a book by {filters.author}?"
        
        # Contradictory genre/title combinations (basic heuristics)
        if filters.title and filters.primary_genre:
            title_lower = filters.title.lower()
            genre_lower = filters.primary_genre.lower()
            
            # Example: "1984" with "self-help" genre 
            if "1984" in title_lower and "self-help" in genre_lower:
                result.conflict.has_conflict = True
                result.conflict.reason = "Title '1984' typically not self-help genre"
                result.conflict.fields = ["title", "primary_genre"]
                result.clarify.should_ask = True
                result.clarify.question = "Did you mean the dystopian novel '1984' or looking for self-help books?"
        
        return result

    def compile_post_filters(self, result: ParseResult) -> Tuple[Callable, Callable]:
        """Compile predicate filter and booster functions with fuzzy matching support"""
        filters = result.filters
        
        def predicate_filter(metadata: Dict[str, Any]) -> bool:
            """Hard filter - must match to be included"""
            # ISBN exact match
            if filters.isbn:
                meta_isbn = metadata.get('isbn13') or metadata.get('isbn10') or metadata.get('isbn')
                if meta_isbn:
                    normalized = normalize_isbn(meta_isbn)
                    if not normalized or normalized.get('isbn13') != filters.isbn:
                        return False
            
            # Language filter
            if filters.lang_filter:
                meta_lang = metadata.get('language', '').lower()
                if meta_lang != filters.lang_filter:
                    return False
            
            # Year filters
            if filters.year:
                meta_year = metadata.get('year')
                if meta_year:
                    try:
                        meta_year = int(meta_year)
                        if isinstance(filters.year, dict):
                            if filters.year.get('year_after') and meta_year <= filters.year['year_after']:
                                return False
                            if filters.year.get('year_before') and meta_year >= filters.year['year_before']:
                                return False
                            if filters.year.get('year_from') and meta_year < filters.year['year_from']:
                                return False
                            if filters.year.get('year_to') and meta_year > filters.year['year_to']:
                                return False
                        elif isinstance(filters.year, int) and meta_year != filters.year:
                            return False
                    except:
                        pass
            
            # Exclude topics/genres
            if filters.exclude_topics:
                meta_topics = []
                if metadata.get('main_topics'):
                    meta_topics.extend(self._normalize_list_field(metadata['main_topics']))
                if metadata.get('mentioned_topics'):
                    meta_topics.extend(self._normalize_list_field(metadata['mentioned_topics']))
                
                for exclude_topic in filters.exclude_topics:
                    if any(exclude_topic in topic.lower() for topic in meta_topics):
                        return False
            
            if filters.exclude_genres:
                meta_genres = []
                if metadata.get('primary_genre'):
                    meta_genres.append(metadata['primary_genre'].lower())
                if metadata.get('secondary_genres'):
                    meta_genres.extend(self._normalize_list_field(metadata['secondary_genres']))
                
                for exclude_genre in filters.exclude_genres:
                    if any(exclude_genre in genre.lower() for genre in meta_genres):
                        return False
            
            return True
        
        def booster_function(metadata: Dict[str, Any]) -> float:
            """Soft boost - increases relevance score"""
            boost = 0.0
            
            # Title match boost
            if filters.title:
                meta_title = (metadata.get('title') or '').lower()
                if filters.title.lower() in meta_title:
                    boost += 0.5
                # Check aliases
                for alias in filters.title_alias:
                    if alias.lower() in meta_title:
                        boost += 0.3
            
            # Author match boost
            if filters.author:
                meta_author = (metadata.get('author') or '').lower()  
                if filters.author.lower() in meta_author:
                    boost += 0.4
            
            # Genre match boost
            if filters.primary_genre:
                meta_genre = (metadata.get('primary_genre') or '').lower()
                if filters.primary_genre.lower() == meta_genre:
                    boost += 0.3
            
            # Topic match boost
            if filters.topics:
                meta_topics = []
                if metadata.get('main_topics'):
                    meta_topics.extend(self._normalize_list_field(metadata['main_topics']))
                if metadata.get('mentioned_topics'):
                    meta_topics.extend(self._normalize_list_field(metadata['mentioned_topics']))
                
                topic_matches = 0
                for filter_topic in filters.topics:
                    if any(filter_topic.lower() in topic.lower() for topic in meta_topics):
                        topic_matches += 1
                
                boost += (topic_matches / len(filters.topics)) * 0.2
            
            return boost
        
        return predicate_filter, booster_function

    def _normalize_list_field(self, field_value: Any) -> List[str]:
        """Normalize list field from metadata (handles JSON strings)"""
        if isinstance(field_value, list):
            return [str(x).strip() for x in field_value if x]
        elif isinstance(field_value, str):
            try:
                parsed = json.loads(field_value)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x]
                else:
                    return [str(field_value).strip()]
            except:
                return [str(field_value).strip()]
        else:
            return [str(field_value).strip()] if field_value else []

    def parse_query(self, query: str, use_llm: bool = True) -> ParseResult:
        """Main parsing method - determines strategy and returns ParseResult"""
        if not self.should_use_mixed_filters(query):
            # Not a mixed query, return simple result
            result = ParseResult()
            result.intent = "free_text"  # Will be overridden by orchestrator
            return result
        
        if use_llm and self.json_llm:
            return self.parse_mixed_filters_with_llm(query)
        else:
            return self.parse_mixed_filters_rule_only(query)