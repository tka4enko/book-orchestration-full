from typing import List, Optional
import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from .settings import (
    CHROMA_DIR, BM25_K, VEC_BOOKS_K, VEC_CONTENT_K, RRF_K, OPENAI_MODEL_EMBED, OPENAI_API_KEY, 
    MIN_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLDS, BM25_THRESHOLDS, 
    LANGUAGE_THRESHOLD_MODIFIERS, MAX_CHUNKS_PER_BOOK,
    CHUNKS_PER_BOOK_IN_CONTENT, CONTENT_SEARCH_EXPAND_K
)
from .utils_isbn import normalize_isbn
from .metadata import detect_lang

logger = logging.getLogger(__name__)


def embeddings():
    return OpenAIEmbeddings(model=OPENAI_MODEL_EMBED, api_key=OPENAI_API_KEY)

def books_store():
    return Chroma(collection_name="books", persist_directory=CHROMA_DIR, embedding_function=embeddings())

def content_store():
    return Chroma(collection_name="content", persist_directory=CHROMA_DIR, embedding_function=embeddings())

def isbn_exact(query: str) -> List[Document]:
    logger.info("🔢 [retrievers.py] isbn_exact - Exact ISBN lookup")
    logger.info(f"    Purpose: Find exact book by ISBN number")
    logger.info(f"    Query: '{query}'")
    
    norm = normalize_isbn(query)
    if not norm: 
        logger.info("    ❌ No valid ISBN found in query")
        return []
    
    logger.info(f"    📖 Normalized ISBN: {norm}")
    logger.info("    🔍 Searching Chroma books collection...")
    
    b = books_store()._collection
    where = {"isbn13": norm["isbn13"]}
    found = b.get(where=where, include=["documents","metadatas"])
    
    docs = [Document(page_content=t, metadata=m) for t,m in zip(found.get("documents") or [], found.get("metadatas") or [])]
    
    logger.info(f"    ✅ Found {len(docs)} exact ISBN matches")
    return docs

def _bm25_from_store(store) -> Optional[BM25Retriever]:
    coll = store._collection
    try:
        cnt = coll.count()
    except Exception:
        cnt = 0
    if not cnt:
        return None
    raw = coll.get(include=["documents","metadatas"])
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    if not texts or not metas:
        return None
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
    if not docs:
        return None
    r = BM25Retriever.from_documents(docs)
    r.k = BM25_K
    return r

def bm25_books() -> Optional[BM25Retriever]:
    return _bm25_from_store(books_store())

def vector_books(k=VEC_BOOKS_K):
    return books_store().as_retriever(search_kwargs={"k": k})

def vector_content(k=VEC_CONTENT_K):
    return content_store().as_retriever(search_kwargs={"k": k})

def vector_books_with_scores(query: str, k=VEC_BOOKS_K):
    """Get documents with similarity scores to avoid recomputing embeddings"""
    logger.info(f"        🔍 [retrievers.py] Searching books vector store (k={k})...")
    logger.info(f"            Uses OpenAI embeddings API call for query")
    logger.info(f"            Query: '{query}'")
    
    store = books_store()
    results = store.similarity_search_with_score(query, k=k)
    
    logger.info(f"        📊 Books vector search: {len(results)} results found")
    for i, (doc, distance) in enumerate(results):
        title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        author = doc.metadata.get('author', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        similarity = max(0, 1 - (distance / 2))
        logger.info(f"            {i+1}. '{title}' by {author} - Distance: {distance:.3f}, Similarity: {similarity:.3f}")
    
    return results

def vector_content_with_scores(query: str, k=VEC_CONTENT_K):
    """Get documents with similarity scores to avoid recomputing embeddings"""
    logger.info(f"        🔍 [retrievers.py] Searching content vector store (k={k})...")
    logger.info(f"            Uses same embedding from previous call (cached)")
    logger.info(f"            Query: '{query}'")
    
    store = content_store()
    results = store.similarity_search_with_score(query, k=k)
    
    logger.info(f"        📄 Content vector search: {len(results)} results found")
    for i, (doc, distance) in enumerate(results):
        title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        chunk_idx = doc.metadata.get('idx', '?') if hasattr(doc, 'metadata') else '?'
        similarity = max(0, 1 - (distance / 2))
        preview = (doc.page_content or '')[:40] + '...' if hasattr(doc, 'page_content') and len(doc.page_content or '') > 40 else (doc.page_content or 'No content')
        logger.info(f"            {i+1}. '{title}' chunk {chunk_idx} - Distance: {distance:.3f}, Similarity: {similarity:.3f}")
        logger.info(f"                Preview: \"{preview}\"")
    
    return results

def diversified_content_search(query: str, chunks_per_book: int = None, total_limit: int = None, expand_k: int = None):
    """Get diversified content chunks - limited per book for better variety"""
    if chunks_per_book is None:
        chunks_per_book = CHUNKS_PER_BOOK_IN_CONTENT
    if total_limit is None:
        total_limit = VEC_CONTENT_K  
    if expand_k is None:
        expand_k = CONTENT_SEARCH_EXPAND_K
        
    logger.info(f"        🎨 [retrievers.py] Diversified content search...")
    logger.info(f"            Query: '{query}'")
    logger.info(f"            Strategy: max {chunks_per_book} chunks per book, expand k={expand_k} → limit to {total_limit}")
    
    store = content_store()
    raw_results = store.similarity_search_with_score(query, k=expand_k)
    
    logger.info(f"        📄 Raw content search: {len(raw_results)} results from expanded search")
    
    # Group results by book
    book_groups = {}
    for doc, distance in raw_results:
        # Get book identifier
        if hasattr(doc, 'metadata') and doc.metadata:
            book_id = doc.metadata.get('document_id')
            if not book_id:
                title = doc.metadata.get('title', 'unknown')
                author = doc.metadata.get('author', '')
                book_id = f"{title}|{author}" if author else title
        else:
            book_id = 'unknown'
        
        if book_id not in book_groups:
            book_groups[book_id] = []
        book_groups[book_id].append((doc, distance))
    
    logger.info(f"        📚 Found {len(book_groups)} unique books in content results")
    
    # Select best chunks from each book
    diversified_results = []
    for book_id, chunks in book_groups.items():
        # Get book title for logging
        book_title = 'Unknown'
        if chunks and hasattr(chunks[0][0], 'metadata') and chunks[0][0].metadata:
            book_title = chunks[0][0].metadata.get('title', 'Unknown')
            
        # Sort by distance (lower = better) and take best chunks
        best_chunks = sorted(chunks, key=lambda x: x[1])[:chunks_per_book]
        diversified_results.extend(best_chunks)
        
        logger.info(f"            📖 '{book_title}': {len(chunks)} chunks available → selected {len(best_chunks)} best")
    
    # Sort final results by relevance and limit
    final_results = sorted(diversified_results, key=lambda x: x[1])[:total_limit]
    
    logger.info(f"        🎯 Diversified results: {len(final_results)}/{len(raw_results)} chunks kept")
    
    # Log final selection with details
    for i, (doc, distance) in enumerate(final_results):
        title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
        chunk_idx = doc.metadata.get('idx', '?') if hasattr(doc, 'metadata') and doc.metadata else '?'
        similarity = max(0, 1 - (distance / 2))
        preview = (doc.page_content or '')[:40] + '...' if hasattr(doc, 'page_content') and len(doc.page_content or '') > 40 else (doc.page_content or 'No content')
        logger.info(f"            {i+1}. '{title}' chunk {chunk_idx} - Distance: {distance:.3f}, Similarity: {similarity:.3f}")
        logger.info(f"                Preview: \"{preview}\"")
    
    return final_results

class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return []

class OptimizedThresholdRetriever:
    def __init__(self, threshold: float = MIN_SIMILARITY_THRESHOLD):
        self.threshold = threshold
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.invoke(query)
    
    def _get_adjusted_threshold(self, base_threshold: float, query_lang: str, doc_lang: str) -> float:
        """Get threshold adjusted for language match/mismatch"""
        if query_lang == "unknown" or doc_lang == "unknown":
            modifier = LANGUAGE_THRESHOLD_MODIFIERS.get("unknown_language", -0.05)
            adjusted = base_threshold + modifier
            logger.info(f"             🌐 Threshold adjustment (unknown): {base_threshold} + {modifier} = {adjusted:.3f}")
            return max(0, adjusted)
        elif query_lang == doc_lang:
            modifier = LANGUAGE_THRESHOLD_MODIFIERS.get("same_language", -0.1)
            adjusted = base_threshold + modifier
            logger.info(f"             🌐 Threshold adjustment ({query_lang}={doc_lang}): {base_threshold} + {modifier} = {adjusted:.3f}")
            return max(0, adjusted)
        else:
            modifier = LANGUAGE_THRESHOLD_MODIFIERS.get("different_language", 0.0)
            adjusted = base_threshold + modifier
            logger.info(f"             🌐 Threshold adjustment ({query_lang}≠{doc_lang}): {base_threshold} + {modifier} = {adjusted:.3f}")
            return max(0, adjusted)
    
    def _smart_deduplication(self, documents: List[Document]) -> List[Document]:
        """Apply smart deduplication - max chunks per book with priority system"""
        if not documents:
            return documents
            
        logger.info(f"    🧹 Smart deduplication: {len(documents)} docs → grouping by books...")
        
        # Group documents by book (using document_id or title+author)
        book_groups = {}
        for doc in documents:
            if not hasattr(doc, 'metadata') or not doc.metadata:
                continue
                
            # Try to get unique book identifier
            doc_id = doc.metadata.get('document_id')
            title = doc.metadata.get('title', '')
            author = doc.metadata.get('author', '')
            
            if doc_id:
                book_key = doc_id
            elif title and author:
                book_key = f"{title}|{author}"
            elif title:
                book_key = title
            else:
                book_key = "unknown"
            
            if book_key not in book_groups:
                book_groups[book_key] = []
            book_groups[book_key].append(doc)
        
        logger.info(f"    📚 Found {len(book_groups)} unique books")
        
        # For each book group, select best documents with priority system
        final_docs = []
        for book_key, book_docs in book_groups.items():
            # Get book title for logging
            book_title = "Unknown"
            if book_docs and hasattr(book_docs[0], 'metadata') and book_docs[0].metadata:
                book_title = book_docs[0].metadata.get('title', 'Unknown')
                
            logger.info(f"      📖 Book: '{book_title}' has {len(book_docs)} documents")
            
            # Separate master chunks and content chunks
            master_chunks = []
            content_chunks = []
            
            for doc in book_docs:
                if hasattr(doc, 'metadata') and doc.metadata and doc.metadata.get('is_master_chunk', False):
                    master_chunks.append(doc)
                else:
                    content_chunks.append(doc)
            
            logger.info(f"        📋 {len(master_chunks)} master chunks, {len(content_chunks)} content chunks")
            
            # Merge master chunk with best content chunk for richer context
            merged_doc = None
            
            if master_chunks:
                # Start with master chunk as base
                master_doc = master_chunks[0]
                merged_doc = Document(
                    page_content=master_doc.page_content if hasattr(master_doc, 'page_content') else '',
                    metadata=master_doc.metadata.copy() if hasattr(master_doc, 'metadata') else {}
                )
                logger.info(f"        📋 Using master chunk as base")
                
                # Add best content chunk for additional context
                if content_chunks:
                    best_content = content_chunks[0]  # First is usually most relevant
                    
                    # Merge page content
                    if hasattr(best_content, 'page_content') and best_content.page_content:
                        if merged_doc.page_content:
                            merged_doc.page_content += "\n\n" + best_content.page_content
                        else:
                            merged_doc.page_content = best_content.page_content
                    
                    # Add content chunk info to metadata
                    if hasattr(best_content, 'metadata') and best_content.metadata:
                        chunk_idx = best_content.metadata.get('idx', '?')
                        merged_doc.metadata['merged_with_chunk'] = chunk_idx
                        logger.info(f"        ➕ Merged with content chunk {chunk_idx}")
                    
                    logger.info(f"        ✅ Created merged document with enriched context")
                else:
                    logger.info(f"        ✅ Using master chunk only (no content chunks)")
                    
            elif content_chunks:
                # No master chunk, use best content chunk
                merged_doc = content_chunks[0]
                logger.info(f"        ✅ Using best content chunk (no master available)")
            
            if merged_doc:
                final_docs.append(merged_doc)
                logger.info(f"        📊 Final: 1 merged document created for '{book_title}'")
        
        logger.info(f"    🎯 Deduplication result: {len(final_docs)}/{len(documents)} documents kept")
        return final_docs
    
    def invoke(self, query: str, intent: str = "free_text") -> List[Document]:
        """Smart hybrid search: BM25 first, then vector search with exclusions"""
        # Get intent-based threshold
        similarity_threshold = SIMILARITY_THRESHOLDS.get(intent, self.threshold)
        
        logger.info("🎯 [retrievers.py] OptimizedThresholdRetriever.invoke - Smart hybrid search pipeline")
        logger.info(f"    Purpose: BM25 first → Vector with exclusions, filter by similarity >= {similarity_threshold}")
        logger.info(f"    Query: '{query}', Intent: '{intent}', Threshold: {similarity_threshold}")
        
        # Save query for length checking in penalty method
        self._current_query = query
        
        # Detect query language
        query_stripped = query.strip()
        
        # Only skip language detection for very short queries or digits
        if len(query_stripped) < 3 or query_stripped.isdigit():
            query_lang = "unknown"
            logger.info(f"    🌐 Using 'unknown' language (too short or numeric)")
        else:
            query_lang = detect_lang(query, fallback="unknown")
            logger.info(f"    🌐 Query language detected: '{query_lang}'")
        
        try:
            # STEP 1: BM25 keyword search first (fastest, most precise for keywords)
            logger.info("    🔤 Step 1: BM25 keyword search (priority pipeline)...")
            bm25_retriever = bm25_books()
            bm25_docs = []
            bm25_doc_ids = set()  # Track found document IDs
            
            if bm25_retriever:
                try:
                    bm25_docs = bm25_retriever.invoke(query) if hasattr(bm25_retriever, 'invoke') else bm25_retriever._get_relevant_documents(query)
                    logger.info(f"        🎯 BM25 found {len(bm25_docs)} keyword matches")
                    
                    # Track document IDs for exclusion from vector search
                    for doc in bm25_docs:
                        if hasattr(doc, 'metadata') and doc.metadata and doc.metadata.get('document_id'):
                            bm25_doc_ids.add(doc.metadata['document_id'])
                    
                    logger.info(f"        📝 Will exclude {len(bm25_doc_ids)} BM25 documents from vector search")
                    
                except Exception as e:
                    logger.warning(f"        BM25 search failed: {e}")
            else:
                logger.info("        BM25 retriever not available (empty collection)")
            
            # STEP 2: Conditional vector search with exclusions
            skip_vector = self._should_skip_vector_search(intent, bm25_docs)
            books_results = []
            content_results = []
            
            if skip_vector:
                logger.info(f"    ⏭️  Step 2: Skipping vector search (BM25 sufficient for intent '{intent}')")
            else:
                logger.info("    📊 Step 2: Vector search in books collection (with BM25 exclusions)...")
                books_results = vector_books_with_scores(query, k=VEC_BOOKS_K)
                
                # Filter out documents already found by BM25
                books_results = [(doc, dist) for doc, dist in books_results 
                               if not (hasattr(doc, 'metadata') and doc.metadata 
                                      and doc.metadata.get('document_id') in bm25_doc_ids)]
                
                logger.info(f"        📊 Retrieved {len(books_results)} books after BM25 exclusions")
                
                logger.info("    📄 Step 3: Diversified vector search in content collection...")
                content_results = diversified_content_search(query)
                logger.info(f"        📄 Retrieved {len(content_results)} diversified content chunks")
            
            # STEP 4: Process results with priority system (BM25 first, vector supplements)
            logger.info("    🔍 Step 4: Processing results with priority system...")
            
            # Priority 1: Process BM25 results first (highest priority)
            filtered_docs = []
            bm25_threshold = BM25_THRESHOLDS.get(intent, 0.4)
            bm25_scores = self._get_bm25_scores(query, bm25_docs) if bm25_docs else []
            
            logger.info(f"        🏆 Priority 1: BM25 keyword results (intent: {intent}, threshold: {bm25_threshold}):")
            bm25_passed = 0
            
            for i, doc in enumerate(bm25_docs):
                title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                author = doc.metadata.get('author', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                
                # Get BM25 score for this document
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.5
                
                logger.info(f"          {i+1}. '{title}' by {author}")
                logger.info(f"             BM25 Score: {bm25_score:.3f} (threshold: {bm25_threshold})")
                
                # Analyze matching tokens
                matching_tokens = self._analyze_matching_tokens(query, doc)
                if matching_tokens:
                    logger.info(f"             Matching tokens: {matching_tokens}")
                
                # Apply BM25 score threshold
                if bm25_score >= bm25_threshold:
                    score_reason = self._explain_bm25_score(bm25_score)
                    logger.info(f"             ✅ PRIORITY ADDED (BM25 {score_reason})")
                    filtered_docs.append(doc)
                    bm25_passed += 1
                else:
                    score_reason = self._explain_bm25_score(bm25_score)
                    logger.info(f"             ❌ FILTERED OUT (BM25 {score_reason})")
            
            logger.info(f"        📊 BM25 summary: {bm25_passed}/{len(bm25_docs)} results passed threshold")
            
            # Priority 2: Process vector results as supplements (excluded duplicates already filtered)
            books_passed = 0
            content_passed = 0
            
            if books_results:
                logger.info(f"        📚 Priority 2A: Vector books results (supplements to BM25):")
                for i, (doc, distance) in enumerate(books_results):
                    # Chroma distance is roughly 0-2, convert to 0-1 similarity
                    similarity = max(0, 1 - (distance / 2))
                    title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                    author = doc.metadata.get('author', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                    doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    
                    logger.info(f"          {i+1}. '{title}' by {author}")
                    logger.info(f"             Distance: {distance:.3f} → Similarity: {similarity:.3f}")
                    
                    # Get adjusted threshold based on language match
                    adjusted_threshold = self._get_adjusted_threshold(similarity_threshold, query_lang, doc_lang)
                    
                    if similarity >= adjusted_threshold:
                        logger.info(f"             ✅ SUPPLEMENT ADDED (similarity {similarity:.3f} >= adjusted threshold {adjusted_threshold:.3f})")
                        filtered_docs.append(doc)
                        books_passed += 1
                    else:
                        logger.info(f"             ❌ FILTERED OUT (similarity {similarity:.3f} < adjusted threshold {adjusted_threshold:.3f})")
                
                logger.info(f"        📊 Vector books summary: {books_passed}/{len(books_results)} passed threshold")
            
            if content_results:
                logger.info(f"        📄 Priority 2B: Vector content results (diversified):")
                for i, (doc, distance) in enumerate(content_results):
                    similarity = max(0, 1 - (distance / 2))
                    title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                    chunk_idx = doc.metadata.get('idx', '?') if hasattr(doc, 'metadata') else '?'
                    doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') else 'unknown'
                    preview = (doc.page_content or '')[:60] + '...' if hasattr(doc, 'page_content') and len(doc.page_content or '') > 60 else (doc.page_content or 'No content')
                    
                    logger.info(f"          {i+1}. '{title}' chunk {chunk_idx}")
                    logger.info(f"             Preview: \"{preview}\"")
                    logger.info(f"             Distance: {distance:.3f} → Similarity: {similarity:.3f}")
                    
                    # Get adjusted threshold based on language match
                    adjusted_threshold = self._get_adjusted_threshold(similarity_threshold, query_lang, doc_lang)
                    
                    if similarity >= adjusted_threshold:
                        logger.info(f"             ✅ CONTENT ADDED (similarity {similarity:.3f} >= adjusted threshold {adjusted_threshold:.3f})")
                        filtered_docs.append(doc)
                        content_passed += 1
                    else:
                        logger.info(f"             ❌ FILTERED OUT (similarity {similarity:.3f} < adjusted threshold {adjusted_threshold:.3f})")
                
                logger.info(f"        📊 Vector content summary: {content_passed}/{len(content_results)} passed threshold")
            
            # Summary of all sources
            total_vector_results = books_passed + content_passed
            logger.info(f"    📊 PIPELINE SUMMARY:")
            logger.info(f"        🏆 BM25 priority results: {bm25_passed}/{len(bm25_docs)} passed")
            logger.info(f"        📚 Vector supplement results: {total_vector_results}/{len(books_results) + len(content_results)} passed")
            logger.info(f"        📋 Total before deduplication: {len(filtered_docs)} documents")
            
            # Apply smart deduplication
            deduplicated_docs = self._smart_deduplication(filtered_docs)
            
            logger.info(f"    🎯 FINAL RESULTS: {len(deduplicated_docs)} documents after deduplication")
            return deduplicated_docs
            
        except Exception as e:
            logger.error(f"    ❌ Hybrid search failed: {e}")
            return []
    
    def _get_bm25_scores(self, query: str, bm25_docs: List[Document]) -> List[float]:
        """Get BM25 scores for documents (simplified estimation based on text overlap)"""
        # This is a simplified approach - real BM25 scores would need the retriever's internal state
        scores = []
        query_tokens = set(query.lower().split())
        
        for doc in bm25_docs:
            # Get ALL text from document for comprehensive analysis
            all_text_parts = []
            
            # Add page_content (main content)
            if hasattr(doc, 'page_content') and doc.page_content:
                all_text_parts.append(doc.page_content)
            
            # Add metadata text
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
                if meta.get('title'): all_text_parts.append(meta['title'])
                if meta.get('author'): all_text_parts.append(meta['author'])
                if meta.get('summary'): all_text_parts.append(meta['summary'])
                if meta.get('primary_genre'): all_text_parts.append(meta['primary_genre'])
                # Add secondary genres as well
                secondary = meta.get('secondary_genres')
                if secondary:
                    if isinstance(secondary, list):
                        all_text_parts.extend(secondary)
                    else:
                        all_text_parts.append(str(secondary))
            
            # Combine all text
            combined_text = ' '.join(all_text_parts).lower()
            doc_tokens = set(combined_text.split())
            
            # Find matching tokens
            matches = query_tokens & doc_tokens
            
            if matches:
                # Base score: ratio of matched query tokens
                base_score = len(matches) / len(query_tokens)
                
                # Boost for partial matches (useful for names like 'Орвелл' matching 'Orwell')
                partial_matches = 0
                for query_token in query_tokens:
                    if any(query_token in doc_token or doc_token in query_token 
                           for doc_token in doc_tokens if len(query_token) > 2):
                        partial_matches += 1
                
                partial_score = partial_matches / len(query_tokens) * 0.5  # Half weight for partial
                
                # Boost for exact author/title matches (higher weight)
                author_boost = 0.0
                title_boost = 0.0
                if hasattr(doc, 'metadata') and doc.metadata:
                    author = (doc.metadata.get('author') or '').lower()
                    title = (doc.metadata.get('title') or '').lower()
                    
                    for token in query_tokens:
                        if token in author:
                            author_boost += 0.3
                        if token in title:
                            title_boost += 0.2
                
                # Final score combination
                score = base_score + partial_score + author_boost + title_boost
                score = min(score, 1.0)  # Cap at 1.0
                
            else:
                score = 0.0
            
            scores.append(score)
        
        return scores
    
    def _analyze_matching_tokens(self, query: str, doc: Document) -> set:
        """Analyze which tokens from query match the document"""
        query_tokens = set(query.lower().split())
        
        # Get document text
        doc_text = ""
        if hasattr(doc, 'page_content') and doc.page_content:
            doc_text = doc.page_content
        elif hasattr(doc, 'metadata') and doc.metadata:
            meta = doc.metadata
            text_parts = []
            if meta.get('title'): text_parts.append(meta['title'])
            if meta.get('author'): text_parts.append(meta['author'])
            if meta.get('summary'): text_parts.append(meta['summary'])
            doc_text = ' '.join(text_parts)
        
        doc_tokens = set(doc_text.lower().split())
        matches = query_tokens & doc_tokens
        
        return matches
    
    def _explain_bm25_score(self, score: float) -> str:
        """Explain BM25 score level"""
        if score >= 0.7:
            return f"HIGH score {score:.3f} - strong keyword matches"
        elif score >= 0.4:
            return f"MEDIUM score {score:.3f} - some keyword matches"
        elif score >= 0.1:
            return f"LOW score {score:.3f} - weak keyword matches"
        else:
            return f"VERY LOW score {score:.3f} - minimal matches"
    
    def _should_skip_vector_search(self, intent: str, bm25_docs: List[Document]) -> bool:
        """Decide whether to skip vector search based on intent and BM25 results"""
        # Get BM25 threshold for this intent
        bm25_threshold = BM25_THRESHOLDS.get(intent, 0.4)
        
        # For exact match intents, if BM25 found good matches, skip vector
        exact_match_intents = ["isbn", "author", "title", "author_title"]
        if intent in exact_match_intents and bm25_docs:
            # Check if BM25 found high-quality matches
            high_quality_matches = 0
            query = getattr(self, '_current_query', '')
            bm25_scores = self._get_bm25_scores(query, bm25_docs) if bm25_docs else []
            
            for i, doc in enumerate(bm25_docs[:3]):  # Check top 3
                score = bm25_scores[i] if i < len(bm25_scores) else 0.5
                if score >= bm25_threshold:
                    high_quality_matches += 1
            
            if high_quality_matches > 0:
                logger.info(f"        🎯 Found {high_quality_matches} high-quality BM25 matches for '{intent}' - skipping vector search")
                return True
        
        # For semantic intents (genre, topic, free_text), always run vector search for better coverage
        semantic_intents = ["genre", "topic", "free_text", "clarify"]
        if intent in semantic_intents:
            logger.info(f"        🧠 Semantic intent '{intent}' - vector search needed for comprehensive results")
            return False
        
        # Default: run vector search
        logger.info(f"        ⚖️  Intent '{intent}' - running vector search for additional coverage")
        return False

def hybrid_rrf():
    retrievers: List = []
    b = bm25_books()
    if b is not None:
        retrievers.append(b)
    retrievers.append(vector_books())
    retrievers.append(vector_content())
    if not retrievers:
        return EmptyRetriever()
    if len(retrievers) == 1:
        return retrievers[0]
    return EnsembleRetriever(retrievers=retrievers, rrf_k=RRF_K)

def hybrid_with_rerank(intent: str = "free_text"):
    # Return the OptimizedThresholdRetriever with smart deduplication
    return OptimizedThresholdRetriever()
