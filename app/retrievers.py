from typing import List, Optional, Dict
import logging
import asyncio
import concurrent.futures
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
from .debug_reporter import set_debug_search_queries, add_debug_step

logger = logging.getLogger(__name__)

# Lazy NLTK imports and setup
_nltk_initialized = False

def _ensure_nltk():
    """Ensure NLTK data is downloaded and ready to use"""
    global _nltk_initialized
    if not _nltk_initialized:
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)  # New format for NLTK 3.9+
            nltk.download('stopwords', quiet=True)
            _nltk_initialized = True
        except Exception as e:
            logger.warning(f"NLTK setup failed: {e}, falling back to simple tokenization")

# Global cache for BM25Retriever to avoid recreating on every request
_bm25_retriever_cache = {}

# Global cache for text normalization to avoid repeated processing
_normalization_cache = {}

def clear_bm25_cache():
    """Clear BM25 retriever cache when documents are added/removed"""
    global _bm25_retriever_cache
    _bm25_retriever_cache.clear()
    logger.info("🔧 BM25 retriever cache cleared")

def clear_normalization_cache():
    """Clear normalization cache when needed"""
    global _normalization_cache
    _normalization_cache.clear()
    logger.info("🔧 Normalization cache cleared")

async def async_bm25_search(bm25_retriever, query: str, search_queries: List[str]) -> tuple:
    """Асинхронный BM25 поиск"""
    try:
        # Выполняем BM25 поиск в отдельном потоке
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_perform_bm25_search, bm25_retriever, query, search_queries)
            result = await loop.run_in_executor(None, future.result)
        return result
    except Exception as e:
        logger.error(f"Async BM25 search failed: {e}")
        return [], {}, 0.0

async def async_vector_books_search(query: str, bm25_doc_ids: set, k: int = 8) -> tuple:
    """Асинхронный векторный поиск в книгах"""
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_perform_vector_books_search, query, bm25_doc_ids, k)
            result = await loop.run_in_executor(None, future.result)
        return result
    except Exception as e:
        logger.error(f"Async vector books search failed: {e}")
        return [], 0.0

async def async_vector_content_search(query: str, k: int = 20) -> tuple:
    """Асинхронный векторный поиск в контенте"""
    try:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_perform_vector_content_search, query, k)
            result = await loop.run_in_executor(None, future.result)
        return result
    except Exception as e:
        logger.error(f"Async vector content search failed: {e}")
        return [], 0.0

def _perform_bm25_search(bm25_retriever, query: str, search_queries: List[str]) -> tuple:
    """Синхронный BM25 поиск для выполнения в отдельном потоке"""
    import time
    bm25_start_time = time.time()
    bm25_docs = []
    bm25_doc_ids = set()
    bm25_detailed_times = []
    
    for search_query in search_queries:
        query_start = time.time()
        try:
            docs = bm25_retriever.get_relevant_documents(search_query)
            query_time = (time.time() - query_start) * 1000
            bm25_detailed_times.append(("Single query", query_time))
            
            # Обрабатываем результаты
            for doc in docs:
                if doc.metadata.get('document_id') not in bm25_doc_ids:
                    bm25_docs.append(doc)
                    bm25_doc_ids.add(doc.metadata.get('document_id'))
                    
        except Exception as e:
            logger.warning(f"BM25 search failed for query '{search_query}': {e}")
    
    pure_bm25_time = (time.time() - bm25_start_time) * 1000
    return bm25_docs, bm25_doc_ids, pure_bm25_time

def _perform_vector_books_search(query: str, bm25_doc_ids: set, k: int = 8) -> tuple:
    """Синхронный векторный поиск в книгах для выполнения в отдельном потоке"""
    import time
    vector_books_start = time.time()
    
    try:
        store = books_store()
        results = store.similarity_search_with_score(query, k=k)
        
        # Фильтруем результаты, исключая найденные BM25
        filtered_results = []
        for doc, distance in results:
            if doc.metadata.get('document_id') not in bm25_doc_ids:
                filtered_results.append((doc, distance))
        
        vector_books_time = (time.time() - vector_books_start) * 1000
        return filtered_results, vector_books_time
        
    except Exception as e:
        logger.error(f"Vector books search failed: {e}")
        return [], 0.0

def _perform_vector_content_search(query: str, k: int = 20) -> tuple:
    """Синхронный векторный поиск в контенте для выполнения в отдельном потоке"""
    import time
    vector_content_start = time.time()
    
    try:
        store = content_store()
        # Расширяем поиск для диверсификации
        expanded_k = min(k * CONTENT_SEARCH_EXPAND_K, 50)
        results = store.similarity_search_with_score(query, k=expanded_k)
        
        # Диверсифицируем результаты
        diversified_results = _diversify_content_results(results, max_chunks_per_book=2, limit=k)
        
        vector_content_time = (time.time() - vector_content_start) * 1000
        return diversified_results, vector_content_time
        
    except Exception as e:
        logger.error(f"Vector content search failed: {e}")
        return [], 0.0

def _diversify_content_results(results: List[tuple], max_chunks_per_book: int = 2, limit: int = 6) -> List[tuple]:
    """Диверсифицирует результаты контента, ограничивая количество чанков на книгу"""
    import time
    book_chunks = {}
    
    for doc, distance in results:
        book_id = doc.metadata.get('book_id') or doc.metadata.get('document_id')
        if book_id not in book_chunks:
            book_chunks[book_id] = []
        book_chunks[book_id].append((doc, distance))
    
    diversified = []
    for book_id, chunks in book_chunks.items():
        # Берем лучшие чанки для каждой книги
        best_chunks = sorted(chunks, key=lambda x: x[1])[:max_chunks_per_book]
        diversified.extend(best_chunks)
    
    # Сортируем по расстоянию и ограничиваем общее количество
    diversified.sort(key=lambda x: x[1])
    return diversified[:limit]

def normalize_text(text: str, detected_lang: Optional[str] = None) -> List[str]:
    """Normalize text using NLTK: tokenize, lowercase, remove stop words"""
    # Check cache first
    if text in _normalization_cache:
        return _normalization_cache[text]
    
    _ensure_nltk()
    
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Use provided language or detect it
        if detected_lang is None:
            lang = detect_lang(text) if len(text.strip()) > 3 else "en"
        else:
            lang = detected_lang
        
        # Map language codes to NLTK stop words
        nltk_lang_map = {
            'en': 'english',
            'cs': 'english',  # Fallback to English for Czech (NLTK has limited language support)
            'ru': 'russian',
            'de': 'german',
            'fr': 'french',
            'es': 'spanish',
            'it': 'italian'
        }
        stop_lang = nltk_lang_map.get(lang, 'english')
        
        # Tokenize and normalize
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words(stop_lang))
        
        # Filter: keep only alphabetic tokens that aren't stop words and have length > 1
        filtered_tokens = [
            token for token in tokens 
            if token.isalpha() and len(token) > 1 and token not in stop_words
        ]
        
        # If all tokens were filtered out (e.g., "the the the"), return a special marker
        # to signal BM25 that this query should not match anything
        if not filtered_tokens:
            logger.warning(f"🔧 normalize_text: All tokens filtered out! Returning special empty marker")
            result = ["__EMPTY_QUERY__"]  # Special token that won't match any real content
        else:
            result = filtered_tokens
        
        # Cache the result
        _normalization_cache[text] = result
        return result
        
    except Exception as e:
        logger.warning(f"NLTK normalization failed: {e}, using simple tokenization")
        # Fallback to simple tokenization
        tokens = text.lower().split()
        # Simple English stop words fallback
        simple_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered = [token for token in tokens if token not in simple_stop_words and len(token) > 1]
        
        # Same check for fallback
        if not filtered:
            logger.warning(f"🔧 normalize_text fallback: All tokens filtered out! Returning special empty marker")
            result = ["__EMPTY_QUERY__"]
        else:
            result = filtered
        
        # Cache the fallback result too
        _normalization_cache[text] = result
        return result



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
    
    # Поиск по обоим ISBN форматам
    found_isbn13 = b.get(where={"isbn13": norm["isbn13"]}, include=["documents","metadatas"])
    found_isbn10 = b.get(where={"isbn10": norm["isbn10"]}, include=["documents","metadatas"]) if norm["isbn10"] else {"documents": [], "metadatas": []}
    
    # Объединяем результаты, избегая дубликатов
    all_docs = (found_isbn13.get("documents") or []) + (found_isbn10.get("documents") or [])
    all_metas = (found_isbn13.get("metadatas") or []) + (found_isbn10.get("metadatas") or [])
    
    # Убираем дубликаты по document_id
    seen_ids = set()
    unique_docs = []
    unique_metas = []
    
    for doc, meta in zip(all_docs, all_metas):
        doc_id = meta.get('document_id')
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
            unique_metas.append(meta)
    
    found = {"documents": unique_docs, "metadatas": unique_metas}
    
    docs = [Document(page_content=t, metadata=m) for t,m in zip(found.get("documents") or [], found.get("metadatas") or [])]
    
    logger.info(f"    ✅ Found {len(docs)} exact ISBN matches")
    return docs

def _bm25_from_store(store) -> Optional[BM25Retriever]:
    # Create cache key based on collection count to detect changes
    coll = store._collection
    try:
        cnt = coll.count()
    except Exception:
        cnt = 0
        
    if not cnt:
        return None
        
    # Use collection name and count as cache key
    collection_name = getattr(coll, 'name', 'unknown')
    cache_key = f"{collection_name}_{cnt}"
    
    # Return cached retriever if available and current
    if cache_key in _bm25_retriever_cache:
        logger.info(f"🔧 Using cached BM25Retriever for {collection_name} ({cnt} docs)")
        return _bm25_retriever_cache[cache_key]
    
    # Create new BM25Retriever
    raw = coll.get(include=["documents","metadatas"])
    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    if not texts or not metas:
        return None
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metas)]
    if not docs:
        return None
        
    logger.info(f"🔧 Creating NEW BM25Retriever with normalize_text preprocessing for {len(docs)} documents")
    r = BM25Retriever.from_documents(docs, preprocess_func=normalize_text)
    r.k = BM25_K
    logger.info(f"🔧 BM25Retriever created successfully with k={BM25_K}")
    
    # Wrap the original get_relevant_documents method to handle empty queries
    original_get_relevant_documents = r._get_relevant_documents
    
    def _get_relevant_documents_with_empty_check(query: str, *, run_manager=None) -> List[Document]:
        # Use globals() to avoid scoping issues
        normalize_func = globals().get('normalize_text')
        if normalize_func:
            # Check if query normalizes to empty
            normalized_query = normalize_func(query)
            if normalized_query == ["__EMPTY_QUERY__"]:
                logger.info(f"🔧 BM25: Query '{query}' normalized to empty, returning no results")
                return []
        return original_get_relevant_documents(query, run_manager=run_manager)
    
    # Replace the method
    r._get_relevant_documents = _get_relevant_documents_with_empty_check
    
    # Cache the retriever
    _bm25_retriever_cache[cache_key] = r
    logger.info(f"🔧 BM25Retriever cached with key: {cache_key}")
    
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
        # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
        similarity = max(0.0, 1.0 - (distance / 2.0))
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
        # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
        similarity = max(0.0, 1.0 - (distance / 2.0))
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
        # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
        similarity = max(0.0, 1.0 - (distance / 2.0))
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
    
    def invoke(self, query: str, intent: str = "free_text", title_aliases: Optional[List[str]] = None, normalized_query_tokens: Optional[List[str]] = None) -> List[Document]:
        """Оптимизированная синхронная версия invoke"""
        return self._invoke_sync(query, intent, title_aliases, normalized_query_tokens)

    def _invoke_sync(self, query: str, intent: str = "free_text", title_aliases: Optional[List[str]] = None, normalized_query_tokens: Optional[List[str]] = None) -> List[Document]:
        """Оптимизированная синхронная версия invoke"""
        import time
        
        # Инициализация и нормализация
        logger.warning(f"🔍 DEBUG Step 1: About to normalize query: '{query}'")
        normalize_start = time.time()
        normalized_query_tokens = normalize_text(query)
        normalized_query_str = ' '.join(normalized_query_tokens)
        normalize_time = (time.time() - normalize_start) * 1000
        logger.warning(f"🔍 DEBUG Step 2: Query normalized: '{query}' → {normalized_query_tokens} (took {normalize_time:.1f}ms)")
        
        # Record normalized queries for debug
        set_debug_search_queries(
            bm25_query=f"Исходный: '{query}' → Нормализованный: '{normalized_query_str}'",
            vector_query=f"Исходный: '{query}' (векторный поиск без нормализации)"
        )
        
        # Check for empty query immediately
        if normalized_query_tokens == ["__EMPTY_QUERY__"]:
            logger.warning(f"🚫 Query '{query}' contains only stop words, returning empty results")
            return []
        
        # Get intent-based threshold
        similarity_threshold = SIMILARITY_THRESHOLDS.get(intent, self.threshold)
        
        logger.info("🎯 [retrievers.py] OptimizedThresholdRetriever.invoke - Optimized hybrid search pipeline")
        logger.info(f"    Purpose: BM25 first → Vector with exclusions, filter by similarity >= {similarity_threshold}")
        logger.info(f"    Query: '{query}', Intent: '{intent}', Threshold: {similarity_threshold}")
        if title_aliases:
            logger.info(f"    🌍 Cross-language aliases from parser: {title_aliases}")
            
        # Save query for length checking in penalty method
        self._current_query = query
        
        # Detect query language first
        lang_start = time.time()
        query_stripped = query.strip()
        
        # Only skip language detection for very short queries or digits
        if len(query_stripped) < 3 or query_stripped.isdigit():
            query_lang = "unknown"
            logger.info(f"    🌐 Using 'unknown' language (too short or numeric)")
        else:
            query_lang = detect_lang(query, fallback="unknown")
            logger.info(f"    🌐 Query language detected: '{query_lang}'")
        lang_time = (time.time() - lang_start) * 1000
        logger.info(f"    🌐 Language detection took: {lang_time:.1f}ms")
        
        # STEP 1: BM25 keyword search with detailed timing
        if normalized_query_tokens is None:
            # Pass the already detected language to avoid duplicate detection
            normalized_query_tokens = normalize_text(query, detected_lang=query_lang)
        
        logger.warning(f"🔍 DEBUG Step 3: Normalized tokens: {normalized_query_tokens}")
        bm25_retriever = bm25_books()
        logger.warning(f"🔍 DEBUG Step 4: BM25 retriever obtained: {bm25_retriever is not None}")
        bm25_docs = []
        bm25_doc_ids = set()  # Track found document IDs
        
        # Measure detailed BM25 timing
        bm25_start_time = time.time()
        bm25_detailed_times = []
        
        # Use only provided aliases (no DB extraction)
        search_queries = [query]
        if title_aliases:
            search_queries.extend(title_aliases)
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in search_queries:
                q_clean = q.lower().strip()
                if q_clean and q_clean not in seen:
                    unique_queries.append(q)
                    seen.add(q_clean)
            search_queries = unique_queries
            logger.info(f"    🔍 Using {len(search_queries)} query variants: {search_queries}")
        
        if bm25_retriever:
            try:
                # Search with all query variants if we have aliases
                if len(search_queries) > 1:
                    all_results = []
                    seen_doc_ids = set()
                    
                    for i, search_q in enumerate(search_queries):
                        variant_start = time.time()
                        logger.info(f"        🔍 BM25 variant #{i+1}: '{search_q}'")
                        
                        if hasattr(bm25_retriever, 'invoke'):
                            variant_docs = bm25_retriever.invoke(search_q)
                        else:
                            variant_docs = bm25_retriever._get_relevant_documents(search_q)
                        
                        variant_time = (time.time() - variant_start) * 1000
                        bm25_detailed_times.append(f"Variant {i+1} '{search_q}': {variant_time:.1f}ms")
                        logger.warning(f"        ⏱️ BM25 variant #{i+1} время: {variant_time:.1f}ms")
                        
                        # Add unique results only
                        added = 0
                        for doc in variant_docs:
                            doc_id = doc.metadata.get('document_id') if hasattr(doc, 'metadata') and doc.metadata else None
                            if doc_id and doc_id not in seen_doc_ids:
                                all_results.append(doc)
                                seen_doc_ids.add(doc_id)
                                added += 1
                            elif not doc_id:  # Fallback for docs without ID
                                all_results.append(doc)
                                added += 1
                        
                        logger.info(f"          Added {added} unique results from '{search_q}'")
                    
                    bm25_docs = all_results
                else:
                    # Single query search
                    single_start = time.time()
                    logger.warning(f"🔍 DEBUG Step 5: About to call BM25 with query: '{query}'")
                    bm25_docs = bm25_retriever.invoke(query) if hasattr(bm25_retriever, 'invoke') else bm25_retriever._get_relevant_documents(query)
                    single_time = (time.time() - single_start) * 1000
                    bm25_detailed_times.append(f"Single query '{query}': {single_time:.1f}ms")
                    logger.warning(f"🔍 DEBUG Step 6: BM25 returned {len(bm25_docs)} docs in {single_time:.1f}ms")
                
                # Measure results processing time
                processing_start = time.time()
                logger.info(f"        🎯 BM25 found {len(bm25_docs)} keyword matches")
                
                # Check if BM25 returned empty results due to empty query (all stop words)
                if len(bm25_docs) == 0:
                    # Check if query was normalized to empty (use already normalized tokens)
                    if normalized_query_tokens == ["__EMPTY_QUERY__"]:
                        logger.warning(f"🚫 Query '{query}' contains only stop words, returning empty results")
                        return []  # Return empty results immediately
                
                # Track document IDs for exclusion from vector search - BUT ONLY if they actually matched!
                scores_start = time.time()
                bm25_scores = self._get_bm25_scores(normalized_query_tokens, bm25_docs) if bm25_docs and normalized_query_tokens else []
                scores_time = (time.time() - scores_start) * 1000
                bm25_detailed_times.append(f"BM25 scores calculation: {scores_time:.1f}ms")
                logger.warning(f"        ⏱️ BM25 scores calculation: {scores_time:.1f}ms")
                
                actual_matches = 0
                
                for i, doc in enumerate(bm25_docs):
                    if hasattr(doc, 'metadata') and doc.metadata and doc.metadata.get('document_id'):
                        # Only exclude if BM25 actually found a match (score > 0.001 to filter false positives)
                        bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                        if bm25_score > 0.001:  # Real match, exclude from vector search
                            bm25_doc_ids.add(doc.metadata['document_id'])
                            actual_matches += 1
                
                processing_time = (time.time() - processing_start) * 1000
                bm25_detailed_times.append(f"Results processing: {processing_time:.1f}ms")
                
                logger.info(f"        📝 BM25 real matches: {actual_matches}/{len(bm25_docs)} → Will exclude {len(bm25_doc_ids)} documents from vector search")
                
            except Exception as e:
                logger.warning(f"        BM25 search failed: {e}")
        else:
            logger.info("        BM25 retriever not available (empty collection)")
        
        bm25_total_time = (time.time() - bm25_start_time) * 1000
        # Calculate pure BM25 search time (excluding processing)
        pure_bm25_time = sum([float(t.split(': ')[1].replace('ms', '')) for t in bm25_detailed_times if 'query' in t or 'scores' in t])
        logger.warning(f"⏱️ BM25 ДЕТАЛЬНОЕ ВРЕМЯ: {bm25_total_time:.1f}ms общее")
        for detail in bm25_detailed_times:
            logger.warning(f"   {detail}")
        logger.warning(f"   Pure BM25 search time: {pure_bm25_time:.1f}ms")
            
        # STEP 2: Conditional vector search with exclusions
        skip_vector = self._should_skip_vector_search(intent, bm25_docs, normalized_query_tokens)
        books_results = []
        content_results = []
        
        if skip_vector:
            logger.info(f"    ⏭️  Step 2: Skipping vector search (BM25 sufficient for intent '{intent}')")
            # Update debug with vector search being skipped
            set_debug_search_queries(
                vector_query=f"Векторный поиск ПРОПУЩЕН для intent '{intent}' (BM25 достаточно)"
            )
        else:
            vector_books_start_time = time.time()
            logger.info("    📊 Step 2: Vector search in books collection (with BM25 exclusions)...")
            # Update debug with actual vector query
            set_debug_search_queries(
                vector_query=f"Исходный: '{query}' (векторный поиск БЕЗ нормализации, с исключением {len(bm25_doc_ids)} BM25 результатов)"
            )
            books_results_raw = vector_books_with_scores(query, k=VEC_BOOKS_K)
            logger.info(f"        📊 Raw vector search found {len(books_results_raw)} books")
            
            # Log what will be excluded
            excluded_count = 0
            kept_results = []
            
            for doc, dist in books_results_raw:
                doc_id = doc.metadata.get('document_id') if hasattr(doc, 'metadata') and doc.metadata else None
                title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
                author = doc.metadata.get('author', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
                # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
                similarity = max(0.0, 1.0 - (dist / 2.0))
                
                if doc_id in bm25_doc_ids:
                    logger.info(f"        ❌ EXCLUDED: '{title}' by {author} (found by BM25, similarity: {similarity:.3f})")
                    excluded_count += 1
                else:
                    logger.info(f"        ✅ KEPT: '{title}' by {author} (not in BM25, similarity: {similarity:.3f})")
                    kept_results.append((doc, dist))
            
            books_results = kept_results
            vector_books_time = (time.time() - vector_books_start_time) * 1000
            logger.info(f"⏱️ [retrievers.py] Vector books search completed in {vector_books_time:.1f}ms")
            logger.info(f"        📊 Vector books summary: {excluded_count} excluded, {len(kept_results)} kept")
            
            # STEP 3: Diversified vector search in content collection
            vector_content_start_time = time.time()
            logger.info("    📄 Step 3: Diversified vector search in content collection...")
            content_results_raw = diversified_content_search(query, total_limit=20)
            vector_content_time = (time.time() - vector_content_start_time) * 1000
            logger.info(f"⏱️ [retrievers.py] Vector content search completed in {vector_content_time:.1f}ms")
            logger.info(f"        📄 Retrieved {len(content_results_raw)} diversified content chunks")
        
        # STEP 4: Processing results with priority system
        filtering_start_time = time.time()
        logger.info("    🔍 Step 4: Processing results with priority system...")
        
        # Priority 1: BM25 keyword results
        bm25_threshold = BM25_THRESHOLDS.get(intent, 0.1)
        logger.info(f"        🏆 Priority 1: BM25 keyword results (intent: {intent}, threshold: {bm25_threshold}):")
        logger.info(f"            🔍 BM25 found {len(bm25_docs)} documents to evaluate:")
        
        bm25_passed = 0
        for i, doc in enumerate(bm25_docs):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            if bm25_score >= bm25_threshold:
                bm25_passed += 1
        
        logger.info(f"        📊 BM25 summary: {bm25_passed}/{len(bm25_docs)} results passed threshold")
        
        # Priority 2A: Vector books results (supplements to BM25)
        logger.info(f"        📚 Priority 2A: Vector books results (supplements to BM25):")
        logger.info(f"            🔍 Vector books found {len(books_results)} documents to evaluate:")
        
        # Apply language-based threshold adjustments
        for doc, dist in books_results:
            title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
            author = doc.metadata.get('author', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
            doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
            
            # Apply language threshold adjustment
            adjusted_threshold = similarity_threshold + LANGUAGE_THRESHOLD_MODIFIERS.get(f"{query_lang}≠{doc_lang}", 0.0)
            logger.info(f"             🌐 Threshold adjustment ({query_lang}≠{doc_lang}): {similarity_threshold} + {LANGUAGE_THRESHOLD_MODIFIERS.get(f'{query_lang}≠{doc_lang}', 0.0)} = {adjusted_threshold}")
        
        # Priority 2B: Vector content results (diversified)
        logger.info(f"        📄 Priority 2B: Vector content results (diversified):")
        
        for i, (doc, dist) in enumerate(content_results):
            title = doc.metadata.get('title', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
            chunk_id = doc.metadata.get('chunk_id', 'Unknown') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown'
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            
            # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
            similarity = max(0.0, 1.0 - (dist / 2.0))
            
            logger.info(f"           {i+1}. '{title}' chunk {chunk_id}")
            logger.info(f"              Preview: \"{preview}\"")
            logger.info(f"              Distance: {dist:.3f} → Similarity: {similarity:.3f}")
            
            # Apply language threshold adjustment
            doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
            adjusted_threshold = similarity_threshold + LANGUAGE_THRESHOLD_MODIFIERS.get(f"{query_lang}≠{doc_lang}", 0.0)
            logger.info(f"              🌐 Threshold adjustment ({query_lang}≠{doc_lang}): {similarity_threshold} + {LANGUAGE_THRESHOLD_MODIFIERS.get(f'{query_lang}≠{doc_lang}', 0.0)} = {adjusted_threshold}")
            
            if similarity >= adjusted_threshold:
                logger.info(f"              ✅ CONTENT ADDED (similarity {similarity:.3f} >= adjusted threshold {adjusted_threshold:.3f})")
            else:
                logger.info(f"              ❌ FILTERED OUT (similarity {similarity:.3f} < adjusted threshold {adjusted_threshold:.3f})")
        
        content_passed = sum(1 for doc, dist in content_results if max(0.0, 1.0 - (dist / 2.0)) >= similarity_threshold)
        logger.info(f"        📊 Vector content summary: {content_passed}/{len(content_results)} passed threshold")
        
        # Collect all documents for processing
        all_docs = []
        
        # Add BM25 results that passed threshold
        for i, doc in enumerate(bm25_docs):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            if bm25_score >= bm25_threshold:
                all_docs.append(doc)
        
        # Add vector books results that passed similarity threshold
        for doc, dist in books_results:
            similarity = max(0.0, 1.0 - (dist / 2.0))
            doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
            adjusted_threshold = similarity_threshold + LANGUAGE_THRESHOLD_MODIFIERS.get(f"{query_lang}≠{doc_lang}", 0.0)
            
            if similarity >= adjusted_threshold:
                all_docs.append(doc)
        
        # Add vector content results that passed similarity threshold
        for doc, dist in content_results:
            similarity = max(0.0, 1.0 - (dist / 2.0))
            doc_lang = doc.metadata.get('language', 'unknown') if hasattr(doc, 'metadata') and doc.metadata else 'unknown'
            adjusted_threshold = similarity_threshold + LANGUAGE_THRESHOLD_MODIFIERS.get(f"{query_lang}≠{doc_lang}", 0.0)
            
            if similarity >= adjusted_threshold:
                all_docs.append(doc)
        
        # STEP 5: Smart deduplication
        logger.info(f"    🧹 Smart deduplication: {len(all_docs)} docs → grouping by books...")
        final_docs = self._deduplicate_documents(all_docs)
        
        filtering_time = (time.time() - filtering_start_time) * 1000
        logger.info(f"⏱️ [retrievers.py] Results filtering completed in {filtering_time:.1f}ms")
        
        # Log pipeline summary
        logger.info(f"    📊 PIPELINE SUMMARY:")
        logger.info(f"        🔤 BM25 found: {len(bm25_docs)} → passed threshold: {bm25_passed}")
        logger.info(f"        📚 Vector books found: {len(books_results_raw) if 'books_results_raw' in locals() else 0} → excluded by BM25: {len(bm25_doc_ids)} → kept: {len(books_results)}")
        logger.info(f"        📄 Vector content found: {len(content_results)} → passed similarity: {content_passed}")
        logger.info(f"        📋 Total documents for deduplication: {len(all_docs)} (BM25: {bm25_passed} + Vector: {len(all_docs) - bm25_passed})")
        logger.info(f"    🎯 FINAL RESULTS: {len(final_docs)} documents after deduplication")
        
        # Calculate and log quality metrics
        query_time = (time.time() - normalize_start) * 1000
        logger.info(f"    📊 QUALITY METRICS - Intent: {intent}, Results: {len(final_docs)}, Query time: {query_time/1000:.2f}s, Abstain: {len(final_docs) == 0}")
        
        # Prepare search metrics
        search_metrics = {
            "bm25_search_ms": pure_bm25_time,
            "vector_books_search_ms": vector_books_time if 'vector_books_time' in locals() else 0.0,
            "vector_content_search_ms": vector_content_time if 'vector_content_time' in locals() else 0.0,
            "results_filtering_ms": filtering_time,
            "initialization_ms": normalize_time + lang_time,
            "total_search_time_ms": (pure_bm25_time if 'pure_bm25_time' in locals() else 0.0) + 
                                  (vector_books_time if 'vector_books_time' in locals() else 0.0) + 
                                  (vector_content_time if 'vector_content_time' in locals() else 0.0) + 
                                  filtering_time
        }
        
        # Log initialization breakdown
        if 'normalize_time' in locals() or 'lang_time' in locals():
            logger.info(f"    📊 ИНИЦИАЛИЗАЦИЯ:")
            if 'normalize_time' in locals():
                logger.info(f"       ├─ Нормализация запроса: {normalize_time:.1f}ms")
            if 'lang_time' in locals():
                logger.info(f"       └─ Определение языка: {lang_time:.1f}ms")
            total_init = search_metrics["initialization_ms"]
            logger.info(f"       📊 Общее время инициализации: {total_init:.1f}ms")
        
        logger.info(f"    📊 SEARCH METRICS: {search_metrics}")
        logger.info(f"    📊 Retrieved {len(final_docs)} documents after similarity filtering")
        
        # Save metrics for debug reporter
        self._last_search_metrics = search_metrics
        
        return final_docs
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Применяет smart deduplication к списку документов."""
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
    
    def _get_bm25_scores(self, query_tokens: List[str], bm25_docs: List[Document]) -> List[float]:
        """Get BM25 scores for documents - using ONLY metadata to avoid duplication with enriched master_text"""
        scores = []
        # Use already normalized query tokens (passed from invoke)
        query_tokens = set(query_tokens)
        
        for doc in bm25_docs:
            # Use ONLY metadata for BM25 scoring (no page_content to avoid duplication)
            metadata_text_parts = []
            
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
                # Core searchable metadata fields
                if meta.get('title'): metadata_text_parts.append(meta['title'])
                if meta.get('author'): metadata_text_parts.append(meta['author'])
                if meta.get('summary'): metadata_text_parts.append(meta['summary'])
                if meta.get('primary_genre'): metadata_text_parts.append(meta['primary_genre'])
                
                # Add secondary genres
                secondary = meta.get('secondary_genres')
                if secondary:
                    if isinstance(secondary, list):
                        metadata_text_parts.extend(secondary)
                    elif isinstance(secondary, str):
                        try:
                            import json
                            parsed = json.loads(secondary)
                            if isinstance(parsed, list):
                                metadata_text_parts.extend(parsed)
                            else:
                                metadata_text_parts.append(str(secondary))
                        except:
                            metadata_text_parts.append(str(secondary))
                    else:
                        metadata_text_parts.append(str(secondary))
                
                # Add topics for keyword matching
                main_topics = meta.get('main_topics')
                if main_topics:
                    if isinstance(main_topics, list):
                        metadata_text_parts.extend(main_topics)
                    elif isinstance(main_topics, str):
                        try:
                            import json
                            parsed = json.loads(main_topics)
                            if isinstance(parsed, list):
                                metadata_text_parts.extend(parsed)
                            else:
                                metadata_text_parts.append(str(main_topics))
                        except:
                            metadata_text_parts.append(str(main_topics))
                
                mentioned = meta.get('mentioned_topics')
                if mentioned:
                    if isinstance(mentioned, list):
                        metadata_text_parts.extend(mentioned)
                    elif isinstance(mentioned, str):
                        try:
                            import json
                            parsed = json.loads(mentioned)
                            if isinstance(parsed, list):
                                metadata_text_parts.extend(parsed)
                            else:
                                metadata_text_parts.append(str(mentioned))
                        except:
                            metadata_text_parts.append(str(mentioned))
                
                # Add year as string for year-based searches
                if meta.get('year'):
                    metadata_text_parts.append(str(meta['year']))
            
            # Combine only metadata text (no page_content duplication) and normalize
            combined_text = ' '.join(metadata_text_parts)
            doc_tokens = set(normalize_text(combined_text))
            
            # Find matching tokens
            matches = query_tokens & doc_tokens
            
            if matches:
                # Base score: ratio of matched query tokens
                base_score = len(matches) / len(query_tokens)
                
                # Boost for partial matches (stricter criteria - same as _analyze_matching_tokens)
                partial_matches = 0
                for query_token in query_tokens:
                    if len(query_token) > 2:
                        for doc_token in doc_tokens:
                            if len(doc_token) > 2:
                                # Stricter partial matching: require significant overlap
                                if (len(query_token) >= 3 and query_token in doc_token) or \
                                   (len(doc_token) >= 3 and doc_token in query_token):
                                    # Additional check: substring must be at least 50% of smaller word
                                    min_len = min(len(query_token), len(doc_token))
                                    overlap_len = len(query_token) if query_token in doc_token else len(doc_token)
                                    if overlap_len >= min_len * 0.5:  # At least 50% overlap
                                        partial_matches += 1
                                        break  # Found match for this query token, move to next
                
                partial_score = partial_matches / len(query_tokens) * 0.5  # Half weight for partial
                
                # Boost for exact author/title matches (higher weight)
                author_boost = 0.0
                title_boost = 0.0
                if hasattr(doc, 'metadata') and doc.metadata:
                    # Normalize author and title the same way
                    author_tokens = set(normalize_text(doc.metadata.get('author') or ''))
                    title_tokens = set(normalize_text(doc.metadata.get('title') or ''))
                    
                    # Count exact token matches in author/title
                    author_matches = len(query_tokens & author_tokens)
                    title_matches = len(query_tokens & title_tokens)
                    
                    author_boost = author_matches * 0.3  # 0.3 per matching token
                    title_boost = title_matches * 0.2   # 0.2 per matching token
                
                # Final score combination
                score = base_score + partial_score + author_boost + title_boost
                score = min(score, 1.0)  # Cap at 1.0
                
            else:
                score = 0.0
            
            scores.append(score)
        
        return scores
    
    def _analyze_matching_tokens(self, query: str, normalized_query_tokens: List[str], doc: Document) -> dict:
        """Analyze which tokens from query match the document metadata (for BM25 analysis)"""
        # Use already normalized query tokens (passed from invoke)
        raw_query_tokens = set(query.lower().split())
        query_tokens = set(normalized_query_tokens)
        
        # Detect language for logging
        query_lang = detect_lang(query) if len(query.strip()) > 3 else "en"
        
        # Use ONLY metadata text (same as _get_bm25_scores does) - no page_content duplication
        metadata_text_parts = []
        
        if hasattr(doc, 'metadata') and doc.metadata:
            meta = doc.metadata
            # Core searchable metadata fields
            if meta.get('title'): metadata_text_parts.append(meta['title'])
            if meta.get('author'): metadata_text_parts.append(meta['author'])
            if meta.get('summary'): metadata_text_parts.append(meta['summary'])
            if meta.get('primary_genre'): metadata_text_parts.append(meta['primary_genre'])
            
            # Add secondary genres
            secondary = meta.get('secondary_genres')
            if secondary:
                if isinstance(secondary, list):
                    metadata_text_parts.extend(secondary)
                elif isinstance(secondary, str):
                    try:
                        import json
                        parsed = json.loads(secondary)
                        if isinstance(parsed, list):
                            metadata_text_parts.extend(parsed)
                        else:
                            metadata_text_parts.append(str(secondary))
                    except:
                        metadata_text_parts.append(str(secondary))
                else:
                    metadata_text_parts.append(str(secondary))
            
            # Add topics
            main_topics = meta.get('main_topics')
            if main_topics:
                if isinstance(main_topics, list):
                    metadata_text_parts.extend(main_topics)
                elif isinstance(main_topics, str):
                    try:
                        import json
                        parsed = json.loads(main_topics)
                        if isinstance(parsed, list):
                            metadata_text_parts.extend(parsed)
                        else:
                            metadata_text_parts.append(str(main_topics))
                    except:
                        metadata_text_parts.append(str(main_topics))
            
            mentioned = meta.get('mentioned_topics')
            if mentioned:
                if isinstance(mentioned, list):
                    metadata_text_parts.extend(mentioned)
                elif isinstance(mentioned, str):
                    try:
                        import json
                        parsed = json.loads(mentioned)
                        if isinstance(parsed, list):
                            metadata_text_parts.extend(parsed)
                        else:
                            metadata_text_parts.append(str(mentioned))
                    except:
                        metadata_text_parts.append(str(mentioned))
            
            # Add year as string
            if meta.get('year'):
                metadata_text_parts.append(str(meta['year']))
        
        # Combine only metadata text
        combined_text = ' '.join(metadata_text_parts).lower()
        doc_tokens = set(combined_text.split())
        
        # Find exact matches
        exact_matches = query_tokens & doc_tokens
        
        # Find partial/fuzzy matches (stricter criteria)
        partial_matches = set()
        for query_token in query_tokens:
            if query_token not in exact_matches and len(query_token) > 2:
                for doc_token in doc_tokens:
                    # Stricter partial matching: require significant overlap
                    if len(doc_token) > 2:
                        # Option 1: Substring must be at least 3 chars
                        if (len(query_token) >= 3 and query_token in doc_token) or \
                           (len(doc_token) >= 3 and doc_token in query_token):
                            # Additional check: substring must be at least 50% of smaller word
                            min_len = min(len(query_token), len(doc_token))
                            overlap_len = len(query_token) if query_token in doc_token else len(doc_token)
                            if overlap_len >= min_len * 0.5:  # At least 50% overlap
                                partial_matches.add(f"{query_token}≈{doc_token}")
        
        return {
            "exact": exact_matches,
            "partial": partial_matches,
            "query_tokens": query_tokens,
            "total_doc_tokens": len(doc_tokens)
        }
    
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
    
    def _should_skip_vector_search(self, intent: str, bm25_docs: List[Document], normalized_query_tokens: List[str] = None) -> bool:
        """Decide whether to skip vector search based on intent and BM25 results"""
        # Get BM25 threshold for this intent
        bm25_threshold = BM25_THRESHOLDS.get(intent, 0.4)
        
        # For exact match intents, if BM25 found good matches, skip vector
        exact_match_intents = ["isbn", "author", "title", "author_title"]
        if intent in exact_match_intents and bm25_docs:
            # Check if BM25 found high-quality matches
            high_quality_matches = 0
            query = getattr(self, '_current_query', '')
            bm25_scores = self._get_bm25_scores(normalized_query_tokens, bm25_docs) if bm25_docs and normalized_query_tokens else []
            
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
