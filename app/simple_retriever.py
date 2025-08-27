from typing import List, Dict, Any
import logging
import asyncio
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from .settings import CHROMA_DIR, OPENAI_MODEL_EMBED, OPENAI_API_KEY, CHUNKS_PER_BOOK_IN_CONTENT

logger = logging.getLogger(__name__)

class SimpleVectorRetriever:
    """Simple vector search without BM25 - only vector search on books and content collections"""
    
    def __init__(self):
        """Initialization of vector stores"""
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_MODEL_EMBED,
            api_key=OPENAI_API_KEY
        )
        
        # Connect to ChromaDB collections
        self.books_vectorstore = Chroma(
            collection_name="books",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )
        
        self.content_vectorstore = Chroma(
            collection_name="content", 
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR
        )
        
        logger.info("🔍 SimpleVectorRetriever initialized")

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Simple vector search with async requests and merge logic
        
        Args:
            query: search query
            k: final number of results (default 5)
            
        Returns:
            List[Dict]: list of results with merged data
        """
        logger.info(f"🔍 Simple vector search: '{query}' (k={k})")
        
        try:
            # Create embedding once for both collections
            logger.info("🔗 Creating query embedding...")
            embedding_start = time.time()
            query_embedding = await self.embeddings.aembed_query(query)
            embedding_time = time.time() - embedding_start
            logger.info(f"⚡ Embedding created in {embedding_time:.3f}s")
            
            # Async requests to both collections with ready embedding
            books_task = asyncio.create_task(
                self._search_books_collection_by_vector(query_embedding, k=5)  # Reduced k
            )
            content_task = asyncio.create_task(
                self._search_content_collection_by_vector(query_embedding, k=10)  # Reduced k
            )
            
            # Wait for results in parallel
            search_start = time.time()
            books_results, content_results = await asyncio.gather(books_task, content_task)
            search_time = time.time() - search_start
            logger.info(f"⚡ ChromaDB search in {search_time:.3f}s")
            
            # Merge logic: combine master records with best chunks
            merge_start = time.time()
            merged_results = self._merge_books_and_content(books_results, content_results)
            merge_time = time.time() - merge_start
            logger.info(f"⚡ Merge completed in {merge_time:.3f}s")
            
            # Sort by relevance and limit
            final_results = sorted(merged_results, key=lambda x: x.get('score', 0), reverse=True)[:k]
            
            logger.info(f"✅ Total: books={len(books_results)}, content={len(content_results)} → merged={len(merged_results)} → final={len(final_results)}")
            
            # DEBUG: Show what we return to orchestrator
            if final_results:
                first_result = final_results[0]
                logger.info(f"🔍 STAGE 1 - SimpleVectorRetriever output:")
                logger.info(f"   First result keys: {list(first_result.keys())}")
                logger.info(f"   Title: {first_result.get('title', 'N/A')}")
                logger.info(f"   Author: {first_result.get('author', 'N/A')}")
                content_preview = first_result.get('content', '')[:500] + '...' if len(first_result.get('content', '')) > 500 else first_result.get('content', '')
                logger.info(f"   Content preview (500 chars): {content_preview}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in simple vector search: {e}")
            return []

    async def _search_books_collection_by_vector(self, query_embedding: List[float], k: int = 8) -> List[Dict[str, Any]]:
        """Async search in books collection with ready embedding"""
        try:
            logger.info(f"📚 Search in books collection with ready embedding (k={k})")
            
            # Execute vector search with ready embedding
            loop = asyncio.get_event_loop()
            results_with_scores = await loop.run_in_executor(
                None, 
                lambda: self.books_vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
            )
            
            processed_results = []
            documents = results_with_scores['documents'][0]
            metadatas = results_with_scores['metadatas'][0]
            distances = results_with_scores['distances'][0]
            
            for i, (doc_content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Convert cosine distance to similarity (like in old retrievers.py)
                # Fix distance-to-similarity: cosine distance [0,2] → similarity [0,1]
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
                result = {
                    'content': doc_content,
                    'metadata': metadata.copy() if metadata else {},
                    'score': similarity,
                    'collection': 'books',
                    'document_id': metadata.get('document_id') if metadata else None,
                    'title': metadata.get('title', 'Unknown') if metadata else 'Unknown',
                    'author': metadata.get('author', 'Unknown') if metadata else 'Unknown'
                }
                processed_results.append(result)
                
            logger.info(f"📊 Books: found {len(processed_results)} master records")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error searching in books collection: {e}")
            return []

    async def _search_content_collection_by_vector(self, query_embedding: List[float], k: int = 15) -> List[Dict[str, Any]]:
        """Async search in content collection with ready embedding"""
        try:
            logger.info(f"📄 Search in content collection with ready embedding (k={k})")
            
            # Execute vector search with ready embedding
            loop = asyncio.get_event_loop()
            results_with_scores = await loop.run_in_executor(
                None,
                lambda: self.content_vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
            )
            
            # Group by books
            documents = results_with_scores['documents'][0]
            metadatas = results_with_scores['metadatas'][0]
            distances = results_with_scores['distances'][0]
            
            book_groups = {}
            for i, (doc_content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Get book identifier
                doc_id = metadata.get('document_id') if metadata else None
                title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
                author = metadata.get('author', '') if metadata else ''
                
                # Create unique book key
                if doc_id:
                    book_key = doc_id
                elif title and author:
                    book_key = f"{title}|{author}"
                else:
                    book_key = title
                
                if book_key not in book_groups:
                    book_groups[book_key] = []
                
                # Convert cosine distance to similarity (like in old retrievers.py)
                similarity = max(0.0, 1.0 - (distance / 2.0))
                book_groups[book_key].append({
                    'doc_content': doc_content,
                    'metadata': metadata,
                    'score': similarity,
                    'document_id': doc_id,
                    'title': title,
                    'author': author
                })
            
            # Select best chunks for each book
            processed_results = []
            for book_key, chunks in book_groups.items():
                # Sort by relevance and take maximum CHUNKS_PER_BOOK_IN_CONTENT
                best_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)[:CHUNKS_PER_BOOK_IN_CONTENT]
                
                for chunk_data in best_chunks:
                    result = {
                        'content': chunk_data['doc_content'],
                        'metadata': chunk_data['metadata'].copy() if chunk_data['metadata'] else {},
                        'score': chunk_data['score'],
                        'collection': 'content',
                        'document_id': chunk_data['document_id'],
                        'title': chunk_data['title'],
                        'author': chunk_data['author'],
                        'chunk_idx': chunk_data['metadata'].get('idx') if chunk_data['metadata'] else None
                    }
                    processed_results.append(result)
                
                logger.info(f"📖 '{chunks[0]['title']}': {len(chunks)} chunks → selected {len(best_chunks)}")
            
            logger.info(f"📊 Content: {len(book_groups)} books, total {len(processed_results)} chunks")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error searching in content collection: {e}")
            return []

    def _merge_books_and_content(self, books_results: List[Dict], content_results: List[Dict]) -> List[Dict]:
        """Simple merge: add best chunk to each book"""
        logger.info(f"🔗 Simple merge: {len(books_results)} books + {len(content_results)} content chunks")
        
        # Start with books results
        merged_results = []
        
        for book in books_results:
            # Copy book as is
            merged = book.copy()
            merged['collection'] = 'books_merged'
            
            # Find best matching chunk
            book_id = book.get('document_id')
            title = book.get('title', 'Unknown')
            
            matching_chunks = [c for c in content_results 
                              if c.get('document_id') == book_id or c.get('title') == title]
            
            # Add best chunk to content
            if matching_chunks:
                best_chunk = max(matching_chunks, key=lambda x: x.get('score', 0))
                merged['content'] += f"\n\n{best_chunk['content']}"
                logger.info(f"✅ '{title}': chunk added")
            
            merged_results.append(merged)
        
        # Add content results without books
        used_titles = {b.get('title') for b in books_results}
        for content in content_results:
            if content.get('title') not in used_titles:
                content_copy = content.copy()
                content_copy['collection'] = 'content_only'
                merged_results.append(content_copy)
        
        logger.info(f"🎯 Simple merge: {len(merged_results)} results")
        return merged_results

    def get_collection_stats(self) -> Dict[str, int]:
        """Get collection statistics"""
        try:
            books_count = self.books_vectorstore._collection.count()
            content_count = self.content_vectorstore._collection.count()
            
            return {
                "books": books_count,
                "content": content_count
            }
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {"books": 0, "content": 0}