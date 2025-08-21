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
    """Простой векторный поиск без BM25 - только векторный поиск по books и content коллекциям"""
    
    def __init__(self):
        """Инициализация векторных хранилищ"""
        self.embeddings = OpenAIEmbeddings(
            model=OPENAI_MODEL_EMBED,
            api_key=OPENAI_API_KEY
        )
        
        # Подключение к коллекциям ChromaDB
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
        Простой векторный поиск с асинхронными запросами и merge логикой
        
        Args:
            query: поисковый запрос
            k: итоговое количество результатов (по умолчанию 5)
            
        Returns:
            List[Dict]: список результатов с объединенными данными
        """
        logger.info(f"🔍 Простой векторный поиск: '{query}' (k={k})")
        
        try:
            # Создаем embedding один раз для обеих коллекций
            logger.info("🔗 Создание embedding запроса...")
            embedding_start = time.time()
            query_embedding = await self.embeddings.aembed_query(query)
            embedding_time = time.time() - embedding_start
            logger.info(f"⚡ Embedding создан за {embedding_time:.3f}с")
            
            # Асинхронные запросы к обеим коллекциям с готовым embedding
            books_task = asyncio.create_task(
                self._search_books_collection_by_vector(query_embedding, k=5)  # Уменьшили k
            )
            content_task = asyncio.create_task(
                self._search_content_collection_by_vector(query_embedding, k=10)  # Уменьшили k
            )
            
            # Ждем результаты параллельно
            search_start = time.time()
            books_results, content_results = await asyncio.gather(books_task, content_task)
            search_time = time.time() - search_start
            logger.info(f"⚡ ChromaDB поиск за {search_time:.3f}с")
            
            # Merge логика: объединяем master записи с лучшими чанками
            merge_start = time.time()
            merged_results = self._merge_books_and_content(books_results, content_results)
            merge_time = time.time() - merge_start
            logger.info(f"⚡ Merge выполнен за {merge_time:.3f}с")
            
            # Сортируем по релевантности и ограничиваем
            final_results = sorted(merged_results, key=lambda x: x.get('score', 0), reverse=True)[:k]
            
            logger.info(f"✅ Итого: books={len(books_results)}, content={len(content_results)} → merged={len(merged_results)} → final={len(final_results)}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка в простом векторном поиске: {e}")
            return []

    async def _search_books_collection_by_vector(self, query_embedding: List[float], k: int = 8) -> List[Dict[str, Any]]:
        """Асинхронный поиск в коллекции books с готовым embedding"""
        try:
            logger.info(f"📚 Поиск в books коллекции с готовым embedding (k={k})")
            
            # Выполняем векторный поиск с готовым embedding
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
                # Конвертируем cosine distance в similarity (как в старом retrievers.py)
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
                
            logger.info(f"📊 Books: найдено {len(processed_results)} master записей")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в books коллекции: {e}")
            return []

    async def _search_content_collection_by_vector(self, query_embedding: List[float], k: int = 15) -> List[Dict[str, Any]]:
        """Асинхронный поиск в коллекции content с готовым embedding"""
        try:
            logger.info(f"📄 Поиск в content коллекции с готовым embedding (k={k})")
            
            # Выполняем векторный поиск с готовым embedding
            loop = asyncio.get_event_loop()
            results_with_scores = await loop.run_in_executor(
                None,
                lambda: self.content_vectorstore._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
            )
            
            # Группируем по книгам
            documents = results_with_scores['documents'][0]
            metadatas = results_with_scores['metadatas'][0]
            distances = results_with_scores['distances'][0]
            
            book_groups = {}
            for i, (doc_content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Получаем идентификатор книги
                doc_id = metadata.get('document_id') if metadata else None
                title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
                author = metadata.get('author', '') if metadata else ''
                
                # Создаем уникальный ключ книги
                if doc_id:
                    book_key = doc_id
                elif title and author:
                    book_key = f"{title}|{author}"
                else:
                    book_key = title
                
                if book_key not in book_groups:
                    book_groups[book_key] = []
                
                # Конвертируем cosine distance в similarity (как в старом retrievers.py)
                similarity = max(0.0, 1.0 - (distance / 2.0))
                book_groups[book_key].append({
                    'doc_content': doc_content,
                    'metadata': metadata,
                    'score': similarity,
                    'document_id': doc_id,
                    'title': title,
                    'author': author
                })
            
            # Выбираем лучшие чанки для каждой книги
            processed_results = []
            for book_key, chunks in book_groups.items():
                # Сортируем по релевантности и берем максимум CHUNKS_PER_BOOK_IN_CONTENT
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
                
                logger.info(f"📖 '{chunks[0]['title']}': {len(chunks)} чанков → выбрано {len(best_chunks)}")
            
            logger.info(f"📊 Content: {len(book_groups)} книг, итого {len(processed_results)} чанков")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска в content коллекции: {e}")
            return []

    def _merge_books_and_content(self, books_results: List[Dict], content_results: List[Dict]) -> List[Dict]:
        """Простой merge: добавляем лучший chunk к каждой book"""
        logger.info(f"🔗 Простой merge: {len(books_results)} books + {len(content_results)} content chunks")
        
        # Начинаем с books результатов
        merged_results = []
        
        for book in books_results:
            # Копируем book как есть
            merged = book.copy()
            merged['collection'] = 'books_merged'
            
            # Ищем лучший подходящий chunk
            book_id = book.get('document_id')
            title = book.get('title', 'Unknown')
            
            matching_chunks = [c for c in content_results 
                              if c.get('document_id') == book_id or c.get('title') == title]
            
            # Добавляем лучший chunk в content
            if matching_chunks:
                best_chunk = max(matching_chunks, key=lambda x: x.get('score', 0))
                merged['content'] += f"\n\n{best_chunk['content']}"
                logger.info(f"✅ '{title}': добавлен chunk")
            
            merged_results.append(merged)
        
        # Добавляем content результаты без books
        used_titles = {b.get('title') for b in books_results}
        for content in content_results:
            if content.get('title') not in used_titles:
                content_copy = content.copy()
                content_copy['collection'] = 'content_only'
                merged_results.append(content_copy)
        
        logger.info(f"🎯 Простой merge: {len(merged_results)} результатов")
        return merged_results

    def get_collection_stats(self) -> Dict[str, int]:
        """Получить статистику коллекций"""
        try:
            books_count = self.books_vectorstore._collection.count()
            content_count = self.content_vectorstore._collection.count()
            
            return {
                "books": books_count,
                "content": content_count
            }
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {"books": 0, "content": 0}