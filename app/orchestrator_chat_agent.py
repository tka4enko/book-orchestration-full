from typing import Optional, Dict, Any, List
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .simple_orchestrator import process_simple_search

logger = logging.getLogger(__name__)

class ChatAgentState(BaseModel):
    # Базовые поля
    session_id: str
    message: str
    
    # Поля для чат-агента
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    reply_message: Optional[str] = None
    chips: List[Dict[str, str]] = Field(default_factory=list)
    should_search: bool = False
    should_recommend: bool = False
    
    # Результаты (унифицированные для чата и поиска)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Метрики производительности
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

# LLM используется только для рекомендаций, не для роутинга
llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7, api_key=OPENAI_API_KEY)

def _format_chat_history(history: List[Dict[str, str]]) -> str:
    """Форматирует историю чата для промпта"""
    if not history:
        return "Пустая история"
    
    formatted = []
    for entry in history[-5:]:  # Последние 5 сообщений
        role = entry.get("role", "user")
        content = entry.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)

def node_chat_response_simple(state: ChatAgentState) -> ChatAgentState:
    """Простой чат ответ без LLM анализа"""
    start_time = time.time()
    
    logger.info("💬 [chat_agent] node_chat_response_simple - Простой чат ответ...")
    
    # Добавляем пользовательское сообщение в историю
    if not state.chat_history:
        state.chat_history = []
    
    state.chat_history.append({
        "role": "user",
        "content": state.message
    })
    
    try:
        # Простые ответы на основе ключевых слов
        message_lower = state.message.lower()
        
        if any(greeting in message_lower for greeting in ["привет", "здравствуй", "hello", "hi"]):
            reply = "Привет! Как дела? Чем могу помочь с книгами?"
            chips = [
                {"text": "Найти книгу", "action": "search"},
                {"text": "Порекомендуй что-то", "action": "recommend"}
            ]
        elif any(phrase in message_lower for phrase in ["не знаю", "скучно", "что делать"]):
            reply = "Понимаю! Может, почитаем что-то интересное? Какой жанр нравится?"
            chips = [
                {"text": "Детектив", "action": "search"}, 
                {"text": "Фантастика", "action": "search"},
                {"text": "Посоветуй сам", "action": "recommend"}
            ]
        else:
            reply = "Интересно! Расскажи больше - что именно тебя интересует?"
            chips = [
                {"text": "Найти конкретную книгу", "action": "search"},
                {"text": "Дай рекомендации", "action": "recommend"}
            ]
        
        # Формируем результат
        state.results = [{
            "message": reply,
            "intent": "chat",
            "chips": chips,
            "chat_mode": True
        }]
        
        # Добавляем в историю
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в chat_response_simple: {e}")
        state.results = [{
            "message": "Произошла ошибка. Чем могу помочь?",
            "intent": "error",
            "chips": [{"text": "Найти книги", "action": "search"}]
        }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["chat_response_simple_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Простой чат ответ за {execution_time:.1f}ms")
    
    return state

def node_chat_response(state: ChatAgentState) -> ChatAgentState:
    """Формирует финальный ответ для чат-режима"""
    start_time = time.time()
    
    logger.info("💬 [chat_agent] node_chat_response - Формирование чат ответа...")
    
    # Формируем структурированный ответ
    state.results = [{
        "message": state.reply_message,
        "intent": "chat",
        "chips": state.chips,
        "chat_mode": True
    }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["chat_response_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Чат ответ сформирован за {execution_time:.1f}ms")
    
    return state

async def node_recommendations(state: ChatAgentState) -> ChatAgentState:
    """Генерирует рекомендации на основе того что есть в ChromaDB"""
    logger.info("💡 [chat_agent] node_recommendations - Генерируем рекомендации из базы...")
    
    start_time = time.time()
    
    try:
        # Подключаемся к ChromaDB для получения статистики и реальных жанров
        from .simple_retriever import SimpleVectorRetriever
        
        retriever = SimpleVectorRetriever()
        stats = retriever.get_collection_stats()
        
        logger.info(f"📊 Статистика базы: books={stats['books']}, content={stats['content']}")
        
        if stats['books'] > 0:
            # Получаем реальные жанры из БД
            try:
                # Делаем запрос к коллекции книг для получения метаданных
                sample_results = await retriever.search("", k=20)  # Получаем книги для анализа жанров
                
                # Извлекаем уникальные жанры из метаданных
                genres = set()
                authors = set()
                
                logger.info(f"📊 Анализируем {len(sample_results)} результатов для извлечения жанров")
                
                for i, result in enumerate(sample_results):
                    metadata = result.get('metadata', {})
                    logger.info(f"📖 Результат {i}: metadata = {metadata}")
                    
                    # Жанры
                    book_genres = metadata.get('genres', [])
                    if isinstance(book_genres, list):
                        genres.update(book_genres)
                    elif isinstance(book_genres, str):
                        genres.add(book_genres)
                    
                    # Авторы для разнообразия
                    author = metadata.get('author', '')
                    if author and author != 'Unknown':
                        authors.add(author)
                
                logger.info(f"🎯 Найденные жанры: {list(genres)}")
                logger.info(f"👤 Найденные авторы: {list(authors)}")
                
                # Формируем чипы на основе реальных данных
                chips = []
                
                # Добавляем реальные жанры (максимум 4)
                real_genres = list(genres)[:4]
                for genre in real_genres:
                    chips.append({"text": genre, "action": "search"})
                
                # Добавляем популярных авторов (максимум 2)
                popular_authors = list(authors)[:2]
                for author in popular_authors:
                    chips.append({"text": author, "action": "search"})
                
                # Если нет данных, используем общие варианты
                if not chips:
                    chips = [
                        {"text": "Покажи всё что есть", "action": "search"},
                        {"text": "Случайная книга", "action": "search"}
                    ]
                
                reply = "Ниже я предоставил несколько интересных вариантов из моей коллекции:"
                
            except Exception as e:
                logger.error(f"❌ Ошибка получения жанров: {e}")
                # Fallback к простым вариантам
                reply = "Ниже я предоставил несколько вариантов из моей коллекции:"
                chips = [
                    {"text": "Покажи всё что есть", "action": "search"},
                    {"text": "Случайная книга", "action": "search"}
                ]
        else:
            reply = "К сожалению, база книг пока пуста. Попробуйте загрузить несколько книг сначала."
            chips = []
        
        # Формируем результат
        state.results = [{
            "message": reply,
            "intent": "recommendations",
            "chips": chips,
            "recommendation_mode": True
        }]
        
        # Добавляем в историю чата
        if not state.chat_history:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["recommendations_ms"] = execution_time
        
        logger.info(f"✅ [chat_agent] Рекомендации сгенерированы за {execution_time:.1f}ms")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [chat_agent] Ошибка генерации рекомендаций: {e}")
        
        # Возвращаем ошибку как результат
        error_message = "Извините, не могу сейчас дать рекомендации. Попробуйте поискать что-то конкретное!"
        
        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["recommendations_error_ms"] = execution_time
        
        return state

def route_message(state: ChatAgentState) -> str:
    """Определяет маршрут с помощью LLM"""
    message = state.message
    
    logger.info(f"🤖 [router] Анализ сообщения: '{message}'")
    
    try:
        # Быстрый LLM запрос для роутинга
        router_prompt = f"""Ты роутер для поиска книг. Анализируй запрос и определи достаточно ли информации для поиска.

SEARCH - если есть конкретная информация для поиска в базе данных книг (имена авторов, названия книг, жанры, ISBN)
RECOMMEND - если пользователь просит помочь выбрать книги или дать рекомендации
CLARIFY - если пользователь хочет найти книги, но НЕ указал достаточно информации для поиска (слишком общие фразы)
CHAT - если это обычное общение, не связанное с книгами или поиском

Подумай: есть ли в запросе достаточно конкретной информации для поиска книг в базе данных?

Сообщение: "{message}"

Ответ (одно слово):"""
        
        response = llm.invoke([("user", router_prompt)]).content.strip().upper()
        
        logger.info(f"🧠 [router] LLM ответ: '{response}'")
        
        # Маппинг ответов на nodes
        if "SEARCH" in response:
            logger.info(f"🔍 [router] Роутинг к поиску")
            return "simple_search"
        elif "RECOMMEND" in response:
            logger.info(f"💡 [router] Роутинг к рекомендациям") 
            return "recommendations"
        elif "CLARIFY" in response:
            logger.info(f"❓ [router] Роутинг к уточнению")
            return "clarify"
        else:
            logger.info(f"💬 [router] Роутинг к чату")
            return "chat_response"
            
    except Exception as e:
        logger.error(f"❌ [router] Ошибка LLM роутинга: {e}")
        # Fallback к простым правилам
        message_lower = message.lower()
        
        if any(cmd in message_lower for cmd in ["найди", "покажи", "ищу"]):
            return "simple_search"
        elif any(phrase in message_lower for phrase in ["посоветуй", "порекомендуй"]):
            return "recommendations"
        else:
            return "chat_response"

async def node_simple_search(state: ChatAgentState) -> ChatAgentState:
    """Выполняет простой поиск используя process_simple_search"""
    logger.info("🔍 [chat_agent] node_simple_search - Используем simple_orchestrator...")
    
    start_time = time.time()
    
    try:
        # Вызываем простой поиск
        search_result = await process_simple_search(state.session_id, state.message)
        
        # Используем готовый форматированный ответ из simple_orchestrator
        message = search_result.get('response', 'Результаты не найдены')
        
        # Добавляем в результаты
        state.results = [{
            "message": message,
            "intent": search_result.get('intent', 'search'),
            "search_mode": True,
            "original_response": search_result.get('response'),
            "raw_results": search_result.get('results', [])
        }]
        
        # Добавляем в историю чата
        if not state.chat_history:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant",
            "content": message
        })
        
        # Копируем метрики производительности
        if search_result.get('performance_metrics'):
            state.performance_metrics.update(search_result['performance_metrics'])
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["simple_search_ms"] = execution_time
        
        logger.info(f"✅ [chat_agent] Простой поиск завершен за {execution_time:.1f}ms")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [chat_agent] Ошибка простого поиска: {e}")
        
        # Возвращаем ошибку как результат
        error_message = "Извините, произошла ошибка при поиске. Попробуйте переформулировать запрос."
        
        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["simple_search_error_ms"] = execution_time
        
        return state

def node_clarify(state: ChatAgentState) -> ChatAgentState:
    """Запрашивает уточнение у пользователя"""
    start_time = time.time()
    
    logger.info("❓ [chat_agent] node_clarify - Запрос уточнения...")
    
    # Добавляем пользовательское сообщение в историю
    if not state.chat_history:
        state.chat_history = []
    
    state.chat_history.append({
        "role": "user",
        "content": state.message
    })
    
    try:
        # Формируем ответ с запросом уточнения
        message_lower = state.message.lower()
        
        if any(word in message_lower for word in ["найти", "найду", "ищу", "поиск"]):
            reply = "Что именно вы хотите найти? Укажите автора, название книги или жанр."
            chips = [
                {"text": "Автор", "action": "search"},
                {"text": "Название книги", "action": "search"},
                {"text": "Жанр", "action": "search"},
                {"text": "Посоветуй сам", "action": "recommend"}
            ]
        elif any(word in message_lower for word in ["книг", "читать", "литератур"]):
            reply = "Какой тип книг вас интересует? Можете указать жанр или конкретные предпочтения."
            chips = [
                {"text": "Фантастика", "action": "search"},
                {"text": "Детектив", "action": "search"},
                {"text": "Классика", "action": "search"},
                {"text": "Дай рекомендации", "action": "recommend"}
            ]
        else:
            reply = "Могу помочь найти книги! Что именно вас интересует?"
            chips = [
                {"text": "Конкретная книга", "action": "search"},
                {"text": "Автор", "action": "search"},
                {"text": "Жанр", "action": "search"},
                {"text": "Посоветуй что-то", "action": "recommend"}
            ]
        
        # Формируем результат
        state.results = [{
            "message": reply,
            "intent": "clarify",
            "chips": chips,
            "clarify_mode": True
        }]
        
        # Добавляем в историю
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в node_clarify: {e}")
        state.results = [{
            "message": "Чем могу помочь? Ищете что-то конкретное?",
            "intent": "error",
            "chips": [{"text": "Найти книги", "action": "search"}]
        }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["clarify_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Уточнение завершено за {execution_time:.1f}ms")
    
    return state

# Создание графа
builder = StateGraph(ChatAgentState)

# Добавляем узлы
builder.add_node("chat_response", node_chat_response_simple)
builder.add_node("recommendations", node_recommendations)
builder.add_node("simple_search", node_simple_search)
builder.add_node("clarify", node_clarify)

# Устанавливаем точку входа с роутингом
builder.set_conditional_entry_point(
    route_message,
    {
        "simple_search": "simple_search",
        "recommendations": "recommendations", 
        "chat_response": "chat_response",
        "clarify": "clarify"
    }
)

# Завершение
builder.add_edge("chat_response", END)
builder.add_edge("recommendations", END)
builder.add_edge("simple_search", END)
builder.add_edge("clarify", END)

# Компилируем граф
chat_agent_graph = builder.compile()