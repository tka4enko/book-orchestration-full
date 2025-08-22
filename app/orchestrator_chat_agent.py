from typing import Optional, Dict, Any, List
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .simple_orchestrator import process_simple_search

logger = logging.getLogger(__name__)

class ChatAgentState(BaseModel):
    # Базовые поля
    session_id: str
    message: str  # Текущее сообщение пользователя
    
    # История всей сессии (накапливается между вызовами)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Поля для чат-агента
    chat_history: List[Dict[str, str]] = Field(default_factory=list)  # Deprecated, используем conversation_history
    reply_message: Optional[str] = None
    chips: List[Dict[str, str]] = Field(default_factory=list)
    should_search: bool = False
    should_recommend: bool = False
    
    # Намерение пользователя (определяется в node_intent)
    intent: Optional[str] = None
    
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
                    
                    # Извлекаем жанры из правильных полей
                    # Primary genre  
                    primary_genre = metadata.get('primary_genre', '')
                    if primary_genre:
                        genres.add(primary_genre)
                    
                    # Secondary genres (JSON string)
                    secondary_genres = metadata.get('secondary_genres', '[]')
                    if secondary_genres and secondary_genres != '[]':
                        try:
                            import json
                            sec_genres = json.loads(secondary_genres)
                            if isinstance(sec_genres, list):
                                genres.update(sec_genres)
                        except:
                            pass
                    
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

async def node_intent(state: ChatAgentState) -> ChatAgentState:
    """Node для анализа намерения пользователя"""
    message = state.message
    
    logger.info(f"🧠 [node_intent] Анализ намерения: '{message}'")
    logger.info(f"📜 [node_intent] История сессии: {len(state.conversation_history)} сообщений")
    logger.info(f"🆔 [node_intent] Session ID: {state.session_id}")
    
    # Добавляем текущее сообщение в историю сессии
    state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": time.time()
    })
    
    try:
        # Быстрый LLM запрос для роутинга
        router_prompt = f"""Ты роутер поиска книг. Определи можно ли найти книги по этому запросу.

SEARCH - если есть ЛЮБАЯ поисковая информация:
- Автор или название книги
- Жанр или тематика  
- Год издания или период
- Описание сюжета или содержания
- Темы и топики книги
- Язык книги
- ISBN номер
- Любые конкретные детали о книге

RECOMMEND - просьбы о советах и рекомендациях

CLARIFY - только очень общие запросы БЕЗ конкретики:
- "найди книгу" (без указания какую)
- "ищу что-то почитать" (без деталей)

CHAT - обычное общение, не связанное с поиском книг

Запрос: "{message}"

ВАЖНО: Любые жанры и тематики - это SEARCH!
Можно ли найти конкретные книги по этому запросу?
Ответ (SEARCH/RECOMMEND/CLARIFY/CHAT):"""
        
        response = llm.invoke([("user", router_prompt)]).content.strip().upper()
        
        logger.info(f"🧠 [router] LLM ответ: '{response}'")
        
        # Сохраняем результат анализа в state
        state.intent = response
        
        logger.info(f"✅ [node_intent] Намерение определено: '{response}'")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [node_intent] Ошибка анализа: {e}")
        state.intent = "CHAT"  # Fallback
        return state

def route_by_intent(state: ChatAgentState) -> str:
    """Роутинг на основе определенного намерения"""
    intent = state.intent
    
    logger.info(f"🔄 [route_by_intent] Роутинг для намерения: '{intent}'")
    
    if "SEARCH" in intent:
        logger.info(f"🔍 [route_by_intent] → simple_search")
        return "simple_search"
    elif "RECOMMEND" in intent:
        logger.info(f"💡 [route_by_intent] → recommendations")
        return "recommendations"
    elif "CLARIFY" in intent:
        logger.info(f"❓ [route_by_intent] → clarify")
        return "clarify"
    else:
        logger.info(f"💬 [route_by_intent] → chat_response")
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
builder.add_node("intent", node_intent)  # Новый node для анализа намерения
builder.add_node("chat_response", node_chat_response_simple)
builder.add_node("recommendations", node_recommendations)
builder.add_node("simple_search", node_simple_search)
builder.add_node("clarify", node_clarify)

# Новая архитектура: сначала анализ намерения, потом роутинг
builder.set_entry_point("intent")

# Условный роутинг после анализа намерения
builder.add_conditional_edges(
    "intent",
    route_by_intent,
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

# Компилируем граф с checkpointer для сохранения состояния между вызовами
memory = MemorySaver()
chat_agent_graph = builder.compile(checkpointer=memory)