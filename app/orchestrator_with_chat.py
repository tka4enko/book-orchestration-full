from typing import Optional, Dict, Any, List
import logging
import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .retrievers import isbn_exact, hybrid_with_rerank
from .utils_isbn import extract_first_isbn

logger = logging.getLogger(__name__)

class ChatStateWithChat(BaseModel):
    session_id: str
    message: str
    intent: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    exclude_filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    need_clarify: bool = False
    clarify_question: Optional[str] = None
    
    # New fields for chat
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    is_chat_mode: bool = False
    reply_message: Optional[str] = None
    chips: List[Dict[str, Any]] = Field(default_factory=list)

llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0, api_key=OPENAI_API_KEY)
chat_llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7, api_key=OPENAI_API_KEY)

# Chat prompt
CHAT_SYS = """You are BookBot, a book search assistant.

MAIN GOAL: Guide users to book search and recommendations!

Rules:
- Answer VERY briefly (1-2 sentences maximum)
- Don't engage in long conversations about other topics  
- ALWAYS actively direct to books after a short answer
- Connect any topic to books where possible
- After 2-3 message exchanges - persistently suggest books

Examples of connecting topics to books:
- Video games → "There are great books about the gaming industry!"  
- Football → "I can find books about football or sports!"
- Work is hard → "Books help relax after work!"

History: {chat_history}
Message: {message}

Return JSON:
{{
    "reply": "short answer + direction to books",
    "chips": [
        {{"text": "Recommend what to read", "action": "chat"}},
        {{"text": "Find books by topic", "action": "search"}}
    ]
}}"""

def _format_chat_history(history: List[Dict[str, Any]]) -> str:
    """Formats chat history"""
    if not history:
        return ""
    
    formatted = []
    for entry in history[-5:]:  # Last 5 messages
        role = entry.get("role", "user")
        content = entry.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)

def _is_agreement_message(message: str) -> bool:
    """Checks if message is an agreement"""
    import re
    agreement_patterns = [
        r'^да$', r'^давай$', r'^хорошо$', r'^ок$', r'^okay$', r'^конечно$',
        r'^согласен$', r'^согласна$', r'^подходит$', r'^отлично$', r'^супер$',
        r'^можно$', r'^пойдет$', r'^идет$', r'^принято$', r'^звучит\s+хорошо$'
    ]
    
    message_lower = message.lower().strip()
    for pattern in agreement_patterns:
        if re.match(pattern, message_lower):
            return True
    return False

def _docs_to_payload(docs):
    """Converts documents to response format"""
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

def node_detect_intent(state: ChatStateWithChat) -> ChatStateWithChat:
    """Определяет интент сообщения"""
    logger.info("�� [orchestrator_with_chat.py] node_detect_intent - Analyzing user intent...")
    logger.info(f"    Input: '{state.message}'")
    
    # Check for ISBN first
    isbn = extract_first_isbn(state.message)
    if isbn:
        logger.info(f"�� ISBN detected: {isbn}")
        state.intent = "isbn"
        state.filters = {"isbn13": isbn["isbn13"], "isbn10": isbn.get("isbn10")}
        return state
    
    # Enhanced LLM intent detection with context analysis
    INTENT_SYS = """You are an advanced intent classifier for a book search assistant.

    CRITICAL: Analyze chat context to understand user agreements and suggestions!

    Intents:
    - isbn: User provides ISBN number
    - author_title: User asks for specific author + title combination  
    - author: User asks for books by specific author
    - title: User asks for specific book title
    - genre: User asks for books in specific genre
    - topic: User asks for books about specific topic/subject
    - clarify: User needs help CHOOSING books (advice, recommendations, "what to read")
    - greeting: User says hello, greetings  
    - chat: General conversation NOT about books (weather, how are you, etc)

    CONTEXT ANALYSIS RULES:
    1. If user agrees with vague words (да, давай, хорошо, ок, конечно) to assistant's suggestion:
       - IF assistant mentioned MULTIPLE options (e.g., "фэнтези или детективы"): 
         → USE "clarify" intent to let user choose specific option
       - IF assistant mentioned ONE specific option: 
         → Extract that specific option and create appropriate intent
       
    2. Examples:
       History: "assistant: Как насчет книг в жанре фэнтези или детективов?"
       User: "давай"
       → {"intent": "clarify", "filters": {"options": ["фэнтези", "детективы"], "type": "genre"}}
       
       History: "assistant: Может быть фантастика?"
       User: "давай"  
       → {"intent": "genre", "filters": {"genre": "фантастика"}}
       
       History: "assistant: Попробуй Стивена Кинга"
       User: "хорошо" 
       → {"intent": "author", "filters": {"author": "Стивен Кинг"}}
       
       History: "assistant: Может книги по психологии?"
       User: "да"
       → {"intent": "topic", "filters": {"topic": "психология"}}

    3. ADVICE/RECOMMENDATION REQUESTS are CLARIFY:
       User: "посоветуй жанр" / "посоветуй что почитать" / "что мне почитать?"
       → {"intent": "clarify", "filters": {"request_type": "genre_advice"}}
       
       User: "посоветуй автора" / "какого автора почитать?"
       → {"intent": "clarify", "filters": {"request_type": "author_advice"}}
       
       User: "не знаю что выбрать" / "помоги выбрать"
       → {"intent": "clarify", "filters": {"request_type": "general_advice"}}

    4. IMPORTANT: For non-specific agreements:
       - If user says just "давай" without clear context from assistant's last message
       - If assistant didn't make specific suggestions in last message
       → USE "chat" intent to continue conversation
       
       BUT: If assistant was asking about books/genres (context shows book discussion):
       User: "окей давай" / "давай" 
       → {"intent": "clarify", "filters": {"request_type": "genre_advice"}}

    5. Look for specific mentions in assistant's previous messages:
       - Author names (Стивен Кинг, Агата Кристи, George Orwell, etc.)
       - Genres (фантастика, детектив, психология, бизнес, фэнтези, etc.) 
       - Topics (искусство, наука, история, etc.)
       - Book titles

    Return JSON: {"intent":"...", "filters":{...}}
    Extract exact values from context when user agrees."""
    
    # Check for agreement for special processing
    is_agreement = _is_agreement_message(state.message)
    if is_agreement:
        logger.info("🤝 Agreement detected - using enhanced context analysis")
    
    # Format history for context
    history_text = _format_chat_history(state.chat_history)
    context = f"Chat history:\n{history_text}\n\nCurrent message: {state.message}"
    
    out = llm.invoke([("system", INTENT_SYS), ("user", context)]).content
    logger.info(f"    LLM intent response: {out}")
    
    import json, re
    try:
        j = json.loads(out)
    except Exception:
        logger.warning("⚠️ LLM response not valid JSON, trying regex...")
        m = re.search(r"\{.*\}", out, re.S)
        j = json.loads(m.group(0)) if m else {"intent":"chat","filters":{}}
    
    state.intent = j.get("intent") or "chat"
    state.filters = j.get("filters") or {}
    
    logger.info(f"✅ Intent detected: '{state.intent}' with filters: {state.filters}")
    return state

def node_chat(state: ChatStateWithChat) -> ChatStateWithChat:
    """Handles regular chat (not book search)"""
    logger.info("�� [orchestrator_with_chat.py] node_chat - Processing chat message...")
    
    try:
        # Специальная обработка clarify intent
        if state.intent == "clarify":
            logger.info("🔍 Clarify intent detected - helping with book choice")
            
            # Обработка выбора между опциями (множественные варианты)
            if state.filters.get("options"):
                options = state.filters.get("options", [])
                if len(options) == 2:
                    reply = f"Отлично! Что именно тебя больше интересует: {options[0]} или {options[1]}?"
                    chips = [
                        {"text": options[0].capitalize(), "action": "search"},
                        {"text": options[1].capitalize(), "action": "search"}
                    ]
                else:
                    option_list = ", ".join(options)
                    reply = f"Хорошо! Выбери что именно: {option_list}"
                    chips = [{"text": opt.capitalize(), "action": "search"} for opt in options]
            
            # Обработка запросов на советы
            elif state.filters.get("request_type"):
                request_type = state.filters["request_type"]
                
                if request_type == "genre_advice":
                    reply = "С удовольствием помогу выбрать жанр! Что тебе больше по душе?"
                    chips = [
                        {"text": "Детектив", "action": "search"},
                        {"text": "Фэнтези", "action": "search"}, 
                        {"text": "Классика", "action": "search"},
                        {"text": "Романы", "action": "search"}
                    ]
                elif request_type == "author_advice":
                    reply = "Отлично! Какого типа авторов предпочитаешь?"
                    chips = [
                        {"text": "Современные авторы", "action": "search"},
                        {"text": "Классики", "action": "search"},
                        {"text": "Зарубежные авторы", "action": "search"},
                        {"text": "Русские авторы", "action": "search"}
                    ]
                else:  # general_advice
                    reply = "Давай подберем что-то интересное! С чего начнем?"
                    chips = [
                        {"text": "Детектив", "action": "search"},
                        {"text": "Фэнтези", "action": "search"},
                        {"text": "Психология", "action": "search"},
                        {"text": "История", "action": "search"}
                    ]
            
            # Fallback для других clarify случаев
            else:
                reply = "Чем могу помочь с выбором книг?"
                chips = [
                    {"text": "Посоветуй жанр", "action": "chat"},
                    {"text": "Найти по теме", "action": "search"}
                ]
                
            state.reply_message = reply
            state.chips = chips
            state.is_chat_mode = True
            return state
        
        # Форматируем историю
        history_text = _format_chat_history(state.chat_history)
        
        # Проверяем длину разговора - если больше 4 сообщений, становимся настойчивее
        chat_length = len(state.chat_history or [])
        if chat_length >= 4:
            logger.info("📢 Long chat detected - being more insistent about books")
            
            # Проверяем - может пользователь согласился на предложение о книгах?
            is_agreement = _is_agreement_message(state.message)
            if is_agreement:
                logger.info("🤝 User agreed after insistence - switching to clarify for advice")
                # Переключаемся на clarify для выбора жанра
                state.intent = "clarify"
                state.filters = {"request_type": "genre_advice"}
                # Возвращаемся к обработке clarify выше
                return node_chat(state)
            
            # Если не согласие - продолжаем настаивать, но менее навязчиво
            result = {
                "reply": "Я BookBot - помощник по книгам. Может, все-таки посмотрим что-то интересное?",
                "chips": [
                    {"text": "Посоветуй жанр", "action": "chat"},
                    {"text": "Найти детектив", "action": "search"},
                    {"text": "Найти фэнтези", "action": "search"}
                ]
            }
        else:
            # Анализируем сообщение через LLM
            chat_prompt = ChatPromptTemplate.from_template(CHAT_SYS)
            chain = chat_prompt | chat_llm | JsonOutputParser()
            
            result = chain.invoke({
                "message": state.message,
                "chat_history": history_text
            })
        
        # Создаем ответ
        reply = result.get("reply", "Извините, не понял.")
        chips = result.get("chips", [])
        
        state.reply_message = reply
        state.chips = chips
        state.is_chat_mode = True
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        state.reply_message = "Извините, что-то пошло не так. Могу помочь найти интересные книги!"
        state.chips = [{"text": "Найти книги", "action": "search"}]
        state.is_chat_mode = True
        return state

def node_route_search(state: ChatStateWithChat) -> ChatStateWithChat:
    """Поиск книг (копируем логику из оригинального)"""
    logger.info("�� [orchestrator_with_chat.py] node_route_search - Searching for relevant documents...")
    logger.info(f"    Intent: '{state.intent}'")
    
    if state.intent == "isbn":
        logger.info("📚 ISBN search path:")
        docs = isbn_exact(state.message)
        if docs:
            logger.info(f"    ✅ Found {len(docs)} exact ISBN matches")
        else:
            logger.info("    ⚠️  No exact ISBN match, trying semantic search...")
            retr = hybrid_with_rerank(intent=state.intent)
            docs = retr.invoke(state.message, intent=state.intent)
            logger.info(f"    📊 Semantic search returned {len(docs)} documents")
            
        state.results = _docs_to_payload(docs[:1])
        return state

    logger.info("🔎 Semantic search path:")
    
    # Use hybrid search with threshold filtering for all other intents
    retr = hybrid_with_rerank(intent=state.intent) 
    docs = retr.invoke(state.message, intent=state.intent)
    
    logger.info(f"    📊 Retrieved {len(docs)} documents")
    
    payload = _docs_to_payload(docs)

    if state.intent == "author_title":
        logger.info("👤📖 Author-Title specific filtering:")
        
        f = state.filters or {}
        a = (f.get("author") or f.get("author_canonical") or "").lower()
        t = (f.get("title") or f.get("title_canonical") or "").lower()
        
        # Filter exact matches
        exact = []
        for p in payload:
            author_match = (p.get("author") or "").lower().find(a) != -1 if a else True
            title_match = (p.get("title") or "").lower().find(t) != -1 if t else True
            
            if author_match and title_match:
                exact.append(p)
        
        state.results = exact[:1] if exact else []
        return state

    logger.info(f"📋 General search results: {len(payload)} documents")
    state.results = payload[:10]
    return state

def node_answer(state: ChatStateWithChat) -> ChatStateWithChat:
    """Формирует финальный ответ"""
    logger.info("�� [orchestrator_with_chat.py] node_answer - Generating final response...")
    
    if state.is_chat_mode:
        # Если это чат, используем готовый ответ
        logger.info("💬 Using chat response")
        
        # Обновляем историю чата
        if state.chat_history is None:
            state.chat_history = []
        
        # Добавляем ответ бота в историю
        state.chat_history.append({
            "role": "assistant", 
            "content": state.reply_message
        })
        
        state.results = [{
            "message": state.reply_message,
            "intent": "chat",
            "chips": state.chips
        }]
        return state
    
    # Если это поиск, форматируем результаты
    logger.info(f"    Formatting {len(state.results)} search results")
    
    if not state.results:
        logger.info("🚫 No results found")
        no_results_msg = f"К сожалению, не найдено книг, соответствующих вашему запросу '{state.message}'. Попробуйте изменить запрос или загрузить больше книг."
        state.results = [{"message": no_results_msg, "intent": state.intent}]
        return state
    
    # Форматируем результаты поиска
    if state.results:
        formatted_results = []
        for i, result in enumerate(state.results[:5]):  # Топ-5 результатов
            title = result.get('title', 'Unknown')
            author = result.get('author', 'Unknown')
            genre = result.get('primary_genre', 'Unknown')
            formatted_results.append(f"{i+1}. «{title}» от {author} ({genre})")
        
        reply = f"Нашёл {len(state.results)} книг по вашему запросу:\n\n" + "\n".join(formatted_results)
        
        # Добавляем чипсы для поиска
        chips = [
            {"text": "Ещё похожие", "action": "search"},
            {"text": "Другой жанр", "action": "search"},
            {"text": "Поговорим о другом", "action": "chat"}
        ]
        
        state.results = [{
            "message": reply,
            "intent": state.intent,
            "chips": chips
        }]
        
        # Обновляем историю чата для поиска
        if state.chat_history is None:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant", 
            "content": reply
        })
    
    return state

def should_route_to_chat(state: ChatStateWithChat) -> str:
    """Определяет, нужно ли обрабатывать как чат или поиск"""
    
    # Если уже определен интент для поиска - используем его
    search_intents = ["isbn", "author", "title", "author_title", "genre", "topic"]
    if state.intent in search_intents:
        return "route_search"
    
    # Если интент указывает на чат - используем его
    chat_intents = ["clarify", "greeting", "farewell", "chat"]
    if state.intent in chat_intents:
        return "chat"
    
    # Если интент не определен или неясен - по умолчанию чат
    return "chat"

# Создаем граф
builder = StateGraph(ChatStateWithChat)
builder.add_node("detect_intent", node_detect_intent)
builder.add_node("chat", node_chat)
builder.add_node("route_search", node_route_search)
builder.add_node("answer", node_answer)

builder.set_entry_point("detect_intent")

# Условные переходы
builder.add_conditional_edges(
    "detect_intent",
    should_route_to_chat,
    {
        "chat": "chat",
        "route_search": "route_search"
    }
)

builder.add_edge("chat", "answer")
builder.add_edge("route_search", "answer")
builder.add_edge("answer", END)

graph_with_chat = builder.compile()
