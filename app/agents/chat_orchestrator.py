from typing import Optional, Dict, Any, List
import logging
import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ..infra.settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from ..services.hybrid_retriever import isbn_exact, hybrid_with_rerank
from ..core.isbn_utils import extract_first_isbn

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
        r'^да$', r'^давай$', r'^хорошо$', r'^ок$', r'^okay$', r'^конечно$',  # Russian: да=yes, давай=let's go, хорошо=good, ок=ok, конечно=of course
        r'^согласен$', r'^согласна$', r'^подходит$', r'^отлично$', r'^супер$',  # Russian: согласен/согласна=agree, подходит=suitable, отлично=excellent, супер=super
        r'^можно$', r'^пойдет$', r'^идет$', r'^принято$', r'^звучит\s+хорошо$'  # Russian: можно=possible, пойдет=will do, идет=goes, принято=accepted, звучит хорошо=sounds good
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
    """Determines message intent"""
    logger.info("�� [orchestrator_with_chat.py] node_detect_intent - Analyzing user intent...")
    logger.info(f"    Input: '{state.message}'")
    
    # Check for ISBN first
    isbn = extract_first_isbn(state.message)
    if isbn:
        logger.info(f"�� ISBN detected: {isbn}")
        state.intent = "isbn"
        state.filters = {"isbn": isbn.get("isbn13") or isbn.get("isbn10") or isbn.get("isbn")}
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
    1. If user agrees with vague words (да, давай, хорошо, ок, конечно) to assistant's suggestion:  # Russian: да=yes, давай=let's go, хорошо=good, ок=ok, конечно=of course
       - IF assistant mentioned MULTIPLE options (e.g., "fantasy or detective"):  # Russian example: фэнтези или детективы 
         → USE "clarify" intent to let user choose specific option
       - IF assistant mentioned ONE specific option: 
         → Extract that specific option and create appropriate intent
       
    2. Examples:
       History: "assistant: How about books in fantasy or detective genre?"  # Russian example: Как насчет книг в жанре фэнтези или детективов?
       User: "let's go"  # Russian example: давай
       → {"intent": "clarify", "filters": {"options": ["fantasy", "detective"], "type": "genre"}}  # Russian terms: фэнтези, детективы
       
       History: "assistant: Maybe science fiction?"  # Russian example: Может быть фантастика?
       User: "let's go"  # Russian example: давай
       → {"intent": "genre", "filters": {"genre": "science fiction"}}  # Russian term: фантастика
       
       History: "assistant: Try Stephen King"  # Russian example: Попробуй Стивена Кинга
       User: "good"  # Russian example: хорошо
       → {"intent": "author", "filters": {"author": "Stephen King"}}  # Russian term: Стивен Кинг
       
       History: "assistant: Maybe books about psychology?"  # Russian example: Может книги по психологии?
       User: "yes"  # Russian example: да
       → {"intent": "topic", "filters": {"topic": "psychology"}}  # Russian term: психология

    3. ADVICE/RECOMMENDATION REQUESTS are CLARIFY:
       User: "recommend a genre" / "recommend what to read" / "what should I read?"  # Russian examples: посоветуй жанр / посоветуй что почитать / что мне почитать?
       → {"intent": "clarify", "filters": {"request_type": "genre_advice"}}
       
       User: "recommend an author" / "which author to read?"  # Russian examples: посоветуй автора / какого автора почитать?
       → {"intent": "clarify", "filters": {"request_type": "author_advice"}}
       
       User: "don't know what to choose" / "help me choose"  # Russian examples: не знаю что выбрать / помоги выбрать
       → {"intent": "clarify", "filters": {"request_type": "general_advice"}}

    4. IMPORTANT: For non-specific agreements:
       - If user says just "let's go" without clear context from assistant's last message  # Russian example: давай
       - If assistant didn't make specific suggestions in last message
       → USE "chat" intent to continue conversation
       
       BUT: If assistant was asking about books/genres (context shows book discussion):
       User: "okay let's go" / "let's go"  # Russian examples: окей давай / давай 
       → {"intent": "clarify", "filters": {"request_type": "genre_advice"}}

    5. Look for specific mentions in assistant's previous messages:
       - Author names (Стивен Кинг, Агата Кристи, George Orwell, etc.)  # Russian: Стивен Кинг=Stephen King, Агата Кристи=Agatha Christie
       - Genres (фантастика, детектив, психология, бизнес, фэнтези, etc.)  # Russian: фантастика=sci-fi, детектив=detective, психология=psychology, бизнес=business, фэнтези=fantasy
       - Topics (искусство, наука, история, etc.)  # Russian: искусство=art, наука=science, история=history
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
        # Special handling for clarify intent
        if state.intent == "clarify":
            logger.info("🔍 Clarify intent detected - helping with book choice")
            
            # Handle choice between options (multiple variants)
            if state.filters.get("options"):
                options = state.filters.get("options", [])
                if len(options) == 2:
                    reply = f"Excellent! What interests you more: {options[0]} or {options[1]}?"  # Translated from Russian: Отлично! Что именно тебя больше интересует...
                else:
                    option_list = ", ".join(options)
                    reply = f"Good! Choose exactly what: {option_list}"  # Translated from Russian: Хорошо! Выбери что именно...
            
            # Handle advice requests
            elif state.filters.get("request_type"):
                request_type = state.filters["request_type"]
                
                if request_type == "genre_advice":
                    reply = "С удовольствием помогу выбрать жанр! Что тебе больше по душе?"  # Russian: I'll gladly help choose a genre! What do you prefer?
                elif request_type == "author_advice":
                    reply = "Отлично! Какого типа авторов предпочитаешь?"  # Russian: Excellent! What type of authors do you prefer?
                else:  # general_advice
                    reply = "Давай подберем что-то интересное! С чего начнем?"  # Russian: Let's find something interesting! Where shall we start?
            
            # Fallback for other clarify cases
            else:
                reply = "Чем могу помочь с выбором книг?"  # Russian: How can I help with choosing books?
                
            state.reply_message = reply
            state.is_chat_mode = True
            return state
        
        # Format history
        history_text = _format_chat_history(state.chat_history)
        
        # Check conversation length - if more than 4 messages, become more insistent
        chat_length = len(state.chat_history or [])
        if chat_length >= 4:
            logger.info("📢 Long chat detected - being more insistent about books")
            
            # Check - maybe user agreed to the book suggestion?
            is_agreement = _is_agreement_message(state.message)
            if is_agreement:
                logger.info("🤝 User agreed after insistence - switching to clarify for advice")
                # Switch to clarify for genre selection
                state.intent = "clarify"
                state.filters = {"request_type": "genre_advice"}
                # Return to clarify processing above
                return node_chat(state)
            
            # If not agreement - continue insisting, but less intrusively
            result = {
                "reply": "Я BookBot - помощник по книгам. Может, все-таки посмотрим что-то интересное?",  # Russian: I'm BookBot - a book assistant. Maybe let's look at something interesting after all?
            }
        else:
            # Analyze message through LLM
            chat_prompt = ChatPromptTemplate.from_template(CHAT_SYS)
            chain = chat_prompt | chat_llm | JsonOutputParser()
            
            result = chain.invoke({
                "message": state.message,
                "chat_history": history_text
            })
        
        # Create response
        reply = result.get("reply", "Извините, не понял.")  # Russian: Sorry, didn't understand.

        state.reply_message = reply
        state.is_chat_mode = True
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        state.reply_message = "Извините, что-то пошло не так. Могу помочь найти интересные книги!"  # Russian: Sorry, something went wrong. I can help find interesting books!
        state.is_chat_mode = True
        return state

def node_route_search(state: ChatStateWithChat) -> ChatStateWithChat:
    """Book search (copying logic from original)"""
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
    """Forms final answer"""
    logger.info("�� [orchestrator_with_chat.py] node_answer - Generating final response...")
    
    if state.is_chat_mode:
        # If this is chat, use ready response
        logger.info("💬 Using chat response")
        
        # Update chat history
        if state.chat_history is None:
            state.chat_history = []
        
        # Add bot response to history
        state.chat_history.append({
            "role": "assistant", 
            "content": state.reply_message
        })
        
        state.results = [{
            "message": state.reply_message,
            "intent": "chat",
        }]
        return state
    
    # If this is search, format results
    logger.info(f"    Formatting {len(state.results)} search results")
    
    if not state.results:
        logger.info("🚫 No results found")
        no_results_msg = f"К сожалению, не найдено книг, соответствующих вашему запросу '{state.message}'. Попробуйте изменить запрос или загрузить больше книг."  # Russian: Unfortunately, no books found matching your query '{state.message}'. Try changing the query or uploading more books.
        state.results = [{"message": no_results_msg, "intent": state.intent}]
        return state
    
    # Format search results
    if state.results:
        formatted_results = []
        for i, result in enumerate(state.results[:5]):  # Top-5 results
            title = result.get('title', 'Unknown')
            author = result.get('author', 'Unknown')
            genre = result.get('primary_genre', 'Unknown')
            formatted_results.append(f"{i+1}. «{title}» от {author} ({genre})")  # Russian: от=by
        
        reply = f"Нашёл {len(state.results)} книг по вашему запросу:\n\n" + "\n".join(formatted_results)  # Russian: Found {count} books for your query:
        
        
        state.results = [{
            "message": reply,
            "intent": state.intent,
        }]
        
        # Update chat history for search
        if state.chat_history is None:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant", 
            "content": reply
        })
    
    return state

def should_route_to_chat(state: ChatStateWithChat) -> str:
    """Determines whether to process as chat or search"""
    
    # If search intent is already determined - use it
    search_intents = ["isbn", "author", "title", "author_title", "genre", "topic"]
    if state.intent in search_intents:
        return "route_search"
    
    # If intent indicates chat - use it
    chat_intents = ["clarify", "greeting", "farewell", "chat"]
    if state.intent in chat_intents:
        return "chat"
    
    # If intent is not determined or unclear - default to chat
    return "chat"

# Create graph
builder = StateGraph(ChatStateWithChat)
builder.add_node("detect_intent", node_detect_intent)
builder.add_node("chat", node_chat)
builder.add_node("route_search", node_route_search)
builder.add_node("answer", node_answer)

builder.set_entry_point("detect_intent")

# Conditional transitions
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
