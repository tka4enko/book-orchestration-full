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
    # Base fields
    session_id: str
    message: str  # Current user message
    
    # Session history (accumulates between calls)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Chat agent fields
    chat_history: List[Dict[str, str]] = Field(default_factory=list)  # Deprecated, use conversation_history
    reply_message: Optional[str] = None
    chips: List[Dict[str, str]] = Field(default_factory=list)
    should_search: bool = False
    should_recommend: bool = False
    
    # User intent (determined in node_intent)
    intent: Optional[str] = None
    
    # Results (unified for chat and search)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

# LLM is used only for recommendations, not for routing
llm = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7, api_key=OPENAI_API_KEY)

def _format_chat_history(history: List[Dict[str, str]]) -> str:
    """Formats chat history for prompt"""
    if not history:
        return "Empty history"
    
    formatted = []
    for entry in history[-5:]:  # Last 5 messages
        role = entry.get("role", "user")
        content = entry.get("content", "")
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)

def node_chat_response_simple(state: ChatAgentState) -> ChatAgentState:
    """Simple chat response without LLM analysis"""
    start_time = time.time()
    
    logger.info("💬 [chat_agent] node_chat_response_simple - Simple chat response...")
    
    # Add user message to history
    if not state.chat_history:
        state.chat_history = []
    
    state.chat_history.append({
        "role": "user",
        "content": state.message
    })
    
    try:
        # Simple responses based on keywords
        message_lower = state.message.lower()
        
        if any(greeting in message_lower for greeting in ["привет", "здравствуй", "hello", "hi"]):  # Russian: привет=hello, здравствуй=greetings
            reply = "Hello! How are you? How can I help you with books?"
            chips = [
                {"text": "Find a book", "action": "search"},
                {"text": "Recommend something", "action": "recommend"}
            ]
        elif any(phrase in message_lower for phrase in ["не знаю", "скучно", "что делать"]):  # Russian: не знаю=don't know, скучно=boring, что делать=what to do
            reply = "I understand! Maybe let's read something interesting? What genre do you like?"
            chips = [
                {"text": "Detective", "action": "search"}, 
                {"text": "Fantasy", "action": "search"},
                {"text": "Recommend yourself", "action": "recommend"}
            ]
        else:
            reply = "Interesting! Tell me more - what exactly interests you?"
            chips = [
                {"text": "Find a specific book", "action": "search"},
                {"text": "Give recommendations", "action": "recommend"}
            ]
        
        # Form result
        state.results = [{
            "message": reply,
            "intent": "chat",
            "chips": chips,
            "chat_mode": True
        }]
        
        # Add to history
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Error in chat_response_simple: {e}")
        state.results = [{
            "message": "An error occurred. How can I help?",  # Translated from Russian: Произошла ошибка. Чем могу помочь?
            "intent": "error",
            "chips": [{"text": "Find books", "action": "search"}]  # Translated from Russian: Найти книги
        }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["chat_response_simple_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Simple chat response in {execution_time:.1f}ms")
    
    return state

def node_chat_response(state: ChatAgentState) -> ChatAgentState:
    """Forms final response for chat mode"""
    start_time = time.time()
    
    logger.info("💬 [chat_agent] node_chat_response - Forming chat response...")
    
    # Form structured response
    state.results = [{
        "message": state.reply_message,
        "intent": "chat",
        "chips": state.chips,
        "chat_mode": True
    }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["chat_response_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Chat response formed in {execution_time:.1f}ms")
    
    return state

async def node_recommendations(state: ChatAgentState) -> ChatAgentState:
    """Generates recommendations based on what's in ChromaDB"""
    logger.info("💡 [chat_agent] node_recommendations - Generating recommendations from database...")
    
    start_time = time.time()
    
    try:
        # Connect to ChromaDB to get statistics and real genres
        from .simple_retriever import SimpleVectorRetriever
        
        retriever = SimpleVectorRetriever()
        stats = retriever.get_collection_stats()
        
        logger.info(f"📊 Database statistics: books={stats['books']}, content={stats['content']}")
        
        if stats['books'] > 0:
            # Get real genres from database
            try:
                # Query books collection to get metadata
                sample_results = await retriever.search("", k=20)  # Get books for genre analysis
                
                # Extract unique genres from metadata
                genres = set()
                authors = set()
                
                logger.info(f"📊 Analyzing {len(sample_results)} results to extract genres")
                
                for i, result in enumerate(sample_results):
                    metadata = result.get('metadata', {})
                    logger.info(f"📖 Result {i}: metadata = {metadata}")
                    
                    # Extract genres from correct fields
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
                    
                    # Authors for diversity
                    author = metadata.get('author', '')
                    if author and author != 'Unknown':
                        authors.add(author)
                
                logger.info(f"🎯 Found genres: {list(genres)}")
                logger.info(f"👤 Found authors: {list(authors)}")
                
                # Form chips based on real data
                chips = []
                
                # Add real genres (maximum 4)
                real_genres = list(genres)[:4]
                for genre in real_genres:
                    chips.append({"text": genre, "action": "search"})
                
                # Add popular authors (maximum 2)
                popular_authors = list(authors)[:2]
                for author in popular_authors:
                    chips.append({"text": author, "action": "search"})
                
                # If no data, use general options
                if not chips:
                    chips = [
                        {"text": "Show everything available", "action": "search"},
                        {"text": "Random book", "action": "search"}
                    ]
                
                reply = "Below I've provided several interesting options from my collection:"
                
            except Exception as e:
                logger.error(f"❌ Error getting genres: {e}")
                # Fallback to simple options
                reply = "Below I've provided several options from my collection:"
                chips = [
                    {"text": "Show everything available", "action": "search"},
                    {"text": "Random book", "action": "search"}
                ]
        else:
            reply = "Unfortunately, the book database is empty. Try uploading some books first."
            chips = []
        
        # Form result
        state.results = [{
            "message": reply,
            "intent": "recommendations",
            "chips": chips,
            "recommendation_mode": True
        }]
        
        # Add to chat history
        if not state.chat_history:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["recommendations_ms"] = execution_time
        
        logger.info(f"✅ [chat_agent] Recommendations generated in {execution_time:.1f}ms")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [chat_agent] Recommendation generation error: {e}")
        
        # Return error as result
        error_message = "Sorry, can't give recommendations now. Try searching for something specific!"  # Translated from Russian: Извините, не могу сейчас дать рекомендации...
        
        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["recommendations_error_ms"] = execution_time
        
        return state

async def node_intent(state: ChatAgentState) -> ChatAgentState:
    """Node for analyzing user intent"""
    message = state.message
    
    logger.info(f"🧠 [node_intent] Intent analysis: '{message}'")
    logger.info(f"📜 [node_intent] Session history: {len(state.conversation_history)} messages")
    logger.info(f"🆔 [node_intent] Session ID: {state.session_id}")
    
    # Add current message to session history
    state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": time.time()
    })
    
    try:
        # Quick LLM request for routing
        router_prompt = f"""You are a book search router. Determine if books can be found for this query.

SEARCH - if there is ANY search information:
- Author or book title
- Genre or topic  
- Publication year or period
- Plot or content description
- Book themes and topics
- Book language
- ISBN number
- Any specific details about the book

RECOMMEND - requests for advice and recommendations

CLARIFY - only very general queries WITHOUT specifics:
- "find a book" (without specifying which one)
- "looking for something to read" (without details)

CHAT - regular conversation not related to book search

Query: "{message}"

IMPORTANT: Any genres and topics - this is SEARCH!
Can specific books be found for this query?
Answer (SEARCH/RECOMMEND/CLARIFY/CHAT):"""
        
        response = llm.invoke([("user", router_prompt)]).content.strip().upper()
        
        logger.info(f"🧠 [router] LLM response: '{response}'")
        
        # Save analysis result in state
        state.intent = response
        
        logger.info(f"✅ [node_intent] Intent determined: '{response}'")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [node_intent] Analysis error: {e}")
        state.intent = "CHAT"  # Fallback
        return state

def route_by_intent(state: ChatAgentState) -> str:
    """Routing based on determined intent"""
    intent = state.intent
    
    logger.info(f"🔄 [route_by_intent] Routing for intent: '{intent}'")
    
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
    """Performs simple search using process_simple_search"""
    logger.info("🔍 [chat_agent] node_simple_search - Using simple_orchestrator...")
    
    start_time = time.time()
    
    try:
        # Call simple search
        search_result = await process_simple_search(state.session_id, state.message)
        
        # Use ready formatted response from simple_orchestrator
        message = search_result.get('response', 'Results not found')
        
        # Add to results
        state.results = [{
            "message": message,
            "intent": search_result.get('intent', 'search'),
            "search_mode": True,
            "original_response": search_result.get('response'),
            "raw_results": search_result.get('results', [])
        }]
        
        # Add to chat history
        if not state.chat_history:
            state.chat_history = []
        
        state.chat_history.append({
            "role": "assistant",
            "content": message
        })
        
        # Copy performance metrics
        if search_result.get('performance_metrics'):
            state.performance_metrics.update(search_result['performance_metrics'])
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["simple_search_ms"] = execution_time
        
        logger.info(f"✅ [chat_agent] Simple search completed in {execution_time:.1f}ms")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [chat_agent] Simple search error: {e}")
        
        # Return error as result
        error_message = "Sorry, a search error occurred. Try rephrasing your query."  # Translated from Russian: Извините, произошла ошибка при поиске...
        
        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]
        
        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["simple_search_error_ms"] = execution_time
        
        return state

def node_clarify(state: ChatAgentState) -> ChatAgentState:
    """Requests clarification from user"""
    start_time = time.time()
    
    logger.info("❓ [chat_agent] node_clarify - Clarification request...")
    
    # Add user message to history
    if not state.chat_history:
        state.chat_history = []
    
    state.chat_history.append({
        "role": "user",
        "content": state.message
    })
    
    try:
        # Form response with clarification request
        message_lower = state.message.lower()
        
        if any(word in message_lower for word in ["найти", "найду", "ищу", "поиск"]):  # Russian: найти=find, найду=will find, ищу=searching, поиск=search
            reply = "What exactly do you want to find? Specify author, book title or genre."
            chips = [
                {"text": "Author", "action": "search"},
                {"text": "Book title", "action": "search"},
                {"text": "Genre", "action": "search"},
                {"text": "Recommend yourself", "action": "recommend"}
            ]
        elif any(word in message_lower for word in ["книг", "читать", "литератур"]):  # Russian: книг=books, читать=read, литератур=literature
            reply = "What type of books interest you? You can specify genre or specific preferences."
            chips = [
                {"text": "Fantasy", "action": "search"},
                {"text": "Detective", "action": "search"},
                {"text": "Classic", "action": "search"},
                {"text": "Give recommendations", "action": "recommend"}
            ]
        else:
            reply = "I can help find books! What exactly interests you?"
            chips = [
                {"text": "Specific book", "action": "search"},
                {"text": "Author", "action": "search"},
                {"text": "Genre", "action": "search"},
                {"text": "Recommend something", "action": "recommend"}
            ]
        
        # Form result
        state.results = [{
            "message": reply,
            "intent": "clarify",
            "chips": chips,
            "clarify_mode": True
        }]
        
        # Add to history
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Error in node_clarify: {e}")
        state.results = [{
            "message": "How can I help? Looking for something specific?",
            "intent": "error",
            "chips": [{"text": "Find books", "action": "search"}]
        }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["clarify_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Clarification completed in {execution_time:.1f}ms")
    
    return state

# Graph creation
builder = StateGraph(ChatAgentState)

# Add nodes
builder.add_node("intent", node_intent)  # New node for intent analysis
builder.add_node("chat_response", node_chat_response_simple)
builder.add_node("recommendations", node_recommendations)
builder.add_node("simple_search", node_simple_search)
builder.add_node("clarify", node_clarify)

# New architecture: first intent analysis, then routing
builder.set_entry_point("intent")

# Conditional routing after intent analysis
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

# Completion
builder.add_edge("chat_response", END)
builder.add_edge("recommendations", END)
builder.add_edge("simple_search", END)
builder.add_edge("clarify", END)

# Compile graph with checkpointer for state preservation between calls
memory = MemorySaver()
chat_agent_graph = builder.compile(checkpointer=memory)