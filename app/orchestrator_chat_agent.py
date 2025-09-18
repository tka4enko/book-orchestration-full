from typing import Optional, Dict, Any, List, Annotated
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .simple_orchestrator import process_simple_search
from .smart_analytics_orchestrator import process_smart_analytics

logger = logging.getLogger(__name__)

class ChatAgentState(BaseModel):
    # Standard LangGraph messages for conversation context
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Base fields
    session_id: str

    # User intent (determined in node_intent)
    intent: Optional[str] = None

    # User preferences and context tracking
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    seen_books: set = Field(default_factory=set)  # Books mentioned/recommended in this session

    # Results (unified for chat and search)
    results: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

    # Legacy fields for compatibility (will be deprecated)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    reply_message: Optional[str] = None
    should_search: bool = False
    should_recommend: bool = False

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

    # Get current message content
    current_message = state.messages[-1] if state.messages else None
    if not current_message:
        logger.error("❌ [node_chat_response_simple] No messages in state")
        return state

    message_content = current_message.content

    try:
        # Simple responses based on keywords
        message_lower = message_content.lower()
        
        if any(greeting in message_lower for greeting in ["привет", "здравствуй", "hello", "hi", "greetings"]):
            reply = "Hello! How are you? How can I help you with books?"
        elif any(phrase in message_lower for phrase in ["не знаю", "скучно", "что делать", "don't know", "boring", "what to do"]):
            reply = "I understand! Maybe let's read something interesting? What genre do you like?"
        else:
            reply = "Interesting! Tell me more - what exactly interests you?"
        
        # Add AI response to messages (standard LangGraph approach)
        state.messages.append(AIMessage(content=reply))

        # Form result
        state.results = [{
            "message": reply,
            "intent": "chat",
            "chat_mode": True
        }]

        # Add to history (legacy compatibility)
        if not state.chat_history:
            state.chat_history = []
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Error in chat_response_simple: {e}")
        error_message = "An error occurred. How can I help?"

        # Add error message to messages
        state.messages.append(AIMessage(content=error_message))

        state.results = [{
            "message": error_message,
            "intent": "error"
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
        "chat_mode": True
    }]
    
    execution_time = (time.time() - start_time) * 1000
    state.performance_metrics["chat_response_ms"] = execution_time
    logger.info(f"⏱️ [chat_agent] Chat response formed in {execution_time:.1f}ms")
    
    return state

async def node_recommendations(state: ChatAgentState) -> ChatAgentState:
    """Generate personalized recommendations using smart recommendation orchestrator"""
    logger.info("💡 [chat_agent] node_recommendations - Using smart recommendation orchestrator...")

    start_time = time.time()

    try:
        # Get current message content
        current_message = state.messages[-1] if state.messages else None
        if not current_message:
            logger.error("❌ [node_recommendations] No messages in state")
            return state

        message_content = current_message.content

        # Convert messages to chat history format
        chat_history = []
        for msg in state.messages[:-1]:  # Exclude current message
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            chat_history.append({"role": role, "content": msg.content})

        # Get books seen in this conversation to exclude from recommendations
        seen_books = set()
        if hasattr(state, 'seen_books') and state.seen_books:
            seen_books = state.seen_books

        # Extract seen book titles from previous results in this session
        for msg in state.messages:
            if isinstance(msg, AIMessage) and "**" in msg.content:
                # Simple extraction of book titles from previous responses
                import re
                book_titles = re.findall(r'\*\*(.*?)\*\*', msg.content)
                seen_books.update(book_titles)

        # Use smart recommendation orchestrator
        from .smart_recommendation_orchestrator import process_smart_recommendations

        recommendation_result = await process_smart_recommendations(
            session_id=state.session_id,
            current_message=message_content,
            chat_history=chat_history,
            user_preferences=state.user_preferences if hasattr(state, 'user_preferences') else None,
            exclude_books=seen_books
        )

        # Extract response and update state
        reply = recommendation_result.get("response", "Sorry, I couldn't generate recommendations right now.")
        recommendations = recommendation_result.get("recommendations", [])

        # Update seen books with new recommendations
        if not hasattr(state, 'seen_books'):
            state.seen_books = set()

        for rec in recommendations:
            if rec.get("title"):
                state.seen_books.add(rec["title"])

        # Add AI response to messages (standard LangGraph approach)
        state.messages.append(AIMessage(content=reply))

        # Form structured result
        state.results = [{
            "message": reply,
            "intent": "recommendations",
            "recommendation_mode": True,
            "recommendations": recommendations,
            "recommendation_stats": recommendation_result.get("recommendation_stats", {}),
            "performance_metrics": recommendation_result.get("performance_metrics", {})
        }]

        # Add to chat history (legacy compatibility)
        if not state.chat_history:
            state.chat_history = []

        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })

        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["recommendations_ms"] = execution_time

        logger.info(f"✅ [chat_agent] Smart recommendations generated in {execution_time:.1f}ms")
        logger.info(f"📊 [chat_agent] Generated {len(recommendations)} recommendations")

        return state

    except Exception as e:
        logger.error(f"❌ [chat_agent] Smart recommendation error: {e}")

        # Return error as result
        error_message = "Sorry, I couldn't generate personalized recommendations right now. Try asking for specific genres or authors!"

        # Add error message to messages
        state.messages.append(AIMessage(content=error_message))

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
    # Get current message from LangGraph messages
    current_message = state.messages[-1] if state.messages else None
    if not current_message:
        logger.error("❌ [node_intent] No messages in state")
        state.intent = "CHAT"
        return state

    message_content = current_message.content

    logger.info(f"🧠 [node_intent] Intent analysis: '{message_content}'")
    logger.info(f"📜 [node_intent] Message history: {len(state.messages)} messages")
    logger.info(f"🆔 [node_intent] Session ID: {state.session_id}")

    # Format conversation context from messages
    context = ""
    if len(state.messages) > 1:
        # Get last few messages for context (excluding current message)
        recent_messages = state.messages[-6:-1]  # Last 5 messages before current
        context_lines = []
        for msg in recent_messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            context_lines.append(f"{role}: {msg.content}")
        context = "\n".join(context_lines)
    
    try:
        # Build router prompt with conversation context
        if context:
            router_prompt = f"""You are a book intent classifier. Analyze the user's query carefully to determine their true intent.

🔍 SEARCH - User wants to FIND SPECIFIC books based on concrete criteria:
- "find books by Stephen King", "show me sci-fi novels"
- "I'm looking for mystery books from the 90s"
- "books about artificial intelligence", "history books"
- Any specific author, title, genre, topic, ISBN, year
- "What books do you have about X?"
- References to specific books: "books like 1984", "similar to Harry Potter"

📊 ANALYTICS - User wants DATA/STATISTICS about the collection:
- "how many books", "count", "statistics", "analyze"
- "what genres do you have", "top authors", "most popular"
- "show me trends", "collection overview"

💡 RECOMMEND - User wants PERSONAL SUGGESTIONS and doesn't know what specifically:
- "recommend something", "what should I read", "suggest books"
- "I'm bored, what's good", "surprise me", "advise something"
- "help me choose", "what would you recommend"
- "I need book recommendations", "give me ideas"
- No specific criteria - just wants curated suggestions

❓ CLARIFY - Very vague requests needing more information:
- "find a book" (no details), "I want to read" (no specifics)
- "something interesting" (too general)

💬 CHAT - Casual conversation not about finding/recommending books:
- Greetings, personal questions, general chat

CONVERSATION CONTEXT:
{context}

CURRENT QUERY: "{message_content}"

ANALYSIS RULES:
1. If user specifies WHAT they want (genre/author/topic) → SEARCH
2. If user asks for personal suggestions without specifics → RECOMMEND
3. If user wants data about collection → ANALYTICS
4. Context matters: "more like this" after recommendations = SEARCH for similar books

Answer (SEARCH/ANALYTICS/RECOMMEND/CLARIFY/CHAT):"""
        else:
            router_prompt = f"""You are a book intent classifier. Analyze the user's query carefully to determine their true intent.

🔍 SEARCH - User wants to FIND SPECIFIC books based on concrete criteria:
- "find books by Stephen King", "show me sci-fi novels"
- "I'm looking for mystery books from the 90s"
- "books about artificial intelligence", "history books"
- Any specific author, title, genre, topic, ISBN, year
- "What books do you have about X?"
- References to specific books: "books like 1984", "similar to Harry Potter"

📊 ANALYTICS - User wants DATA/STATISTICS about the collection:
- "how many books", "count", "statistics", "analyze"
- "what genres do you have", "top authors", "most popular"
- "show me trends", "collection overview"

💡 RECOMMEND - User wants PERSONAL SUGGESTIONS and doesn't know what specifically:
- "recommend something", "what should I read", "suggest books"
- "I'm bored, what's good", "surprise me", "advise something"
- "help me choose", "what would you recommend"
- "I need book recommendations", "give me ideas"
- No specific criteria - just wants curated suggestions

❓ CLARIFY - Very vague requests needing more information:
- "find a book" (no details), "I want to read" (no specifics)
- "something interesting" (too general)

💬 CHAT - Casual conversation not about finding/recommending books:
- Greetings, personal questions, general chat

Query: "{message_content}"

ANALYSIS RULES:
1. If user specifies WHAT they want (genre/author/topic) → SEARCH
2. If user asks for personal suggestions without specifics → RECOMMEND
3. If user wants data about collection → ANALYTICS

Answer (SEARCH/ANALYTICS/RECOMMEND/CLARIFY/CHAT):"""
        
        response = llm.invoke([("user", router_prompt)]).content.strip().upper()
        
        logger.info(f"🧠 [router] LLM response: '{response}'")
        
        # Save analysis result in state
        state.intent = response

        # Extract user preferences from chat context if this is a RECOMMEND intent
        if "RECOMMEND" in response:
            logger.info("💡 [node_intent] Extracting user preferences from chat context")
            try:
                from .smart_recommendation_orchestrator import _extract_user_preferences

                # Convert messages to chat history format for preference extraction
                chat_history = []
                for msg in state.messages[:-1]:  # Exclude current message
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    chat_history.append({"role": role, "content": msg.content})

                # Extract preferences asynchronously
                extracted_preferences = await _extract_user_preferences(message_content, chat_history)
                state.user_preferences = extracted_preferences

                logger.info(f"🎯 [node_intent] Extracted preferences: {extracted_preferences}")

            except Exception as e:
                logger.warning(f"⚠️ [node_intent] Failed to extract preferences: {e}")
                # Initialize empty preferences if extraction fails
                state.user_preferences = {
                    "likes": {"genres": [], "authors": [], "themes": [], "book_types": []},
                    "dislikes": {"genres": [], "authors": [], "themes": []},
                    "context": {"mood": "general reading", "situation": "leisure", "goal": "entertainment"},
                    "recommendation_type": "discovery"
                }

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
    elif "ANALYTICS" in intent:
        logger.info(f"📊 [route_by_intent] → analytics")
        return "analytics"
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
        # Get current message content
        current_message = state.messages[-1] if state.messages else None
        if not current_message:
            logger.error("❌ [node_simple_search] No messages in state")
            return state

        message_content = current_message.content

        # Extract user preferences from conversation context for preference-aware filtering
        user_preferences = None
        if len(state.messages) > 1:
            try:
                from .smart_recommendation_orchestrator import _extract_user_preferences

                # Convert messages to chat history format for preference extraction
                chat_history = []
                for msg in state.messages[:-1]:  # Exclude current message
                    role = "assistant" if isinstance(msg, AIMessage) else "user"
                    chat_history.append({"role": role, "content": msg.content})

                # Extract preferences asynchronously (lightweight for search)
                user_preferences = await _extract_user_preferences(message_content, chat_history)
                logger.info(f"🎯 [node_simple_search] Extracted preferences for filtering: {user_preferences}")

            except Exception as e:
                logger.warning(f"⚠️ [node_simple_search] Failed to extract preferences: {e}")

        # Call simple search with preference context
        search_result = await process_simple_search(state.session_id, message_content, user_preferences)

        # Use ready formatted response from simple_orchestrator
        response_text = search_result.get('response', 'Results not found')

        # Add AI response to messages (standard LangGraph approach)
        state.messages.append(AIMessage(content=response_text))

        # Add to results for compatibility
        state.results = [{
            "message": response_text,
            "intent": search_result.get('intent', 'search'),
            "search_mode": True,
            "original_response": search_result.get('response'),
            "raw_results": search_result.get('results', [])
        }]

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
        error_message = "Sorry, a search error occurred. Try rephrasing your query."

        # Add error message to messages
        state.messages.append(AIMessage(content=error_message))

        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]

        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["simple_search_error_ms"] = execution_time

        return state

async def node_analytics(state: ChatAgentState) -> ChatAgentState:
    """Performs analytics using process_smart_analytics"""
    logger.info("📊 [chat_agent] node_analytics - Using smart_analytics_orchestrator...")

    start_time = time.time()

    try:
        # Get current message content
        current_message = state.messages[-1] if state.messages else None
        if not current_message:
            logger.error("❌ [node_analytics] No messages in state")
            return state

        message_content = current_message.content

        # Call analytics processing
        analytics_result = await process_smart_analytics(state.session_id, message_content)

        # Use ready formatted response from analytics_orchestrator
        response_text = analytics_result.get('response', 'Analytics not available')

        # Add AI response to messages (standard LangGraph approach)
        state.messages.append(AIMessage(content=response_text))

        # Add to results for compatibility
        state.results = [{
            "message": response_text,
            "intent": analytics_result.get('intent', 'analytics'),
            "analytics_mode": True,
            "analytics_stats": analytics_result.get('analytics_stats', {}),
            "raw_results": analytics_result.get('results', [])
        }]

        # Copy performance metrics
        if analytics_result.get('performance_metrics'):
            state.performance_metrics.update(analytics_result['performance_metrics'])

        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["analytics_ms"] = execution_time

        logger.info(f"✅ [chat_agent] Analytics completed in {execution_time:.1f}ms")

        return state

    except Exception as e:
        logger.error(f"❌ [chat_agent] Analytics error: {e}")

        # Return error as result
        error_message = "Sorry, analytics processing failed. Try rephrasing your query."

        # Add error message to messages
        state.messages.append(AIMessage(content=error_message))

        state.results = [{
            "message": error_message,
            "intent": "error",
            "error": str(e)
        }]

        execution_time = (time.time() - start_time) * 1000
        state.performance_metrics["analytics_error_ms"] = execution_time

        return state

def node_clarify(state: ChatAgentState) -> ChatAgentState:
    """Requests clarification from user"""
    start_time = time.time()

    logger.info("❓ [chat_agent] node_clarify - Clarification request...")

    # Get current message content
    current_message = state.messages[-1] if state.messages else None
    if not current_message:
        logger.error("❌ [node_clarify] No messages in state")
        return state

    message_content = current_message.content

    try:
        # Form response with clarification request
        message_lower = message_content.lower()
        
        if any(word in message_lower for word in ["найти", "найду", "ищу", "поиск"]):  # Russian: найти=find, найду=will find, ищу=searching, поиск=search
            reply = "What exactly do you want to find? Specify author, book title or genre."
        elif any(word in message_lower for word in ["книг", "читать", "литератур"]):  # Russian: книг=books, читать=read, литератур=literature
            reply = "What type of books interest you? You can specify genre or specific preferences."
        else:
            reply = "I can help find books! What exactly interests you?"
        
        # Add AI response to messages (standard LangGraph approach)
        state.messages.append(AIMessage(content=reply))

        # Form result
        state.results = [{
            "message": reply,
            "intent": "clarify",
            "clarify_mode": True
        }]

        # Add to history (legacy compatibility)
        if not state.chat_history:
            state.chat_history = []
        state.chat_history.append({
            "role": "assistant",
            "content": reply
        })
        
    except Exception as e:
        logger.error(f"❌ Error in node_clarify: {e}")
        error_message = "How can I help? Looking for something specific?"

        # Add error message to messages
        state.messages.append(AIMessage(content=error_message))

        state.results = [{
            "message": error_message,
            "intent": "error"
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
builder.add_node("analytics", node_analytics)
builder.add_node("clarify", node_clarify)

# New architecture: first intent analysis, then routing
builder.set_entry_point("intent")

# Conditional routing after intent analysis
builder.add_conditional_edges(
    "intent",
    route_by_intent,
    {
        "simple_search": "simple_search",
        "analytics": "analytics",
        "recommendations": "recommendations",
        "chat_response": "chat_response",
        "clarify": "clarify"
    }
)

# Completion
builder.add_edge("chat_response", END)
builder.add_edge("recommendations", END)
builder.add_edge("simple_search", END)
builder.add_edge("analytics", END)
builder.add_edge("clarify", END)

# Compile graph with checkpointer for state preservation between calls
memory = MemorySaver()
chat_agent_graph = builder.compile(checkpointer=memory)