from typing import Optional, Dict, Any, List, Annotated
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from ..infra.settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from ..services.search_service import process_simple_search
from ..services.analytics_service import process_smart_analytics

logger = logging.getLogger(__name__)

class ChatAgentState(BaseModel):
    # Standard LangGraph messages for conversation context
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Base fields
    session_id: str

    # User intent (determined in node_intent)
    intent: Optional[str] = None
    previous_intent: Optional[str] = None
    conversation_mode: str = "discovery"

    # Enhanced context tracking for intent determination
    intent_history: List[str] = Field(default_factory=list)  # Track all intents in order
    last_successful_mode: Optional[str] = None  # Last mode that provided results
    conversation_turns: int = 0  # Count of user messages in this session

    # User preferences and context tracking
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    current_preferences: Optional[Dict[str, Any]] = None  # Latest extracted preferences for "еще" fallback
    previous_criteria: set = Field(default_factory=set)  # Previous themes/genres for change detection
    seen_books: List[str] = Field(default_factory=list)  # Book IDs mentioned/recommended in this session

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

# LLM clients with different temperatures for different tasks
llm_generative = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.7, api_key=OPENAI_API_KEY)  # For creative responses
llm_deterministic = ChatOpenAI(model=OPENAI_MODEL_CHAT, temperature=0.0, api_key=OPENAI_API_KEY)  # For intent classification

async def extract_and_update_preferences(
    state: ChatAgentState,
    message_content: str,
    chat_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Universal preference extraction and state update function for both search and recommendations."""
    try:
        from ..services.recommendation_service import _extract_user_preferences

        # Extract preferences using the same LLM and prompt as recommendations
        extracted_preferences = await _extract_user_preferences(message_content, chat_history)

        # Smart preference update function
        user_preferences = _smart_preference_update(state, extracted_preferences, message_content)

        # Update state with final preferences
        state.user_preferences = user_preferences
        state.current_preferences = user_preferences

        # Check if user changed themes/criteria for filter management
        current_themes = user_preferences.get("likes", {}).get("themes", [])
        current_genres = user_preferences.get("likes", {}).get("genres", [])
        current_criteria = set(current_themes + current_genres)

        # Get previous criteria from state
        previous_criteria = getattr(state, 'previous_criteria', set())

        # Determine if criteria changed significantly
        criteria_changed = bool(current_criteria and previous_criteria and not current_criteria.intersection(previous_criteria))
        should_clear_filters = criteria_changed

        if should_clear_filters:
            logger.info(f"📝 [extract_preferences] Criteria changed from {previous_criteria} to {current_criteria}, clearing filters")
            # Clear seen books for new topic
            if hasattr(state, 'seen_books'):
                state.seen_books = []

        # Store current criteria for next comparison
        state.previous_criteria = current_criteria

        logger.info(f"🎯 [extract_preferences] Final preferences: {user_preferences}")

        return {
            "user_preferences": user_preferences,
            "criteria_changed": criteria_changed,
            "should_clear_filters": should_clear_filters,
            "current_criteria": current_criteria,
            "previous_criteria": previous_criteria
        }

    except Exception as e:
        logger.error(f"❌ [extract_preferences] Failed: {e}")
        # Return default preferences on error
        default_preferences = {
            "likes": {"genres": [], "authors": [], "themes": [], "book_types": []},
            "dislikes": {"genres": [], "authors": [], "themes": []},
            "context": {"mood": "general reading", "situation": "leisure", "goal": "entertainment"},
            "recommendation_type": "discovery"
        }
        state.user_preferences = default_preferences

        return {
            "user_preferences": default_preferences,
            "criteria_changed": False,
            "should_clear_filters": False,
            "current_criteria": set(),
            "previous_criteria": set()
        }


def _smart_preference_update(
    state: ChatAgentState,
    extracted_preferences: Dict[str, Any],
    message_content: str
) -> Dict[str, Any]:
    """Simply decides whether to use extracted preferences or preserve previous ones."""

    # Check if LLM extracted any meaningful new preferences
    extracted_likes = extracted_preferences.get("likes", {})
    has_new_preferences = any([
        extracted_likes.get("themes"),
        extracted_likes.get("genres"),
        extracted_likes.get("authors")
    ])

    logger.info(f"🔧 [smart_update] Message: '{message_content}'")
    logger.info(f"🔧 [smart_update] Has new preferences: {has_new_preferences}")

    if has_new_preferences:
        # New preferences found - use them
        logger.info(f"🔄 [smart_update] Using new extracted preferences")
        return extracted_preferences
    else:
        # No new preferences - use previous if available
        if hasattr(state, 'current_preferences') and state.current_preferences:
            logger.info(f"🔄 [smart_update] No new preferences, using previous")
            return state.current_preferences
        else:
            logger.info(f"🔄 [smart_update] No preferences available, using extracted")
            return extracted_preferences


async def _generate_smart_clarification(message_content: str, state: ChatAgentState) -> str:
    """Generate intelligent clarification questions based on available metadata"""
    try:
        # Get available metadata for context-aware clarification
        from ..services.vector_retriever import SimpleVectorRetriever
        retriever = SimpleVectorRetriever()
        available_books = await retriever.get_all_books_metadata()

        # Extract unique metadata values for intelligent suggestions
        authors = set()
        genres = set()
        topics = set()
        years = set()

        for book in available_books[:50]:  # Sample for performance
            if book.get("author"):
                authors.add(book["author"])
            if book.get("primary_genre"):
                genres.add(book["primary_genre"])
            if book.get("topics"):
                topics.update(book["topics"][:3])  # First 3 topics
            if book.get("year"):
                years.add(str(book["year"]))

        # Create metadata context for LLM
        metadata_context = {
            "sample_authors": list(authors)[:10],
            "available_genres": list(genres)[:10],
            "popular_topics": list(topics)[:10],
            "year_range": f"{min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}"
        }

        clarification_prompt = f"""You are a helpful librarian assistant. The user said: "{message_content}"

This query is unclear or too vague. Generate a helpful clarification question that guides the user to be more specific.

AVAILABLE METADATA:
- Authors: {', '.join(metadata_context['sample_authors'])}
- Genres: {', '.join(metadata_context['available_genres'])}
- Topics: {', '.join(metadata_context['popular_topics'])}
- Years: {metadata_context['year_range']}

Generate a friendly clarification question that:
1. Acknowledges their request
2. Offers specific options based on available metadata
3. Helps them narrow down their search

Be conversational and helpful. Suggest concrete options they can choose from."""

        response = await llm_generative.ainvoke([("user", clarification_prompt)])
        return response.content.strip()

    except Exception as e:
        logger.error(f"❌ Error generating smart clarification: {e}")
        return "I'd love to help you find books! Could you be more specific about what you're looking for? You can mention author names, genres, or topics that interest you."


async def _generate_engaging_chat_response(message_content: str, state: ChatAgentState) -> str:
    """Generate engaging, human-like chat response that motivates book exploration"""
    try:
        # Build conversation context
        conversation_context = ""
        if len(state.messages) > 1:
            recent_messages = state.messages[-4:-1]  # Last 3 messages for context
            context_lines = []
            for msg in recent_messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                context_lines.append(f"{role}: {msg.content}")
            conversation_context = "\n".join(context_lines)

        chat_prompt = f"""You are a friendly, passionate librarian who loves books and reading. You're having a casual conversation with someone.

CONVERSATION CONTEXT:
{conversation_context if conversation_context else "This is the start of the conversation"}

USER'S MESSAGE: "{message_content}"

YOUR PERSONALITY:
- Warm, enthusiastic, and genuinely interested in people
- Passionate about books but not pushy
- Natural conversationalist who responds authentically
- Subtly guides conversations toward reading and books
- Uses casual, friendly language

RESPONSE GUIDELINES:
1. Respond naturally to what they said (acknowledge their message)
2. Show genuine interest and empathy
3. Gently connect to books/reading when appropriate
4. Ask engaging follow-up questions
5. Be conversational, not robotic
6. If they seem bored/lost, suggest books as a solution
7. Share enthusiasm for reading in a natural way

Generate a warm, human response that feels like talking to a friend who happens to love books."""

        response = await llm_generative.ainvoke([("user", chat_prompt)])
        return response.content.strip()

    except Exception as e:
        logger.error(f"❌ Error generating engaging chat response: {e}")
        return "That's interesting! You know, I find that a good book can often help with whatever we're going through. What kind of stories do you usually enjoy?"


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

async def node_chat_response_simple(state: ChatAgentState) -> ChatAgentState:
    """Intelligent chat response with book motivation"""
    start_time = time.time()

    logger.info("💬 [chat_agent] node_chat_response_simple - Intelligent chat response...")

    # Get current message content
    current_message = state.messages[-1] if state.messages else None
    if not current_message:
        logger.error("❌ [node_chat_response_simple] No messages in state")
        return state

    message_content = current_message.content

    try:
        # Generate human-like response that guides to books
        reply = await _generate_engaging_chat_response(message_content, state)
        
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
        seen_book_ids = set(state.seen_books) if hasattr(state, 'seen_books') and state.seen_books else set()
        logger.info(f"🔍 [node_recommendations] Current seen_books: {list(seen_book_ids)} (total: {len(seen_book_ids)})")

        # Note: We now track book IDs from structured results, not by parsing Markdown
        # This eliminates format-dependent tracking and ensures stable deduplication

        # Extract user preferences using universal function
        logger.info("🧠 [node_recommendations] Extracting user preferences from chat context...")
        pref_result = await extract_and_update_preferences(state, message_content, chat_history)
        user_preferences = pref_result["user_preferences"]

        # Use smart recommendation orchestrator
        from ..services.recommendation_service import process_smart_recommendations

        recommendation_result = await process_smart_recommendations(
            session_id=state.session_id,
            current_message=message_content,
            chat_history=chat_history,
            user_preferences=user_preferences,
            exclude_books=seen_book_ids
        )

        # Extract response and update state
        reply = recommendation_result.get("response", "Sorry, I couldn't generate recommendations right now.")
        recommendations = recommendation_result.get("recommendations", [])

        # Update seen books with new recommendations
        if not hasattr(state, 'seen_books'):
            state.seen_books = []

        # Track books by ID to ensure stable deduplication
        for rec in recommendations:
            book_id = rec.get("document_id")
            if book_id:
                # Add unique IDs only (list with deduplication)
                unique_books = set(state.seen_books)
                unique_books.add(book_id)
                state.seen_books = list(unique_books)

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

        # Update last successful mode for context tracking
        if recommendations:
            state.last_successful_mode = "RECOMMEND"

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

    # Count conversation turns (user messages only)
    user_message_count = sum(1 for msg in state.messages if isinstance(msg, HumanMessage))
    state.conversation_turns = user_message_count

    logger.info(f"🧠 [node_intent] Intent analysis: '{message_content}'")
    logger.info(f"📜 [node_intent] Message history: {len(state.messages)} messages")
    logger.info(f"🆔 [node_intent] Session ID: {state.session_id}")
    logger.info(f"🔄 [node_intent] Previous intent: {state.previous_intent}")
    logger.info(f"🎯 [node_intent] Conversation mode: {state.conversation_mode}")
    logger.info(f"📊 [node_intent] Intent history: {state.intent_history}")
    logger.info(f"🔗 [node_intent] Conversation turns: {state.conversation_turns}")
    logger.info(f"✅ [node_intent] Last successful mode: {state.last_successful_mode}")

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
        # Universal context-aware intent prompt
        router_prompt = f"""You are an intelligent book intent classifier. Analyze the user's current query within the conversation context to determine their true intent.

🔍 SEARCH - User wants to FIND SPECIFIC books based on concrete criteria:
- "find books by Stephen King", "show me sci-fi novels"
- "I'm looking for mystery books from the 90s"
- "books about artificial intelligence", "history books"
- Any specific author, title, genre, topic, ISBN, year
- "What books do you have about X?"
- References to specific books: "books like 1984", "similar to Harry Potter"

📊 ANALYTICS - User asks about LIBRARY METADATA (what exists in collection):
- Questions starting with "what genres/authors/topics do you have"
- Requests for information ABOUT the database content
- Meta-information queries about collection statistics

💡 RECOMMEND - User wants PERSONAL book suggestions:
- Asks for book recommendations or suggestions
- Wants personal advice on what to read
- Seeking curated suggestions without specific criteria

❓ CLARIFY - Very vague SEARCH requests needing more information:
- "find a book" (no search details), "I want to read" (no search specifics)
- Vague search queries that need clarification to proceed

💬 CHAT - Casual conversation, greetings, or general discussion:
- Greetings, personal questions, general chat
- Statements about not wanting to read or book preferences
- General life conversations unrelated to book searching

CONVERSATION CONTEXT:
Previous Intent: {state.previous_intent or "None"}
Intent History: {" → ".join(state.intent_history[-3:]) if state.intent_history else "None"} (last 3)
Conversation Mode: {state.conversation_mode}
Conversation Turns: {state.conversation_turns}
Last Successful Mode: {state.last_successful_mode or "None"}
Recent Messages:
{context if context else "No previous context"}

CURRENT QUERY: "{message_content}"

CONTEXT-AWARE ANALYSIS:
Use your intelligence to understand user intent. Prioritize explicit intent over context:

1. EXPLICIT INTENT FIRST: If user gives specific search criteria (titles, authors, genres), classify as SEARCH regardless of previous context
2. CONTEXT FOR AMBIGUITY: Only use previous intent for truly ambiguous queries
3. CLEAR TRANSITIONS: Users can switch between different intents - don't force continuity

Apply your natural language understanding to determine intent, giving priority to what the user explicitly asks for in their current message.

Answer only the intent (SEARCH/ANALYTICS/RECOMMEND/CLARIFY/CHAT):"""
        
        response = (await llm_deterministic.ainvoke([("user", router_prompt)])).content.strip().upper()

        logger.info(f"🧠 [router] LLM response: '{response}'")

        # Save previous intent before updating
        state.previous_intent = state.intent

        # Track intent in history
        if response and response not in ["CHAT", "CLARIFY"]:
            state.intent_history.append(response)
            # Keep only last 10 intents to avoid memory bloat
            if len(state.intent_history) > 10:
                state.intent_history = state.intent_history[-10:]

        # Save analysis result in state
        state.intent = response

        # Note: User preferences will be extracted in node_recommendations if needed

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

        # Extract user preferences using universal function
        chat_history = []
        for msg in state.messages[:-1]:  # Exclude current message
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            chat_history.append({"role": role, "content": msg.content})

        pref_result = await extract_and_update_preferences(state, message_content, chat_history)
        user_preferences = pref_result["user_preferences"]
        logger.info(f"🎯 [node_simple_search] Using preferences for filtering: {user_preferences}")

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

        # Update last successful mode for context tracking
        if state.results and state.results[0].get("raw_results"):
            state.last_successful_mode = "SEARCH"

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

async def node_clarify(state: ChatAgentState) -> ChatAgentState:
    """Intelligent clarification using metadata analysis"""
    start_time = time.time()

    logger.info("❓ [chat_agent] node_clarify - Intelligent clarification request...")

    # Get current message content
    current_message = state.messages[-1] if state.messages else None
    if not current_message:
        logger.error("❌ [node_clarify] No messages in state")
        return state

    message_content = current_message.content

    try:
        # Get metadata for intelligent clarification
        reply = await _generate_smart_clarification(message_content, state)

        logger.info(f"🤖 [node_clarify] Generated clarification: {reply}")
        
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