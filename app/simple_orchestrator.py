from typing import Optional, Dict, Any, List
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .simple_retriever import SimpleVectorRetriever
from .simple_llm_filter import SimpleLLMFilter

logger = logging.getLogger(__name__)

class SimpleSearchState(BaseModel):
    """State for simple search without complex logic"""
    session_id: str
    message: str
    # Search results
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    # Results after LLM filtering
    filtered_results: List[Dict[str, Any]] = Field(default_factory=list)
    # Analysis from LLM
    intent: Optional[str] = None
    analysis: Optional[str] = None
    negative_filters: Optional[str] = None
    uncertainty_note: Optional[str] = None
    # Final response
    final_response: Optional[str] = None
    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    # Errors
    error: Optional[str] = None

# Component initialization
simple_retriever = SimpleVectorRetriever()
llm_filter = SimpleLLMFilter()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

def search_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 1: Simple vector search"""
    logger.info(f"🔍 [Step 1] Simple search for: '{state.message}'")
    
    start_time = time.time()
    
    try:
        # Execute async search using concurrent.futures
        import concurrent.futures
        import asyncio
        
        def run_search():
            # Create new event loop in separate thread
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(simple_retriever.search(state.message, k=5))
            finally:
                loop.close()
        
        # Execute in separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_search)
            search_results = future.result()
        
        state.search_results = search_results
        state.performance_metrics['search_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 1] Found {len(search_results)} results in {state.performance_metrics['search_time']:.2f}s")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 1] Search error: {e}")
        state.error = f"Search error: {e}"
        state.performance_metrics['search_time'] = time.time() - start_time
        return state

def filter_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 2: LLM filtering and analysis"""
    logger.info(f"🧠 [Step 2] LLM filtering {len(state.search_results)} results")
    
    start_time = time.time()
    
    try:
        # If there's an error on previous step, skip
        if state.error:
            logger.warning("⚠️ [Step 2] Skipping due to error on previous step")
            return state
            
        # If no search results, skip filtering
        if not state.search_results:
            logger.info("ℹ️ [Step 2] No results to filter")
            state.filtered_results = []
            state.intent = "no_results"
            state.analysis = "Search results not found"
            state.performance_metrics['filter_time'] = time.time() - start_time
            return state
        
        # Execute LLM filtering
        import concurrent.futures
        import asyncio
        
        def run_filter():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(llm_filter.filter_and_analyze(state.message, state.search_results))
            finally:
                loop.close()
        
        # Execute in separate thread
        llm_filter_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_filter)
            filter_result = future.result()
        llm_filter_time = time.time() - llm_filter_start
        logger.info(f"⚡ LLM filter in {llm_filter_time:.3f}s")
        
        # Update state
        state.filtered_results = filter_result['filtered_results']
        state.intent = filter_result['intent']
        state.analysis = filter_result['analysis']
        state.negative_filters = filter_result.get('negative_filters', '')
        state.uncertainty_note = filter_result.get('uncertainty_note', '')
        state.performance_metrics['filter_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 2] LLM analysis: intent='{state.intent}', "
                   f"filtered {len(state.filtered_results)}/{len(state.search_results)} in {state.performance_metrics['filter_time']:.2f}s")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 2] LLM filtering error: {e}")
        # In case of error return all results
        state.filtered_results = state.search_results
        state.intent = "filter_error"
        state.analysis = f"Filtering error: {e}"
        state.performance_metrics['filter_time'] = time.time() - start_time
        return state

def format_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 3: Format final response"""
    logger.info(f"💬 [Step 3] Formatting response for {len(state.filtered_results)} results")
    
    start_time = time.time()
    
    try:
        # If there's a critical error, return error
        if state.error and not state.filtered_results:
            state.final_response = f"Sorry, an error occurred during search: {state.error}"
            state.performance_metrics['format_time'] = time.time() - start_time
            return state
        
        # If no results
        if not state.filtered_results:
            state.final_response = _format_no_results_response(state.message)
            state.performance_metrics['format_time'] = time.time() - start_time
            return state
        
        # Generate response using LLM
        import concurrent.futures
        import asyncio
        
        def run_format():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_generate_formatted_response(
                    state.message, 
                    state.filtered_results, 
                    state.intent,
                    state.analysis,
                    state.uncertainty_note
                ))
            finally:
                loop.close()
        
        # Execute in separate thread
        llm_format_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_format)
            formatted_response = future.result()
        llm_format_time = time.time() - llm_format_start
        logger.info(f"⚡ LLM format in {llm_format_time:.3f}s")
        
        state.final_response = formatted_response
        state.performance_metrics['format_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 3] Response formatted in {state.performance_metrics['format_time']:.2f}s")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 3] Formatting error: {e}")
        # Fallback to simple formatting
        state.final_response = _format_simple_response(state.filtered_results)
        state.performance_metrics['format_time'] = time.time() - start_time
        return state

async def _generate_formatted_response(query: str, results: List[Dict], intent: str, analysis: str, uncertainty_note: str = "") -> str:
    """Generates formatted response using LLM"""
    
    # Minimal information for quick formatting with content
    results_info = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Unknown')
        author = result.get('author', 'Unknown')
        content = result.get('content', '')
        
        result_text = f"""{i}. "{title}" - {author}"""
        
        # Add year only if available
        metadata = result.get('metadata', {})
        year = metadata.get('year', '')
        if year:
            result_text += f" ({year})"
        
        # Add full content 
        if content:
            result_text += f"\n   {content}"
        
        results_info.append(result_text)
    
    results_text = "\n\n".join(results_info)
    
    # Brief motivating prompt for reading
    system_prompt = """You are an experienced librarian. Create a brief and motivating response in English.

IMPORTANT: Response should be short (200-300 characters)!

For each book write:
1. "📖 «Title» — Author (year)"
2. Brief plot description (1-2 sentences)
3. Why it's worth reading

Be concise but inspiring!"""

    user_prompt = f"""User is looking for: "{query}"

Found books:
{results_text}

Create an inspiring response that motivates reading!"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        logger.info(f"📝 LLM generated response: {response.content[:200]}...")
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        logger.info("🔄 Using fallback formatting")
        return _format_simple_response(results)

def _format_simple_response(results: List[Dict]) -> str:
    """Simple formatting without LLM in case of error"""
    if not results:
        return "Unfortunately, no suitable books found."
    
    lines = [f"Found {len(results)} book(s):"]
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Unknown')
        author = result.get('author', 'Unknown')
        
        line = f"{i}. \"{title}\" - {author}"
        
        # Add year if available
        metadata = result.get('metadata', {})
        year = metadata.get('year', '')
        if year:
            line += f" ({year})"
            
        lines.append(line)
    
    return "\n".join(lines)

def _format_no_results_response(query: str) -> str:
    """Formats response when no results found"""
    return f"""Unfortunately, no books found for query "{query}".

Try:
- Check spelling of title or author
- Use more general terms
- Try search by genre or topic

I can help with another query!"""

# Graph creation
def create_simple_search_graph():
    """Creates graph for simple search"""
    
    workflow = StateGraph(SimpleSearchState)
    
    # Add nodes
    workflow.add_node("search", search_step)
    workflow.add_node("filter", filter_step) 
    workflow.add_node("format", format_step)
    
    # Define flow
    workflow.set_entry_point("search")
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()

# Create graph
simple_search_graph = create_simple_search_graph()

async def process_simple_search(session_id: str, message: str) -> Dict[str, Any]:
    """
    Main function for processing simple search
    
    Args:
        session_id: session identifier
        message: user message
        
    Returns:
        Dict with search results
    """
    logger.info(f"🚀 [Simple Search] Starting processing: session_id={session_id}, query='{message}'")
    
    total_start_time = time.time()
    
    try:
        # Create initial state
        initial_state = SimpleSearchState(
            session_id=session_id,
            message=message
        )
        
        # Execute graph
        final_state = simple_search_graph.invoke(initial_state)
        
        # Total time
        total_time = time.time() - total_start_time
        
        # Process result (can be dict or object)
        if hasattr(final_state, 'performance_metrics'):
            final_state.performance_metrics['total_time'] = total_time
            performance_metrics = final_state.performance_metrics
            search_results = final_state.search_results
            filtered_results = final_state.filtered_results
            final_response = final_state.final_response
            intent = final_state.intent
            analysis = final_state.analysis
            uncertainty_note = getattr(final_state, 'uncertainty_note', '')
        else:
            # Если final_state это dict
            performance_metrics = final_state.get('performance_metrics', {})
            performance_metrics['total_time'] = total_time
            search_results = final_state.get('search_results', [])
            filtered_results = final_state.get('filtered_results', [])
            final_response = final_state.get('final_response', 'Ошибка обработки')
            intent = final_state.get('intent', 'error')
            analysis = final_state.get('analysis', 'Анализ недоступен')
            uncertainty_note = final_state.get('uncertainty_note', '')
        
        # Формируем результат
        result = {
            "response": final_response,
            "results": [
                {
                    "title": r.get('title', 'Unknown'),
                    "author": r.get('author', 'Unknown'),
                    "score": r.get('score', 0.0),
                    "collection": r.get('collection', 'unknown')
                }
                for r in filtered_results
            ],
            "intent": intent,
            "analysis": analysis,
            "uncertainty_note": uncertainty_note,
            "performance_metrics": performance_metrics,
            "search_stats": {
                "total_found": len(search_results),
                "after_filter": len(filtered_results),
                "total_time": total_time
            }
        }
        
        logger.info(f"🎯 [Simple Search] Завершено за {total_time:.2f}с: "
                   f"найдено {len(search_results)}, отфильтровано {len(filtered_results)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ [Simple Search] Критическая ошибка: {e}")
        
        return {
            "response": f"Извините, произошла ошибка при обработке запроса: {e}",
            "results": [],
            "intent": "error",
            "analysis": f"Критическая ошибка: {e}",
            "performance_metrics": {"total_time": time.time() - total_start_time},
            "search_stats": {
                "total_found": 0,
                "after_filter": 0,
                "total_time": time.time() - total_start_time
            }
        }