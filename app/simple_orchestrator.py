from typing import Optional, Dict, Any, List
import logging
import time
import json
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from .settings import OPENAI_API_KEY
from .simple_retriever import SimpleVectorRetriever

logger = logging.getLogger(__name__)

class SimpleSearchState(BaseModel):
    """State for simple search without complex logic"""
    session_id: str
    message: str
    user_preferences: Optional[Dict[str, Any]] = None
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
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)
filter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

@traceable(name="search_step")
async def search_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 1: Simple vector search"""
    logger.info(f"🔍 [Step 1] Simple search for: '{state.message}'")

    # Add trace metadata
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_tags(["search", "vector_search"])
        current_run.add_inputs({"query": state.message, "session_id": state.session_id})
        current_run.add_metadata({"step": "search", "query_length": len(state.message)})

    start_time = time.time()
    
    try:
        # Execute async search directly
        search_results = await simple_retriever.search(state.message, k=5)
        
        state.search_results = search_results
        state.performance_metrics['search_time'] = time.time() - start_time

        # Update trace with results
        if current_run:
            current_run.add_outputs({"num_results": len(search_results), "search_time": state.performance_metrics['search_time']})
            current_run.add_metadata({
                "results_found": len(search_results),
                "execution_time_s": state.performance_metrics['search_time']
            })

        logger.info(f"✅ [Step 1] Found {len(search_results)} results in {state.performance_metrics['search_time']:.2f}s")

        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 1] Search error: {e}")
        state.error = f"Search error: {e}"
        state.performance_metrics['search_time'] = time.time() - start_time
        return state

async def _execute_llm_filtering(query: str, search_results: List[Dict[str, Any]], user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute LLM filtering directly without wrapper"""

    logger.info(f"🧠 LLM analysis of query: '{query}' for {len(search_results)} results")

    if not search_results:
        return {
            "filtered_results": [],
            "intent": "no_results",
            "analysis": "No results to analyze",
            "total_found": 0,
            "total_filtered": 0
        }

    try:
        # Prepare data for LLM
        results_summary = _prepare_results_for_llm(search_results)

        # Create prompts
        system_prompt = _create_filter_system_prompt(user_preferences)
        user_prompt = _create_filter_user_prompt(query, results_summary, user_preferences)

        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = await filter_llm.ainvoke(messages)

        # Parse LLM response
        analysis_result = _parse_filter_llm_response(response.content, search_results)

        logger.info(f"✅ LLM analysis: intent='{analysis_result['intent']}', "
                   f"filtered={analysis_result['total_filtered']}/{analysis_result['total_found']}")

        return analysis_result

    except Exception as e:
        logger.error(f"❌ Error in LLM filter: {e}")
        # Return all results in case of error
        return {
            "filtered_results": search_results,
            "intent": "error",
            "analysis": f"Analysis error: {e}",
            "total_found": len(search_results),
            "total_filtered": len(search_results)
        }

def _create_filter_system_prompt(user_preferences: Optional[Dict[str, Any]] = None) -> str:
    """Creates analytical system prompt for filtering"""
    base_prompt = """You are a library search filter. Your job is simple: find which books match the user's query.

FOR ISBN QUERIES (highest priority):
- If user query contains numbers like "978-12-345-678-9" or "9781234567890"
- Look for "ISBN:" in each book's content
- Remove dashes from both: query "978-12-345-678-9" becomes "9781234567890"
- Remove dashes from content: "ISBN: 9781234567890" becomes "9781234567890"
- If numbers match exactly → INCLUDE that book index in filtered_indices

EXAMPLE:
Query: "978-12-345-678-9" → digits: "9781234567890"
Book content: "ISBN: 9781234567890" → digits: "9781234567890"
Match found → Return {"filtered_indices": [0], "note": "ISBN match"}

OTHER QUERIES:
- Author: match "Author:" section
- Title: match book title
- Topic: match "Topics:" section

MATCHING CRITERIA:
- ISBN queries: if ISBN found in Content preview → INCLUDE automatically
- Author queries: check "Author: " section in Content preview
- Title queries: check book title at start of Content preview
- Genre queries: check "Genre: " section in Content preview
- Topic queries: check "Topics: " section in Content preview
- For multi-criteria: ALL specified elements must match

PRINCIPLE: Balance accuracy with helpfulness. Include books that are reasonably related to the query theme, even if not exact matches. Consider semantic similarity scores and related concepts. Higher similarity scores (>0.3) suggest stronger relevance."""

    # Add user preferences section if available
    if user_preferences:
        preferences_section = _format_preferences_for_prompt(user_preferences)
        if preferences_section:
            base_prompt += f"\n\nUSER PREFERENCES (consider for ranking and filtering):\n{preferences_section}"

    base_prompt += """

RESPONSE FORMAT:
{
  "filtered_indices": [indices after strict analysis],
  "note": "analytical justification of decision"
}"""
    return base_prompt

def _create_filter_user_prompt(query: str, results_summary: str, user_preferences: Optional[Dict[str, Any]] = None) -> str:
    """Creates user prompt with query and results"""
    prompt = f"""Query: "{query}"

Results:
{results_summary}

Select suitable books."""

    # Add context about user preferences if available
    if user_preferences:
        context = _extract_preference_context(user_preferences)
        if context:
            prompt += f"\n\nContext: {context}"

    return prompt

def _format_preferences_for_prompt(user_preferences: Dict[str, Any]) -> str:
    """Format user preferences for inclusion in system prompt"""
    try:
        preferences_lines = []

        likes = user_preferences.get("likes", {})
        dislikes = user_preferences.get("dislikes", {})
        context = user_preferences.get("context", {})

        # Add liked genres/authors/themes
        if likes.get("genres"):
            preferences_lines.append(f"- Prefers genres: {', '.join(likes['genres'])}")
        if likes.get("authors"):
            preferences_lines.append(f"- Prefers authors: {', '.join(likes['authors'])}")
        if likes.get("themes"):
            preferences_lines.append(f"- Interested in themes: {', '.join(likes['themes'])}")

        # Add dislikes
        if dislikes.get("genres"):
            preferences_lines.append(f"- Dislikes genres: {', '.join(dislikes['genres'])}")
        if dislikes.get("authors"):
            preferences_lines.append(f"- Dislikes authors: {', '.join(dislikes['authors'])}")

        # Add context
        if context.get("mood"):
            preferences_lines.append(f"- Current mood: {context['mood']}")
        if context.get("goal"):
            preferences_lines.append(f"- Reading goal: {context['goal']}")

        return "\n".join(preferences_lines) if preferences_lines else ""

    except Exception as e:
        logger.warning(f"⚠️ Error formatting preferences: {e}")
        return ""

def _extract_preference_context(user_preferences: Dict[str, Any]) -> str:
    """Extract brief context from user preferences for user prompt"""
    try:
        context_parts = []

        context = user_preferences.get("context", {})
        if context.get("mood") and context["mood"] != "general reading":
            context_parts.append(f"user wants {context['mood']}")
        if context.get("goal") and context["goal"] != "entertainment":
            context_parts.append(f"goal is {context['goal']}")

        return ", ".join(context_parts) if context_parts else ""

    except Exception as e:
        logger.warning(f"⚠️ Error extracting preference context: {e}")
        return ""

def _prepare_results_for_llm(search_results: List[Dict[str, Any]]) -> str:
    """Prepares search results for passing to LLM"""
    summary_lines = []

    for i, result in enumerate(search_results):
        # Include similarity score and collection info
        content = result.get('content', '')
        score = result.get('score', 0.0)
        collection = result.get('collection', 'unknown')

        summary_line = f"[{i}] Similarity: {score:.3f} | Collection: {collection}\n{content}"
        summary_lines.append(summary_line)

    return "\n\n".join(summary_lines)

def _parse_filter_llm_response(llm_response: str, original_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parses LLM response and returns filtered results"""

    logger.info(f"🔍 Parsing LLM response: {llm_response[:200]}...")

    try:
        # Clean response from possible markdown blocks
        clean_response = llm_response.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:]
        if clean_response.endswith('```'):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()

        # Parse JSON
        parsed = json.loads(clean_response)

        filtered_indices = parsed.get('filtered_indices', [])
        note = parsed.get('note', 'Filtering completed')

        # Filter results by indices
        filtered_results = []
        for idx in filtered_indices:
            if 0 <= idx < len(original_results):
                filtered_results.append(original_results[idx])
            else:
                logger.warning(f"⚠️ Invalid index in filtered_indices: {idx}")

        return {
            "filtered_results": filtered_results,
            "intent": "filtered",
            "analysis": note,
            "negative_filters": "",
            "uncertainty_note": "",
            "total_found": len(original_results),
            "total_filtered": len(filtered_results)
        }

    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing JSON response from LLM: {e}")
        logger.error(f"LLM response: {llm_response}")

        # Fallback: return all results
        return {
            "filtered_results": original_results,
            "intent": "parse_error",
            "analysis": f"Failed to parse LLM response: {e}",
            "negative_filters": "",
            "uncertainty_note": "",
            "total_found": len(original_results),
            "total_filtered": len(original_results)
        }

    except Exception as e:
        logger.error(f"❌ Unexpected error parsing LLM response: {e}")

        return {
            "filtered_results": original_results,
            "intent": "error",
            "analysis": f"Processing error: {e}",
            "negative_filters": "",
            "uncertainty_note": "",
            "total_found": len(original_results),
            "total_filtered": len(original_results)
        }

@traceable(name="filter_step")
async def filter_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 2: LLM filtering and analysis"""
    logger.info(f"🧠 [Step 2] LLM filtering {len(state.search_results)} results")

    # Add trace metadata
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_tags(["filter", "llm_analysis"])
        current_run.add_inputs({
            "query": state.message,
            "num_search_results": len(state.search_results),
            "user_preferences": state.user_preferences is not None
        })
        current_run.add_metadata({"step": "filter", "input_results": len(state.search_results)})

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
        
        # Execute LLM filtering directly (preserve trace context)
        llm_filter_start = time.time()
        filter_result = await _execute_llm_filtering(
            state.message,
            state.search_results,
            user_preferences=state.user_preferences
        )
        llm_filter_time = time.time() - llm_filter_start
        logger.info(f"⚡ LLM filter in {llm_filter_time:.3f}s")
        
        # Update state
        state.filtered_results = filter_result['filtered_results']
        state.intent = filter_result['intent']
        state.analysis = filter_result['analysis']
        state.negative_filters = filter_result.get('negative_filters', '')
        state.uncertainty_note = filter_result.get('uncertainty_note', '')
        state.performance_metrics['filter_time'] = time.time() - start_time

        # Update trace with results
        if current_run:
            current_run.add_outputs({
                "intent": state.intent,
                "num_filtered_results": len(state.filtered_results),
                "filter_time": state.performance_metrics['filter_time']
            })
            current_run.add_metadata({
                "filtered_results": len(state.filtered_results),
                "intent_detected": state.intent,
                "execution_time_s": state.performance_metrics['filter_time'],
                "llm_filter_time_s": llm_filter_time
            })

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

@traceable(name="format_step")
async def format_step(state: SimpleSearchState) -> SimpleSearchState:
    """Step 3: Format final response"""
    logger.info(f"💬 [Step 3] Formatting response for {len(state.filtered_results)} results")

    # Add trace metadata
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_tags(["format", "response_generation"])
        current_run.add_inputs({
            "query": state.message,
            "num_filtered_results": len(state.filtered_results),
            "intent": state.intent,
            "analysis": state.analysis
        })
        current_run.add_metadata({"step": "format", "input_results": len(state.filtered_results)})

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
        
        # Generate response using LLM directly (preserve trace context)
        llm_format_start = time.time()

        # Minimal information for quick formatting with content
        results_info = []
        for i, result in enumerate(state.filtered_results, 1):
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

        user_prompt = f"""User is looking for: "{state.message}"

Found books:
{results_text}

Create an inspiring response that motivates reading!"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await llm.ainvoke(messages)
            formatted_response = response.content

            logger.info(f"📝 LLM generated response: {response.content[:200]}...")

        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            logger.info("🔄 Using fallback formatting")
            formatted_response = _format_simple_response(state.filtered_results)

        llm_format_time = time.time() - llm_format_start
        logger.info(f"⚡ LLM format in {llm_format_time:.3f}s")
        
        state.final_response = formatted_response
        state.performance_metrics['format_time'] = time.time() - start_time

        # Update trace with results
        if current_run:
            current_run.add_outputs({
                "formatted_response": state.final_response,
                "format_time": state.performance_metrics['format_time']
            })
            current_run.add_metadata({
                "response_length": len(state.final_response) if state.final_response else 0,
                "execution_time_s": state.performance_metrics['format_time'],
                "llm_format_time_s": llm_format_time
            })

        logger.info(f"✅ [Step 3] Response formatted in {state.performance_metrics['format_time']:.2f}s")

        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 3] Formatting error: {e}")
        # Fallback to simple formatting
        state.final_response = _format_simple_response(state.filtered_results)
        state.performance_metrics['format_time'] = time.time() - start_time
        return state


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

@traceable(name="process_simple_search")
async def process_simple_search(session_id: str, message: str, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main function for processing simple search

    Args:
        session_id: session identifier
        message: user message
        user_preferences: optional user preferences for filtering/scoring

    Returns:
        Dict with search results
    """
    logger.info(f"🚀 [Simple Search] Starting processing: session_id={session_id}, query='{message}'")

    # Add trace metadata
    current_run = get_current_run_tree()
    if current_run:
        current_run.add_tags(["simple_search", "orchestrator"])
        current_run.add_inputs({
            "session_id": session_id,
            "message": message,
            "has_user_preferences": user_preferences is not None
        })
        current_run.add_metadata({
            "query_length": len(message),
            "session_id": session_id,
            "preferences_provided": user_preferences is not None
        })

    total_start_time = time.time()
    
    try:
        # Create initial state
        initial_state = SimpleSearchState(
            session_id=session_id,
            message=message,
            user_preferences=user_preferences
        )
        
        # Execute graph asynchronously
        final_state = await simple_search_graph.ainvoke(initial_state)
        
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
            # If final_state is a dict
            performance_metrics = final_state.get('performance_metrics', {})
            performance_metrics['total_time'] = total_time
            search_results = final_state.get('search_results', [])
            filtered_results = final_state.get('filtered_results', [])
            final_response = final_state.get('final_response', 'Processing error')
            intent = final_state.get('intent', 'error')
            analysis = final_state.get('analysis', 'Analysis unavailable')
            uncertainty_note = final_state.get('uncertainty_note', '')
        
        # Form the result
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
        
        # Update trace with final results
        if current_run:
            current_run.add_outputs({
                "response": final_response,
                "num_results": len(filtered_results),
                "intent": intent,
                "total_time": total_time
            })
            current_run.add_metadata({
                "total_found": len(search_results),
                "after_filter": len(filtered_results),
                "intent_detected": intent,
                "total_execution_time_s": total_time
            })

        logger.info(f"🎯 [Simple Search] Completed in {total_time:.2f}s: "
                   f"found {len(search_results)}, filtered {len(filtered_results)}")

        return result
        
    except Exception as e:
        logger.error(f"❌ [Simple Search] Critical error: {e}")
        
        return {
            "response": f"Sorry, an error occurred while processing the request: {e}",
            "results": [],
            "intent": "error",
            "analysis": f"Critical error: {e}",
            "performance_metrics": {"total_time": time.time() - total_start_time},
            "search_stats": {
                "total_found": 0,
                "after_filter": 0,
                "total_time": time.time() - total_start_time
            }
        }