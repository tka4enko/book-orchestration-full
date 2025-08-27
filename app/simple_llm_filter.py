from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class SimpleLLMFilter:
    """LLM filter with system prompt for analyzing and filtering search results"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,  # Deterministic result
            api_key=OPENAI_API_KEY
        )
        logger.info("🧠 SimpleLLMFilter initialized")
    
    async def filter_and_analyze(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes user query and filters search results
        
        Args:
            query: original user query
            search_results: vector search results
            
        Returns:
            Dict with filtered results and analysis metadata
        """
        logger.info(f"🧠 LLM analysis of query: '{query}' for {len(search_results)} results")
        
        # DEBUG: Show what we received from retriever
        if search_results:
            logger.info(f"🔍 STAGE 2 - LLM Filter input:")
            logger.info(f"   First result keys: {list(search_results[0].keys())}")
            if 'content' in search_results[0]:
                content_preview = search_results[0]['content'][:300]
                logger.info(f"   Content preview (300 chars): {content_preview}...")
                logger.info(f"   Full content length: {len(search_results[0]['content'])} chars")
        
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
            results_summary = self._prepare_results_for_llm(search_results)
            
            # DEBUG: Show what we send to LLM
            logger.info(f"🔍 STAGE 2B - Data prepared for LLM:")
            logger.info(f"   Results summary length: {len(results_summary)} chars")
            logger.info(f"   Results summary preview (500 chars): {results_summary[:500]}...")
            
            # Create prompt
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, results_summary)
            
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # DEBUG: Show what we send to LLM
            logger.info(f"🔍 STAGE 2C - LLM prompts:")
            logger.info(f"   System prompt length: {len(system_prompt)} chars")
            logger.info(f"   FULL SYSTEM PROMPT:")
            logger.info(f"   {system_prompt}")
            logger.info(f"   FULL USER PROMPT:")
            logger.info(f"   {user_prompt}")
            
            response = await self.llm.ainvoke(messages)
            
            # DEBUG: Show LLM raw response
            logger.info(f"🔍 STAGE 3 - LLM raw response:")
            logger.info(f"   FULL LLM RESPONSE:")
            logger.info(f"   {response.content}")
            
            # Parse LLM response
            analysis_result = self._parse_llm_response(response.content, search_results)
            
            # DEBUG: Show final filter output
            logger.info(f"🔍 STAGE 4 - LLM Filter final output:")
            logger.info(f"   Intent: {analysis_result['intent']}")
            logger.info(f"   Filtered: {analysis_result['total_filtered']}/{analysis_result['total_found']}")
            logger.info(f"   Analysis: {analysis_result.get('analysis', 'N/A')}")
            
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
    
    def _create_system_prompt(self) -> str:
        """Creates analytical system prompt for filtering"""
        return """You are a library search filter. Your job is simple: find which books match the user's query.

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

PRINCIPLE: Balance accuracy with helpfulness. Include books that are reasonably related to the query theme, even if not exact matches. Consider semantic similarity scores and related concepts. Higher similarity scores (>0.3) suggest stronger relevance.

RESPONSE FORMAT:
{
  "filtered_indices": [indices after strict analysis],
  "note": "analytical justification of decision"
}"""

    def _create_user_prompt(self, query: str, results_summary: str) -> str:
        """Creates user prompt with query and results"""
        return f"""Query: "{query}"

Results:
{results_summary}

Select suitable books."""

    def _prepare_results_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
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

    def _parse_llm_response(self, llm_response: str, original_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses LLM response and returns filtered results"""
        logger.info(f"🔍 Parsing LLM response: {llm_response[:200]}...")
        
        try:
            import json
            
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