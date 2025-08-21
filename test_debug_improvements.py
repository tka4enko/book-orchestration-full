#!/usr/bin/env python3
"""
Test script for enhanced debug reporter functionality
"""

import os
import sys
import asyncio

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Set debug environment variables
os.environ['DEBUG_REPORTER_ENABLED'] = 'true'
os.environ['DEBUG_LLM_ANALYSIS'] = 'false'  # Disable LLM analysis for faster testing

async def test_search_query():
    """Test a simple search query to see enhanced debug output"""
    from app.orchestrator import graph as orchestrator_graph
    
    print("🧪 Testing enhanced debug reporter with query: 'книга про сон и здоровье'")
    print("="*100)
    
    # Create test input
    input_data = {
        "message": "книга про сон и здоровье",
        "session_id": "test_debug_session",
        "intent": "",
        "filters": {},
        "exclude_filters": {},
        "results": [],
        "need_clarify": False,
        "clarify_question": None,
        "mixed_filters_result": None
    }
    
    try:
        # Run the orchestrator
        result = await orchestrator_graph.ainvoke(input_data)
        
        # Manually finalize debug to see the full report
        from app.debug_reporter import finalize_debug
        finalize_debug(len(result.get('results', [])), 1000.0)  # Mock execution time
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results count: {len(result.get('results', []))}")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_search_query())
    sys.exit(0 if success else 1)