#!/usr/bin/env python3
"""
Test script for debug reporter system with LLM analysis
Run with: python test_debug_llm.py
"""

import os

# Set debug environment BEFORE importing modules
os.environ['DEBUG_REPORTER_ENABLED'] = 'true'
os.environ['DEBUG_LLM_ANALYSIS'] = 'true'  # Enable LLM analysis

from app.orchestrator import ChatState, graph

def test_debug_with_llm():
    """Test the debug reporter system with LLM analysis"""
    
    test_query = "Animal Farm by George Orwell"
    
    print("🚀 Testing Debug Reporter with LLM Analysis")
    print("=" * 50)
    print(f"📝 Query: '{test_query}'")
    print("-" * 30)
    
    try:
        state = ChatState(session_id='llm_test', message=test_query)
        result = graph.invoke(state)
        
        # Brief result summary
        results_count = len([r for r in result.get('results', []) if isinstance(r, dict) and 'title' in r])
        print(f"✅ Query processed successfully. Found {results_count} results.")
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_with_llm()