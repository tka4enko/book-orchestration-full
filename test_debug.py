#!/usr/bin/env python3
"""
Test script for debug reporter system
Run with: python test_debug.py
"""

import os

# Set debug environment BEFORE importing modules
os.environ['DEBUG_REPORTER_ENABLED'] = 'true'
os.environ['DEBUG_LLM_ANALYSIS'] = 'false'  # Set to 'true' to enable LLM analysis

from app.orchestrator import ChatState, graph

def test_debug_system():
    """Test the debug reporter system with various queries"""
    
    test_queries = [
        "сон детский",  # Simple topic query
        "Animal Farm by George Orwell",  # Author + title
        "books about leadership",  # Topic search  
        "isbn:978-0451524935",  # ISBN search (if you have data)
        "fantasy books but not romance"  # Negative filter
    ]
    
    print("🚀 Testing Debug Reporter System")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: '{query}'")
        print("-" * 30)
        
        try:
            state = ChatState(session_id=f'test_{i}', message=query)
            result = graph.invoke(state)
            
            # Brief result summary
            results_count = len([r for r in result.get('results', []) if isinstance(r, dict) and 'title' in r])
            print(f"✅ Query processed successfully. Found {results_count} results.")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_debug_system()