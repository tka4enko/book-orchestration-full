#!/usr/bin/env python3
"""
Test negative_filter intent cases
"""

import logging
import time
from app.orchestrator import graph, ChatState

# Configure minimal logging
logging.basicConfig(level=logging.INFO)

def test_negative_filter(test_id, query, expected_intent="negative_filter"):
    print(f"\n{'='*60}")
    print(f"🧪 TEST {test_id}: '{query}'")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Create state and run full pipeline
        state = ChatState(session_id="test", message=query)
        print(f"   📤 Input: '{query}'")
        
        # Run complete graph
        output = graph.invoke(state)
        
        query_time = time.time() - start_time
        
        # Extract results
        intent = output.get("intent", "unknown")
        retrieved_count = len(output.get("results", []))
        exclude_filters = output.get("exclude_filters", {})
        filters = output.get("filters", {})
        
        print(f"   🎯 Intent: {intent} (expected: {expected_intent})")
        print(f"   🔍 Filters: {filters}")
        print(f"   🚫 Exclude filters: {exclude_filters}")
        print(f"   📊 Retrieved: {retrieved_count} documents")
        print(f"   ⏱️ Query time: {query_time:.2f}s")
        
        # Show found documents
        for i, result in enumerate(output.get("results", [])[:3]):
            if isinstance(result, dict):
                doc_id = result.get("document_id", "unknown")
                title = result.get("title", "Unknown")
                author = result.get("author", "Unknown")
                genre = result.get("primary_genre", "Unknown")
                print(f"   {i+1}. {title} by {author} (Genre: {genre}, ID: {doc_id[:8] if doc_id else 'N/A'}...)")
        
        # Check if intent detection works
        intent_correct = intent == expected_intent
        print(f"   Intent detection: {'✅' if intent_correct else '❌'}")
        
        return {
            "intent": intent,
            "intent_correct": intent_correct,
            "filters": filters,
            "exclude_filters": exclude_filters,
            "results": output.get("results", []),
            "query_time": query_time
        }
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    test_cases = [
        ("A010", "A book about \"Big Brother\" and censorship but not science fiction"),
        ("A011", "self-help NOT spirituality about feedback"),
        ("A038", "Дай книгу по самодисциплине, но не про духовность")
    ]
    
    results = {}
    
    print("🚀 TESTING NEGATIVE_FILTER INTENT CASES")
    print("=" * 80)
    
    for test_id, query in test_cases:
        result = test_negative_filter(test_id, query)
        results[test_id] = result
    
    # Summary
    print(f"\n🏁 SUMMARY")
    print("=" * 40)
    
    correct_intents = sum(1 for r in results.values() if r.get("intent_correct", False))
    total_tests = len(results)
    
    print(f"Intent detection: {correct_intents}/{total_tests} correct")
    
    for test_id, result in results.items():
        intent = result.get("intent", "error")
        status = "✅" if result.get("intent_correct", False) else "❌"
        print(f"  {test_id}: {status} {intent}")
    
    return results

if __name__ == "__main__":
    results = main()