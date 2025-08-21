#!/usr/bin/env python3
"""
Full pipeline testing script for BookBot
Tests from query parsing to final response formatting
"""

import json
import logging
from typing import List, Dict, Any

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_query_pipeline(query: str, test_id: str = "TEST") -> Dict[str, Any]:
    """Test complete query pipeline: parsing → retrieval → response"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST {test_id}: '{query}'")
    print(f"{'='*60}")
    
    results = {}
    
    try:
        # Step 1: Use LangGraph for complete pipeline
        print("\n🚀 STEP 1: Full LangGraph Pipeline")
        from app.orchestrator import graph, ChatState
        
        # Create state and run full pipeline
        state = ChatState(session_id="test", message=query)
        print(f"   📤 Input: '{query}'")
        
        # Run complete graph
        output = graph.invoke(state)
        
        # Extract results (output is dict from LangGraph)
        print(f"   🎯 Intent: {output.get('intent', 'unknown')}")
        print(f"   📊 Retrieved: {len(output.get('results', []))} documents")
        
        results['intent'] = output.get('intent', 'unknown')
        results['retrieved_count'] = len(output.get('results', []))
        results['document_ids'] = []
        results['clarify'] = output.get('clarify', False)
        results['response'] = output.get('response', '')
        
        # Show document details
        for i, result in enumerate(output.get('results', [])[:3]):  # Show top 3
            if isinstance(result, dict):
                doc_id = result.get('document_id', 'unknown')
                title = result.get('title', 'Unknown')
                author = result.get('author', 'Unknown')
                print(f"   {i+1}. {title} by {author} (ID: {doc_id[:8]}...)")
                results['document_ids'].append(doc_id)
            else:
                print(f"   {i+1}. Result without expected format: {type(result)}")
        
        # Step 2: Show response 
        print(f"\n📝 STEP 2: Generated Response")
        response_text = results['response']
        print(f"   📄 Response length: {len(response_text)} characters")
        print(f"   📄 Response preview: {response_text[:200]}...")
        results['response_length'] = len(response_text)
        results['response_preview'] = response_text[:200]
        
        # Step 3: Test specific parsing components
        print(f"\n🔍 STEP 3: Component Testing")
        from app.mixed_filters_parser import MixedFiltersParser
        
        parser = MixedFiltersParser()
        if parser.should_use_mixed_filters(query):
            print(f"   🔍 Mixed filters applicable - testing JSON determinism...")
            parse_result = parser.parse_query(query, use_llm=True)
            print(f"   📊 Parsed filters: {parse_result.filters.model_dump()}")
            if parse_result.conflict.has_conflict:
                print(f"   ⚠️  Conflict detected: {parse_result.conflict.reason}")
            if parse_result.clarify.should_ask:
                print(f"   ❓ Clarification needed: {parse_result.clarify.question}")
            results['filters'] = parse_result.filters.model_dump()
            results['conflict'] = parse_result.conflict.has_conflict
        else:
            print(f"   ℹ️  Simple query - no mixed filters parsing needed")
            results['filters'] = None
            results['conflict'] = False
        
        # Step 4: Final verification
        print(f"\n✅ STEP 4: Final Results")
        print(f"   Intent: {results['intent']}")
        print(f"   Documents: {results['retrieved_count']}")
        print(f"   Response: {results['response_length']} chars") 
        print(f"   Clarify needed: {results.get('clarify', False)}")
        results['success'] = True
        
    except Exception as e:
        print(f"\n❌ PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    return results

def run_test_suite():
    """Run test suite on selected queries"""
    
    # Load test queries
    with open('/Users/markupus/Python_Projects/bookbot_full/test_queries.json', 'r') as f:
        test_data = json.load(f)
    
    # Select key test cases to run
    key_tests = [
        ("A001", "Animal Farm by George Orwell"),  # Should abstain/clarify
        ("A002", "1984 by Aldous Huxley"),        # Conflict detection
        ("A004", "Radical Candor Kim Scott"),     # Cross-language alias
        ("A016", "Georg Orwel 1984"),             # Fuzzy matching
        ("A011", "self-help NOT spirituality about feedback"),  # Negative filters
    ]
    
    print("🚀 STARTING FULL PIPELINE TEST SUITE")
    print("=" * 80)
    
    all_results = {}
    
    for test_id, query in key_tests:
        try:
            result = test_query_pipeline(query, test_id)
            all_results[test_id] = result
        except Exception as e:
            print(f"❌ Test {test_id} failed: {e}")
            all_results[test_id] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n🏁 TEST SUITE SUMMARY")
    print("=" * 40)
    
    successful = sum(1 for r in all_results.values() if r.get('success', False))
    total = len(all_results)
    
    print(f"Tests passed: {successful}/{total}")
    
    for test_id, result in all_results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        intent = result.get('intent', 'unknown')
        doc_count = result.get('retrieved_count', 0)
        print(f"  {test_id}: {status} - Intent: {intent}, Docs: {doc_count}")
    
    return all_results

if __name__ == "__main__":
    results = run_test_suite()