#!/usr/bin/env python3
"""
Quick test to verify A033 runtime error fix
"""

import logging
import time
from app.orchestrator import graph, ChatState

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)

def test_a033():
    print("🧪 Testing A033 - radikální otevřenost genre:self-help")
    print("=" * 60)
    
    query = "radikální otevřenost genre:self-help"
    
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
        document_ids = [r.get("document_id") for r in output.get("results", []) if r.get("document_id")]
        response = output.get("response", "")
        clarify = output.get("clarify", False)
        
        print(f"   🎯 Intent: {intent}")
        print(f"   📊 Retrieved: {retrieved_count} documents")
        print(f"   ⏱️ Query time: {query_time:.2f}s")
        
        # Show found documents
        for i, result in enumerate(output.get("results", [])[:3]):
            if isinstance(result, dict):
                doc_id = result.get("document_id", "unknown")
                title = result.get("title", "Unknown")
                author = result.get("author", "Unknown")
                print(f"   {i+1}. {title} by {author} (ID: {doc_id[:8] if doc_id else 'N/A'}...)")
        
        print(f"   ✅ SUCCESS - No runtime error!")
        print(f"   📋 Expected document ID: b3fbff9c-3d55-46e2-a5b6-3b628b6f9fc1")
        print(f"   📋 Got document IDs: {document_ids}")
        
        expected_id = "b3fbff9c-3d55-46e2-a5b6-3b628b6f9fc1"
        if expected_id in document_ids:
            print(f"   🎉 EXPECTED DOCUMENT FOUND!")
            return True
        else:
            print(f"   ⚠️  Expected document not found in top results")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_a033()
    exit(0 if success else 1)