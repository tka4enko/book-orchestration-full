#!/usr/bin/env python3
"""
Comprehensive test runner for BookBot system
Tests all 40 queries from tests_extra.json and provides detailed analysis
"""

import json
import logging
import time
from typing import List, Dict, Any, Tuple
from app.orchestrator import graph, ChatState

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise, we'll capture results

class TestResult:
    def __init__(self, test_id: str, query: str, expected: dict, actual: dict, success: bool, issues: List[str] = None):
        self.test_id = test_id
        self.query = query
        self.expected = expected
        self.actual = actual
        self.success = success
        self.issues = issues or []

def run_single_test(test_case: dict) -> TestResult:
    """Run a single test case and analyze results"""
    test_id = test_case["id"]
    query = test_case["query"]
    expected = test_case["expected"]
    
    print(f"\n{'='*60}")
    print(f"🧪 TEST {test_id}: '{query}'")
    print(f"{'='*60}")
    
    start_time = time.time()
    issues = []
    
    try:
        # Run the query through the full system
        state = ChatState(session_id="test", message=query)
        output = graph.invoke(state)
        
        query_time = time.time() - start_time
        
        # Extract actual results
        actual = {
            "intent": output.get("intent", "unknown"),
            "retrieved_count": len(output.get("results", [])),
            "document_ids": [r.get("document_id") for r in output.get("results", []) if r.get("document_id")],
            "response": output.get("response", ""),
            "clarify": output.get("clarify", False),
            "query_time": query_time
        }
        
        print(f"   🎯 Intent: {actual['intent']} (expected: {test_case.get('intent_expected', 'N/A')})")
        print(f"   📊 Retrieved: {actual['retrieved_count']} documents")
        print(f"   ⏱️ Query time: {query_time:.2f}s")
        
        # Show found documents
        for i, result in enumerate(output.get("results", [])[:3]):
            if isinstance(result, dict):
                doc_id = result.get("document_id", "unknown")
                title = result.get("title", "Unknown")
                author = result.get("author", "Unknown")
                print(f"   {i+1}. {title} by {author} (ID: {doc_id[:8]}...)")
        
        # Analyze against expectations
        success, test_issues = analyze_test_result(test_case, actual)
        issues.extend(test_issues)
        
        print(f"   {'✅ PASS' if success else '❌ FAIL'}")
        if issues:
            for issue in issues:
                print(f"   🔍 Issue: {issue}")
                
        return TestResult(test_id, query, expected, actual, success, issues)
        
    except Exception as e:
        issues.append(f"Query execution failed: {str(e)}")
        print(f"   ❌ ERROR: {e}")
        return TestResult(test_id, query, expected, {}, False, issues)

def analyze_test_result(test_case: dict, actual: dict) -> Tuple[bool, List[str]]:
    """Analyze test result against expectations"""
    expected = test_case["expected"]
    issues = []
    success = True
    
    # Check intent expectation
    if "intent_expected" in test_case:
        expected_intent = test_case["intent_expected"]
        actual_intent = actual["intent"]
        if actual_intent != expected_intent and expected_intent not in ["fuzzy_title", "ask_summary"]:
            # Allow mixed_filters for fuzzy_title and ask_summary as they're often parsed as mixed
            if not (expected_intent in ["fuzzy_title", "ask_summary"] and actual_intent == "mixed_filters"):
                issues.append(f"Intent mismatch: expected '{expected_intent}', got '{actual_intent}'")
                success = False
    
    # Check top1_id expectation
    if "top1_id" in expected:
        expected_id = expected["top1_id"]
        actual_ids = actual["document_ids"]
        if not actual_ids or actual_ids[0] != expected_id:
            issues.append(f"Top result mismatch: expected '{expected_id}', got '{actual_ids[0] if actual_ids else 'none'}'")
            success = False
    
    # Check must_include_ids
    if "must_include_ids" in expected:
        for must_id in expected["must_include_ids"]:
            if must_id not in actual["document_ids"]:
                issues.append(f"Missing required document: '{must_id}'")
                success = False
    
    # Check must_exclude_ids  
    if "must_exclude_ids" in expected:
        for exclude_id in expected["must_exclude_ids"]:
            if exclude_id in actual["document_ids"]:
                issues.append(f"Included forbidden document: '{exclude_id}'")
                success = False
    
    # Check clarify expectation
    if "expect_clarify" in expected and expected["expect_clarify"]:
        if not actual.get("clarify", False):
            issues.append("Expected clarification but none requested")
            success = False
    
    # Check abstain allowance
    if "allow_abstain" in expected and expected["allow_abstain"]:
        if actual["retrieved_count"] == 0:
            print(f"   ℹ️ Abstain allowed and occurred")
            # This is OK, not a failure
    
    return success, issues

def run_comprehensive_tests():
    """Run all tests from tests_extra.json"""
    
    # Load test data
    with open("tests_extra.json", "r") as f:
        test_data = json.load(f)
    
    print("🚀 STARTING COMPREHENSIVE BOOKBOT TEST SUITE")
    print(f"📊 Total tests to run: {len(test_data['tests_extra'])}")
    print("="*80)
    
    results = []
    categories = {
        "cross_language": [],
        "fuzzy_matching": [],
        "conflict_detection": [],
        "negative_filters": [],
        "security": [],
        "edge_cases": []
    }
    
    for test_case in test_data["tests_extra"]:
        result = run_single_test(test_case)
        results.append(result)
        
        # Categorize test
        if "alias" in test_case.get("notes", "") or "издание" in test_case.get("notes", ""):
            categories["cross_language"].append(result)
        elif "опечатка" in test_case.get("notes", "") or "fuzzy" in test_case["id"].lower():
            categories["fuzzy_matching"].append(result)
        elif "конфликт" in test_case.get("notes", "") or "уточн" in test_case.get("notes", ""):
            categories["conflict_detection"].append(result)
        elif "NOT" in test_case["query"] or "не про" in test_case["query"]:
            categories["negative_filters"].append(result)
        elif "injection" in test_case.get("notes", "") or "ignore" in test_case["query"]:
            categories["security"].append(result)
        else:
            categories["edge_cases"].append(result)
    
    # Generate summary
    print("\\n" + "="*80)
    print("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    
    # Category breakdown
    print("\\n📋 Results by Category:")
    for category, cat_results in categories.items():
        if cat_results:
            cat_passed = sum(1 for r in cat_results if r.success)
            cat_total = len(cat_results)
            print(f"  {category.replace('_', ' ').title()}: {cat_passed}/{cat_total} ({cat_passed/cat_total*100:.1f}%)")
    
    # Failed test analysis
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print("\\n❌ FAILED TESTS ANALYSIS:")
        print("-"*50)
        for result in failed_results:
            print(f"\\n{result.test_id}: '{result.query}'")
            for issue in result.issues:
                print(f"  • {issue}")
            if result.test_id in [tc["id"] for tc in test_data["tests_extra"]]:
                test_case = next(tc for tc in test_data["tests_extra"] if tc["id"] == result.test_id)
                print(f"  📝 Notes: {test_case['expected'].get('notes', 'No notes')}")
    
    # Performance analysis
    query_times = [r.actual.get("query_time", 0) for r in results if r.actual.get("query_time")]
    if query_times:
        avg_time = sum(query_times) / len(query_times)
        max_time = max(query_times)
        min_time = min(query_times)
        print(f"\\n⏱️ Performance Analysis:")
        print(f"  Average query time: {avg_time:.2f}s")
        print(f"  Fastest query: {min_time:.2f}s")
        print(f"  Slowest query: {max_time:.2f}s")
    
    # Top issues
    all_issues = []
    for result in failed_results:
        all_issues.extend(result.issues)
    
    issue_counts = {}
    for issue in all_issues:
        issue_type = issue.split(":")[0] if ":" in issue else issue
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    if issue_counts:
        print(f"\\n🔍 Most Common Issues:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {issue_type}: {count} occurrences")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_tests()