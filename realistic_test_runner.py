#!/usr/bin/env python3
"""
Realistic Test Runner for BookBot System
Runs 40 realistic user queries and analyzes quality of final responses
"""

import json
import logging
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
from app.orchestrator import graph, ChatState

# Configure logging to capture detailed information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticTestRunner:
    def __init__(self):
        self.results = []
        self.stats = {
            "total_tests": 0,
            "successful": 0,
            "failed": 0,
            "clarifications": 0,
            "correct_books": 0,
            "wrong_books": 0,
            "intent_correct": 0,
            "intent_wrong": 0,
            "category_stats": defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})
        }
        
    def load_tests(self, filename: str = "realistic_tests.json") -> List[Dict]:
        """Load test cases from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["realistic_tests"]
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case and analyze the result"""
        test_id = test_case["id"]
        query = test_case["query"]
        category = test_case["category"]
        expected_intent = test_case["expected_intent"]
        expected_outcome = test_case["expected_outcome"]
        expected_doc_id = test_case.get("expected_doc_id")
        
        print(f"\n{'='*80}")
        print(f"🧪 TEST {test_id} [{category.upper()}]: '{query}'")
        print(f"   Expected: {expected_intent} → {expected_outcome}")
        if expected_doc_id:
            print(f"   Expected book ID: {expected_doc_id[:8]}...")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Run query through the system
            state = ChatState(session_id=f"test_{test_id}", message=query)
            result = graph.invoke(state)
            
            query_time = time.time() - start_time
            
            # Extract key information
            actual_intent = result.get("intent", "unknown")
            results_list = result.get("results", [])
            
            # Find actual book ID from results
            actual_doc_id = None
            if results_list:
                # Look for structured results with document_id
                for res in results_list:
                    if isinstance(res, dict) and res.get("document_id"):
                        actual_doc_id = res["document_id"]
                        break
            
            # Analyze the response
            analysis = self.analyze_response(test_case, result, actual_intent, actual_doc_id, query_time)
            
            # Print immediate feedback
            print(f"   🎯 Intent: {actual_intent} {'✅' if actual_intent == expected_intent else '❌'}")
            print(f"   📊 Results: {len(results_list)} documents")
            print(f"   ⏱️ Query time: {query_time:.2f}s")
            
            if actual_doc_id:
                if actual_doc_id == expected_doc_id:
                    print(f"   📚 Book: CORRECT ✅ {actual_doc_id[:8]}...")
                else:
                    print(f"   📚 Book: WRONG ❌ Got {actual_doc_id[:8]}... Expected {expected_doc_id[:8] if expected_doc_id else 'None'}...")
            else:
                print(f"   📚 Book: NOT FOUND ❓")
            
            print(f"   📋 Overall: {'SUCCESS' if analysis['overall_success'] else 'FAILED'} {'✅' if analysis['overall_success'] else '❌'}")
            
            # Show final response preview
            final_response = self.extract_final_response(results_list)
            if final_response:
                print(f"   💬 Response: \"{final_response[:100]}...\"")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Test {test_id} failed with exception: {e}")
            analysis = {
                "test_id": test_id,
                "query": query,
                "category": category,
                "expected_intent": expected_intent,
                "expected_outcome": expected_outcome,
                "expected_doc_id": expected_doc_id,
                "actual_intent": "error",
                "actual_doc_id": None,
                "query_time": 0,
                "overall_success": False,
                "error": str(e),
                "issues": [f"System error: {e}"]
            }
            print(f"   💥 ERROR: {e}")
            return analysis
    
    def analyze_response(self, test_case: Dict, result: Dict, actual_intent: str, actual_doc_id: str, query_time: float) -> Dict:
        """Analyze the quality of the system response"""
        
        test_id = test_case["id"]
        expected_intent = test_case["expected_intent"]
        expected_outcome = test_case["expected_outcome"]
        expected_doc_id = test_case.get("expected_doc_id")
        
        analysis = {
            "test_id": test_id,
            "query": test_case["query"],
            "category": test_case["category"],
            "expected_intent": expected_intent,
            "expected_outcome": expected_outcome,
            "expected_doc_id": expected_doc_id,
            "actual_intent": actual_intent,
            "actual_doc_id": actual_doc_id,
            "query_time": query_time,
            "issues": []
        }
        
        # Check intent classification
        intent_correct = actual_intent == expected_intent
        analysis["intent_correct"] = intent_correct
        if not intent_correct:
            analysis["issues"].append(f"Wrong intent: expected {expected_intent}, got {actual_intent}")
        
        # Check if correct book was found
        book_correct = False
        if expected_outcome == "success" and expected_doc_id:
            book_correct = actual_doc_id == expected_doc_id
            analysis["book_correct"] = book_correct
            if not book_correct and actual_doc_id:
                analysis["issues"].append(f"Wrong book found: expected {expected_doc_id[:8]}..., got {actual_doc_id[:8]}...")
            elif not actual_doc_id:
                analysis["issues"].append("No book found when one was expected")
        elif expected_outcome == "failure":
            book_correct = actual_doc_id is None
            analysis["book_correct"] = book_correct
            if actual_doc_id:
                analysis["issues"].append("Found book when none should be found")
        elif expected_outcome == "clarify":
            # For clarify, we expect the system to find the right book but ask for clarification
            book_correct = actual_doc_id == expected_doc_id if expected_doc_id else True
            analysis["book_correct"] = book_correct
            # Should also check if clarification was requested
            clarify_requested = result.get("need_clarify", False) or result.get("clarify_question")
            analysis["clarify_requested"] = clarify_requested
            if not clarify_requested:
                analysis["issues"].append("Should have requested clarification but didn't")
        
        # Analyze final response quality
        results_list = result.get("results", [])
        final_response = self.extract_final_response(results_list)
        analysis["final_response"] = final_response
        analysis["response_quality"] = self.assess_response_quality(final_response, test_case)
        
        # Overall success determination
        if expected_outcome == "success":
            analysis["overall_success"] = intent_correct and book_correct and len(analysis["issues"]) == 0
        elif expected_outcome == "failure":
            analysis["overall_success"] = book_correct  # Should not find any book
        elif expected_outcome == "clarify":
            analysis["overall_success"] = book_correct and analysis.get("clarify_requested", False)
        else:
            analysis["overall_success"] = False
        
        return analysis
    
    def extract_final_response(self, results_list: List) -> str:
        """Extract the final formatted response from results"""
        if not results_list:
            return ""
        
        # Look for formatted message (usually last in results)
        for result in reversed(results_list):
            if isinstance(result, dict) and result.get("message"):
                return result["message"]
        
        return ""
    
    def assess_response_quality(self, response: str, test_case: Dict) -> Dict:
        """Assess the quality of the final response"""
        if not response:
            return {"score": 0, "issues": ["No response generated"]}
        
        quality = {"score": 100, "issues": []}
        
        # Check for obvious errors
        if "error" in response.lower() or "failed" in response.lower():
            quality["score"] -= 50
            quality["issues"].append("Response contains error messages")
        
        # Check response length (too short might be incomplete)
        if len(response) < 50:
            quality["score"] -= 20
            quality["issues"].append("Response is very short")
        
        # Check if response is in Russian (as expected by system)
        if not any(ord(char) > 127 for char in response):  # Simple check for Cyrillic
            quality["score"] -= 10
            quality["issues"].append("Response not in Russian as expected")
        
        return quality
    
    def run_all_tests(self) -> Dict:
        """Run all tests and generate comprehensive report"""
        print("🚀 STARTING REALISTIC BOOKBOT TEST SUITE")
        print(f"📊 Loading tests from realistic_tests.json...")
        
        tests = self.load_tests()
        self.stats["total_tests"] = len(tests)
        
        print(f"📊 Total tests to run: {len(tests)}")
        print("="*100)
        
        # Run all tests
        for test_case in tests:
            analysis = self.run_single_test(test_case)
            self.results.append(analysis)
            
            # Update statistics
            category = test_case["category"]
            self.stats["category_stats"][category]["total"] += 1
            
            if analysis["overall_success"]:
                self.stats["successful"] += 1
                self.stats["category_stats"][category]["success"] += 1
            else:
                self.stats["failed"] += 1
                self.stats["category_stats"][category]["fail"] += 1
            
            if analysis.get("intent_correct"):
                self.stats["intent_correct"] += 1
            else:
                self.stats["intent_wrong"] += 1
            
            if analysis.get("book_correct"):
                self.stats["correct_books"] += 1
            else:
                self.stats["wrong_books"] += 1
        
        # Generate final report
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        total = self.stats["total_tests"]
        
        print(f"\n{'='*100}")
        print("🏁 REALISTIC TEST SUITE RESULTS")
        print(f"{'='*100}")
        
        # Overall statistics
        success_rate = (self.stats["successful"] / total * 100) if total > 0 else 0
        intent_accuracy = (self.stats["intent_correct"] / total * 100) if total > 0 else 0
        book_accuracy = (self.stats["correct_books"] / total * 100) if total > 0 else 0
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Total tests: {total}")
        print(f"   ✅ Successful: {self.stats['successful']} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   🎯 Intent accuracy: {self.stats['intent_correct']}/{total} ({intent_accuracy:.1f}%)")
        print(f"   📚 Book accuracy: {self.stats['correct_books']}/{total} ({book_accuracy:.1f}%)")
        
        # Category breakdown
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, stats in self.stats["category_stats"].items():
            cat_total = stats["total"] 
            cat_success = stats["success"]
            cat_rate = (cat_success / cat_total * 100) if cat_total > 0 else 0
            print(f"   {category:12}: {cat_success:2}/{cat_total:2} ({cat_rate:5.1f}%) {'✅' if cat_rate >= 80 else '❌' if cat_rate < 50 else '⚠️'}")
        
        # Failed tests analysis
        failed_tests = [r for r in self.results if not r["overall_success"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ANALYSIS ({len(failed_tests)} tests):")
            print("-" * 80)
            
            for result in failed_tests:
                print(f"\n{result['test_id']} [{result['category']}]: '{result['query']}'")
                print(f"   Expected: {result['expected_intent']} → {result['expected_outcome']}")
                print(f"   Got: {result['actual_intent']} → {'success' if result.get('overall_success') else 'failed'}")
                for issue in result.get("issues", []):
                    print(f"   • {issue}")
        
        # Top issues analysis
        all_issues = []
        for result in failed_tests:
            all_issues.extend(result.get("issues", []))
        
        issue_counts = defaultdict(int)
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue.split(" ")[0:3] 
            issue_key = " ".join(issue_type) if isinstance(issue_type, list) else issue_type
            issue_counts[issue_key] += 1
        
        if issue_counts:
            print(f"\n🔍 MOST COMMON ISSUES:")
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {issue_type}: {count} times")
        
        # Performance analysis
        query_times = [r.get("query_time", 0) for r in self.results if r.get("query_time")]
        if query_times:
            avg_time = sum(query_times) / len(query_times)
            max_time = max(query_times)
            min_time = min(query_times)
            print(f"\n⏱️ PERFORMANCE ANALYSIS:")
            print(f"   Average query time: {avg_time:.2f}s")
            print(f"   Fastest query: {min_time:.2f}s")
            print(f"   Slowest query: {max_time:.2f}s")
        
        # Save detailed results to file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "successful": self.stats["successful"],
                "failed": self.stats["failed"],
                "success_rate": success_rate,
                "intent_accuracy": intent_accuracy,
                "book_accuracy": book_accuracy
            },
            "category_stats": dict(self.stats["category_stats"]),
            "detailed_results": self.results,
            "failed_tests": failed_tests,
            "performance": {
                "avg_query_time": avg_time if query_times else 0,
                "max_query_time": max_time if query_times else 0,
                "min_query_time": min_time if query_times else 0
            }
        }
        
        with open("realistic_test_report.json", "w", encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: realistic_test_report.json")
        print(f"{'='*100}")
        
        return report_data

if __name__ == "__main__":
    runner = RealisticTestRunner()
    report = runner.run_all_tests()
    
    # Print final summary
    success_rate = report["summary"]["success_rate"]
    print(f"\n🎯 FINAL RESULT: {success_rate:.1f}% SUCCESS RATE")
    
    if success_rate >= 85:
        print("🏆 EXCELLENT - System performing very well!")
    elif success_rate >= 70:
        print("✅ GOOD - System working well with room for improvement")
    elif success_rate >= 50:
        print("⚠️ AVERAGE - System needs significant improvements")
    else:
        print("❌ POOR - System needs major fixes")