#!/usr/bin/env python3
"""
Analyze intent classification errors from the test suite
"""

import json
import logging
from collections import defaultdict
from app.orchestrator import graph, ChatState

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)

def analyze_intent_classification():
    """Analyze all intent classification errors to understand patterns"""
    
    # Load test data
    with open("tests_extra.json", "r") as f:
        test_data = json.load(f)
    
    print("🔍 ANALYZING INTENT CLASSIFICATION ERRORS")
    print("=" * 80)
    
    errors = []
    correct = 0
    total = 0
    
    error_patterns = defaultdict(list)
    
    for test_case in test_data["tests_extra"]:
        test_id = test_case["id"]
        query = test_case["query"]
        expected_intent = test_case.get("intent_expected")
        
        if not expected_intent:
            continue
            
        total += 1
        
        try:
            # Test intent detection
            state = ChatState(session_id="test", message=query)
            output = graph.invoke(state)
            actual_intent = output.get("intent", "unknown")
            
            if actual_intent == expected_intent:
                correct += 1
            else:
                error_info = {
                    "test_id": test_id,
                    "query": query,
                    "expected": expected_intent,
                    "actual": actual_intent,
                    "lang": test_case.get("lang", "unknown")
                }
                errors.append(error_info)
                
                # Categorize error patterns
                error_key = f"{expected_intent} → {actual_intent}"
                error_patterns[error_key].append(error_info)
                
                print(f"❌ {test_id}: '{query[:60]}...'")
                print(f"   Expected: {expected_intent}, Got: {actual_intent}")
                print()
                
        except Exception as e:
            print(f"💥 {test_id}: ERROR - {e}")
            
    print(f"\n📊 INTENT CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total tests: {total}")
    print(f"Correct: {correct}")
    print(f"Errors: {len(errors)}")
    print(f"Accuracy: {correct/total*100:.1f}%")
    
    print(f"\n🔍 ERROR PATTERNS ANALYSIS")
    print("=" * 50)
    
    for error_pattern, cases in sorted(error_patterns.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(cases)
        print(f"\n{error_pattern}: {count} cases")
        for case in cases[:3]:  # Show first 3 examples
            print(f"  • {case['test_id']}: '{case['query'][:50]}...'")
        if len(cases) > 3:
            print(f"  ... and {len(cases)-3} more")
    
    print(f"\n📋 DETAILED ERROR ANALYSIS")
    print("=" * 50)
    
    # Group by expected intent
    by_expected = defaultdict(list)
    for error in errors:
        by_expected[error["expected"]].append(error)
    
    for expected_intent in sorted(by_expected.keys()):
        cases = by_expected[expected_intent]
        print(f"\n🎯 Expected '{expected_intent}' ({len(cases)} errors):")
        
        actual_intents = defaultdict(int)
        for case in cases:
            actual_intents[case["actual"]] += 1
            
        for actual, count in sorted(actual_intents.items(), key=lambda x: x[1], reverse=True):
            print(f"  → Got '{actual}': {count} times")
            examples = [c for c in cases if c["actual"] == actual][:2]
            for ex in examples:
                print(f"    • {ex['test_id']}: '{ex['query'][:45]}...'")
    
    return errors, error_patterns

def categorize_improvements(errors, error_patterns):
    """Suggest specific improvements based on error patterns"""
    
    print(f"\n🛠️  SUGGESTED IMPROVEMENTS")
    print("=" * 50)
    
    improvements = []
    
    # Common error patterns and their solutions
    
    # 1. Author/title confusion  
    author_title_errors = [e for e in errors if e["expected"] == "author_title" and e["actual"] in ["mixed_filters", "topic", "title", "author"]]
    if author_title_errors:
        print(f"\n1. 📚 AUTHOR+TITLE DETECTION ({len(author_title_errors)} errors)")
        print("   Problem: Queries with both author and title not detected properly")
        print("   Solution: Improve natural language author+title pattern detection")
        for err in author_title_errors[:2]:
            print(f"   Example: '{err['query']}' → expected {err['expected']}, got {err['actual']}")
        improvements.append({
            "type": "author_title_detection",
            "priority": "high",
            "count": len(author_title_errors)
        })
    
    # 2. Mixed filters vs simple intents
    mixed_filter_errors = [e for e in errors if e["actual"] == "mixed_filters" and e["expected"] in ["author_title", "topic", "genre"]]
    if mixed_filter_errors:
        print(f"\n2. 🔀 MIXED FILTERS OVER-CLASSIFICATION ({len(mixed_filter_errors)} errors)")
        print("   Problem: Simple queries classified as mixed_filters")
        print("   Solution: Tighten mixed_filters criteria, improve simple intent detection")
        for err in mixed_filter_errors[:2]:
            print(f"   Example: '{err['query']}' → expected {err['expected']}, got {err['actual']}")
        improvements.append({
            "type": "mixed_filters_overuse", 
            "priority": "high",
            "count": len(mixed_filter_errors)
        })
    
    # 3. Topic vs Genre confusion
    topic_genre_errors = [e for e in errors if (e["expected"] == "topic" and e["actual"] == "genre") or (e["expected"] == "genre" and e["actual"] == "topic")]
    if topic_genre_errors:
        print(f"\n3. 🏷️  TOPIC vs GENRE CONFUSION ({len(topic_genre_errors)} errors)")
        print("   Problem: Unclear distinction between topics and genres")
        print("   Solution: Better prompt examples and clearer definitions")
        for err in topic_genre_errors[:2]:
            print(f"   Example: '{err['query']}' → expected {err['expected']}, got {err['actual']}")
        improvements.append({
            "type": "topic_genre_confusion",
            "priority": "medium", 
            "count": len(topic_genre_errors)
        })
    
    # 4. Language-specific issues
    lang_errors = defaultdict(list)
    for err in errors:
        lang_errors[err["lang"]].append(err)
    
    for lang, lang_errs in lang_errors.items():
        if len(lang_errs) > 3:  # Significant language-specific issues
            print(f"\n4. 🌐 LANGUAGE-SPECIFIC ISSUES - {lang.upper()} ({len(lang_errs)} errors)")
            print(f"   Problem: High error rate for {lang} queries")
            print(f"   Solution: Add more {lang} examples to prompts")
            for err in lang_errs[:2]:
                print(f"   Example: '{err['query']}' → expected {err['expected']}, got {err['actual']}")
            improvements.append({
                "type": f"language_{lang}_issues",
                "priority": "medium",
                "count": len(lang_errs)
            })
    
    # 5. Free text vs specific intents
    free_text_errors = [e for e in errors if e["actual"] == "free_text" and e["expected"] != "free_text"]
    if free_text_errors:
        print(f"\n5. 📝 FREE TEXT OVER-CLASSIFICATION ({len(free_text_errors)} errors)")
        print("   Problem: Specific queries falling back to free_text")
        print("   Solution: Improve pattern recognition for structured queries")
        for err in free_text_errors[:2]:
            print(f"   Example: '{err['query']}' → expected {err['expected']}, got {err['actual']}")
        improvements.append({
            "type": "free_text_overuse",
            "priority": "high",
            "count": len(free_text_errors)
        })
    
    return improvements

if __name__ == "__main__":
    errors, patterns = analyze_intent_classification()
    improvements = categorize_improvements(errors, patterns)
    
    print(f"\n🎯 TOP IMPROVEMENT PRIORITIES:")
    print("=" * 40)
    
    high_priority = [i for i in improvements if i["priority"] == "high"]
    medium_priority = [i for i in improvements if i["priority"] == "medium"]
    
    for i, imp in enumerate(sorted(high_priority, key=lambda x: x["count"], reverse=True), 1):
        print(f"{i}. HIGH: {imp['type']} ({imp['count']} errors)")
    
    for i, imp in enumerate(sorted(medium_priority, key=lambda x: x["count"], reverse=True), len(high_priority)+1):
        print(f"{i}. MEDIUM: {imp['type']} ({imp['count']} errors)")