#!/usr/bin/env python3
"""
Realistic test runner для проверки качества поиска
"""
import json
import asyncio
import time
from app.simple_orchestrator import process_simple_search

async def run_single_test(test_data):
    """Запускает один тест и возвращает результат"""
    query = test_data['query']
    expected_outcome = test_data['expected_outcome']
    expected_doc_id = test_data.get('expected_doc_id')
    
    print(f"\n🔍 Тест {test_data['id']}: '{query}'")
    print(f"   Категория: {test_data['category']}")
    print(f"   Ожидается: {expected_outcome}")
    
    try:
        start_time = time.time()
        result = await process_simple_search("test_session", query)
        elapsed = time.time() - start_time
        
        # Analyze result
        found_books = result.get('results', [])
        found_any = len(found_books) > 0
        
        if expected_outcome == "success":
            # Should find the needed book
            success = False
            if expected_doc_id:
                # Check that we found the right book by ID
                for book in found_books:
                    if book.get('metadata', {}).get('document_id') == expected_doc_id:
                        success = True
                        break
            else:
                # Should just find something
                success = found_any
                
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Result: {status} - found {len(found_books)} books in {elapsed:.2f}s")
            
            if found_books:
                for i, book in enumerate(found_books):
                    title = book.get('title', 'Unknown')
                    author = book.get('author', 'Unknown')
                    score = book.get('score', 0)
                    print(f"   [{i+1}] {title} by {author} (score: {score:.3f})")
                    
        elif expected_outcome == "failure":
            # Should not have found a book
            success = not found_any
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Result: {status} - found {len(found_books)} books (expected 0)")
            
        elif expected_outcome == "clarify":
            # Should find but with clarification about inaccuracy
            success = found_any  # Basic check - found something
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   Result: {status} - found {len(found_books)} books (needs clarification)")
            
        return {
            'test_id': test_data['id'],
            'query': query,
            'category': test_data['category'],
            'expected': expected_outcome,
            'success': success,
            'found_count': len(found_books),
            'elapsed': elapsed,
            'books': found_books
        }
        
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        return {
            'test_id': test_data['id'],
            'query': query,
            'category': test_data['category'],
            'expected': expected_outcome,
            'success': False,
            'error': str(e),
            'found_count': 0,
            'elapsed': 0,
            'books': []
        }

async def run_realistic_tests(limit=None):
    """Runs realistic tests"""
    print("🚀 Running Realistic Tests")
    print("=" * 80)
    
    # Load tests
    with open('realistic_tests.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tests = data['realistic_tests']
    if limit:
        tests = tests[:limit]
    
    print(f"📊 Total tests: {len(tests)}")
    
    # Statistics by categories
    categories = {}
    for test in tests:
        cat = test['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("📋 Categories:")
    for cat, count in categories.items():
        print(f"   {cat}: {count} tests")
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    
    for test in tests:
        result = await run_single_test(test)
        results.append(result)
        
        if result['success']:
            passed += 1
        else:
            failed += 1
    
    # Final report
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    # Statistics by categories
    print("\n📋 Results by categories:")
    cat_stats = {}
    for result in results:
        cat = result['category']
        if cat not in cat_stats:
            cat_stats[cat] = {'passed': 0, 'failed': 0}
        
        if result['success']:
            cat_stats[cat]['passed'] += 1
        else:
            cat_stats[cat]['failed'] += 1
    
    for cat, stats in cat_stats.items():
        total = stats['passed'] + stats['failed']
        success_rate = stats['passed'] / total * 100 if total > 0 else 0
        print(f"   {cat}: {stats['passed']}/{total} ({success_rate:.1f}%)")
    
    # Save detailed results
    with open('realistic_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_tests': len(tests),
                'passed': passed,
                'failed': failed,
                'success_rate': passed/(passed+failed)*100,
                'category_stats': cat_stats
            },
            'detailed_results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Детальные результаты сохранены в realistic_test_results.json")
    
    return results

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(run_realistic_tests(limit))