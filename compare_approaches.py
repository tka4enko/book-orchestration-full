#!/usr/bin/env python3
"""
Сравнение старого и нового подходов поиска
"""

import requests
import json
import time
import sys

def test_endpoint(url: str, query: str, session_id: str, approach_name: str):
    """Тестирует конкретный endpoint"""
    
    print(f"\n🔍 {approach_name}: '{query}'")
    print("-" * 50)
    
    payload = {
        "session_id": session_id,
        "message": query
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            results_count = len(data.get('results', []))
            intent = data.get('intent', 'N/A')
            
            print(f"✅ Успешно: {results_count} результатов")
            print(f"   Intent: {intent}")
            print(f"   Время: {elapsed_time:.2f}с")
            
            # Показываем первые 3 результата
            results = data.get('results', [])[:3]
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Unknown')
                author = result.get('author', 'Unknown')
                
                # Для старого подхода может быть другая структура
                if 'score' in result:
                    score = result.get('score', 0.0)
                    print(f"   {i}. \"{title}\" - {author} (score: {score:.3f})")
                else:
                    print(f"   {i}. \"{title}\" - {author}")
            
            return {
                'success': True,
                'results_count': results_count,
                'time': elapsed_time,
                'intent': intent
            }
        else:
            print(f"❌ Ошибка HTTP {response.status_code}")
            return {
                'success': False,
                'time': elapsed_time,
                'error': f"HTTP {response.status_code}"
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Ошибка: {e}")
        return {
            'success': False,
            'time': elapsed_time,
            'error': str(e)
        }

def compare_approaches(query: str):
    """Сравнивает оба подхода для одного запроса"""
    
    print(f"\n{'='*70}")
    print(f"🆚 СРАВНЕНИЕ ПОДХОДОВ: '{query}'")
    print(f"{'='*70}")
    
    base_url = "http://localhost:8000"
    session_id = f"compare_{int(time.time())}"
    
    # Тестируем старый подход
    old_result = test_endpoint(
        f"{base_url}/chat", 
        query, 
        session_id, 
        "СТАРЫЙ ПОДХОД (BM25 + Vector)"
    )
    
    # Тестируем новый подход
    new_result = test_endpoint(
        f"{base_url}/simple_chat", 
        query, 
        session_id, 
        "НОВЫЙ ПОДХОД (Simple Vector + LLM Filter)"
    )
    
    # Сравнение
    print(f"\n📊 СРАВНЕНИЕ:")
    print(f"   Старый подход:")
    if old_result['success']:
        print(f"     ✅ {old_result['results_count']} результатов за {old_result['time']:.2f}с")
        print(f"     Intent: {old_result.get('intent', 'N/A')}")
    else:
        print(f"     ❌ Ошибка: {old_result.get('error', 'Unknown')}")
    
    print(f"   Новый подход:")
    if new_result['success']:
        print(f"     ✅ {new_result['results_count']} результатов за {new_result['time']:.2f}с")
        print(f"     Intent: {new_result.get('intent', 'N/A')}")
    else:
        print(f"     ❌ Ошибка: {new_result.get('error', 'Unknown')}")
    
    # Анализ производительности
    if old_result['success'] and new_result['success']:
        time_diff = new_result['time'] - old_result['time']
        if time_diff < 0:
            print(f"     ⚡ Новый подход быстрее на {abs(time_diff):.2f}с")
        else:
            print(f"     🐌 Новый подход медленнее на {time_diff:.2f}с")

def main():
    """Основная функция"""
    
    if len(sys.argv) > 1:
        # Сравниваем для переданного запроса
        query = " ".join(sys.argv[1:])
        compare_approaches(query)
    else:
        # Набор тестовых запросов
        test_queries = [
            "найди книги Орвелла",
            "1984",
            "детективы но не Агата Кристи", 
            "фантастика",
            "книги о психологии"
        ]
        
        print("🧪 СРАВНИТЕЛЬНЫЕ ТЕСТЫ")
        print("=" * 70)
        print("Тестируем оба подхода на наборе запросов...")
        
        for query in test_queries:
            compare_approaches(query)
            time.sleep(1)  # Небольшая пауза между тестами
        
        print(f"\n🎉 Сравнительные тесты завершены!")
        print("\nℹ️ Для тестирования конкретного запроса:")
        print("   python compare_approaches.py 'ваш запрос'")

if __name__ == "__main__":
    main()