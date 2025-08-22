#!/usr/bin/env python3
"""
Тест для нового простого поиска через /simple_chat endpoint
"""

import requests
import json
import time
import sys

def test_simple_chat(query: str, session_id: str = "test_simple"):
    """Тестирует новый простой поиск"""
    
    print(f"🔍 Тестируем простой поиск: '{query}'")
    print("=" * 60)
    
    url = "http://localhost:8000/simple_chat"
    payload = {
        "session_id": session_id,
        "message": query
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Время запроса: {elapsed_time:.2f}с")
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Успешный ответ:")
            print(f"   Intent: {data.get('intent', 'N/A')}")
            print(f"   Анализ: {data.get('analysis', 'N/A')}")
            print(f"   Результатов: {len(data.get('results', []))}")
            
            # Show results
            results = data.get('results', [])
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Unknown')
                author = result.get('author', 'Unknown')
                score = result.get('score', 0.0)
                collection = result.get('collection', 'unknown')
                
                print(f"   {i}. \"{title}\" - {author}")
                print(f"      Score: {score:.3f}, Collection: {collection}")
            
            print(f"\n💬 Ответ пользователю:")
            print(f"   {data.get('response', 'N/A')}")
            
            # Performance metrics
            perf_metrics = data.get('performance_metrics', {})
            search_stats = data.get('search_stats', {})
            
            if perf_metrics:
                print(f"\n📊 Метрики производительности:")
                for key, value in perf_metrics.items():
                    print(f"   {key}: {value:.2f}с")
            
            if search_stats:
                print(f"\n📈 Статистика поиска:")
                for key, value in search_stats.items():
                    if isinstance(value, (int, float)):
                        if key.endswith('time'):
                            print(f"   {key}: {value:.2f}с")
                        else:
                            print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
                        
        else:
            print(f"❌ Ошибка HTTP {response.status_code}:")
            try:
                error_data = response.json()
                print(f"   {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
    
    print("=" * 60)

def main():
    """Запуск тестов"""
    
    if len(sys.argv) > 1:
        # Use provided query
        query = " ".join(sys.argv[1:])
        test_simple_chat(query)
    else:
        # Set of test queries
        test_queries = [
            "найди книги Орвелла",
            "1984 Оруэлл",
            "детективы",
            "фантастика но не Азимов",
            "книги о психологии",
            "Толстой Война и мир",
            "романы кроме русских авторов",
        ]
        
        print("🧪 Запускаем набор тестов для простого поиска")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Тест {i}/{len(test_queries)}")
            test_simple_chat(query, f"test_simple_{i}")
            
            if i < len(test_queries):
                print("\n⏳ Пауза между тестами...")
                time.sleep(2)
        
        print("\n🎉 Все тесты завершены!")

if __name__ == "__main__":
    main()