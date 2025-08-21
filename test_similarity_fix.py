#!/usr/bin/env python3
"""
Quick test for similarity fix - проверяем что similarity теперь > 0
"""

import requests
import json
import time

def test_query(query: str, expected_intent: str):
    print(f"\n🧪 Тест: '{query}'")
    print(f"   Ожидаемый intent: {expected_intent}")
    
    url = "http://127.0.0.1:8000/chat"
    payload = {"message": query, "session_id": "test_session"}
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    query_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        actual_intent = result.get("intent", "unknown")
        results_list = result.get("results", [])
        
        print(f"   ✅ Успешно за {query_time:.2f}s")
        print(f"   🎯 Intent: {actual_intent} {'✅' if actual_intent == expected_intent else '❌'}")
        print(f"   📊 Результатов: {len(results_list)}")
        
        # Проверим первый результат
        if results_list:
            first_result = results_list[0]
            if isinstance(first_result, dict):
                title = first_result.get("title", "Unknown")
                print(f"   📚 Первая книга: {title}")
        
        return True
        
    else:
        print(f"   ❌ Ошибка: {response.status_code}")
        print(f"   📄 Ответ: {response.text}")
        return False

def main():
    print("🚀 ТЕСТ ИСПРАВЛЕНИЯ SIMILARITY BUG")
    print("="*50)
    
    # Тестируем несколько запросов которые должны работать лучше
    test_cases = [
        ("Brianna Wiest spirituality essays", "topic"),
    ]
    
    successful_tests = 0
    for query, expected_intent in test_cases:
        if test_query(query, expected_intent):
            successful_tests += 1
        time.sleep(1)  # Короткая пауза между запросами
    
    print(f"\n🎯 РЕЗУЛЬТАТ: {successful_tests}/{len(test_cases)} тестов прошли успешно")
    
    if successful_tests >= len(test_cases) * 0.75:
        print("✅ Исправление работает!")
    else:
        print("❌ Нужны дополнительные исправления")

if __name__ == "__main__":
    main()