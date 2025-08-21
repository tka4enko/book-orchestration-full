#!/usr/bin/env python3
"""
Генератор краткого отчета по результатам тестирования BookBot
"""

import json
import sys
from datetime import datetime
from collections import Counter

def generate_report(results_file: str = "test_results.json"):
    """Генерирует краткий отчет по результатам тестирования"""
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения файла результатов: {e}")
        return
    
    results = data.get("results", [])
    summary = data.get("summary", {})
    timestamp = data.get("timestamp", "")
    
    # Основная статистика
    total = summary.get("total", len(results))
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print("=" * 60)
    print("📊 ОТЧЕТ ПО ТЕСТИРОВАНИЮ BOOKBOT")
    print("=" * 60)
    print(f"🕒 Время выполнения: {timestamp}")
    print(f"📈 Общая статистика:")
    print(f"   • Всего тестов: {total}")
    print(f"   • ✅ Успешно: {passed} ({success_rate:.1f}%)")
    print(f"   • ❌ Провалено: {failed} ({100-success_rate:.1f}%)")
    print()
    
    # Анализ по типам намерений
    intent_stats = Counter()
    intent_success = Counter()
    
    for result in results:
        expected_intent = result.get("intent_expected")
        actual_intent = result.get("intent_actual")
        success = result.get("success", False)
        
        intent_stats[expected_intent] += 1
        if success:
            intent_success[expected_intent] += 1
    
    print("📋 Статистика по типам намерений:")
    for intent in sorted(intent_stats.keys()):
        total_intent = intent_stats[intent]
        success_intent = intent_success.get(intent, 0)
        rate = (success_intent / total_intent * 100) if total_intent > 0 else 0
        print(f"   • {intent}: {success_intent}/{total_intent} ({rate:.0f}%)")
    print()
    
    # Анализ основных проблем
    issue_types = Counter()
    for result in results:
        if not result.get("success", False):
            for issue in result.get("issues", []):
                if "Intent mismatch" in issue:
                    issue_types["Неверное определение намерения"] += 1
                elif "Top1 ID mismatch" in issue:
                    issue_types["Неверный результат поиска"] += 1
                elif "Missing required IDs" in issue:
                    issue_types["Не найдены обязательные книги"] += 1
                elif "Expected clarification" in issue:
                    issue_types["Не запросил уточнение"] += 1
                elif "Found excluded IDs" in issue:
                    issue_types["Вернул исключенные результаты"] += 1
    
    print("🔍 Основные типы проблем:")
    for issue, count in issue_types.most_common():
        print(f"   • {issue}: {count}")
    print()
    
    # Успешные тесты
    successful_tests = [r for r in results if r.get("success", False)]
    if successful_tests:
        print("✅ Успешные тесты:")
        for result in successful_tests:
            print(f"   • {result['test_id']}: {result['query'][:50]}...")
        print()
    
    # Наиболее проблемные тесты
    failed_tests = [r for r in results if not r.get("success", False)]
    high_priority_failures = []
    
    for result in failed_tests:
        issues = result.get("issues", [])
        # Приоритет проблем
        if any("Missing required IDs" in issue for issue in issues):
            high_priority_failures.append((result, "Не найдена ожидаемая книга"))
        elif any("Top1 ID mismatch" in issue for issue in issues):
            high_priority_failures.append((result, "Неверный результат поиска"))
    
    if high_priority_failures:
        print("🚨 Критические проблемы (топ-10):")
        for result, problem in high_priority_failures[:10]:
            print(f"   • {result['test_id']}: {problem}")
            print(f"     Запрос: {result['query']}")
            print(f"     Ожидал: {result['intent_expected']}, получил: {result['intent_actual']}")
            print()
    
    # Рекомендации по улучшению
    print("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    print()
    
    if issue_types.get("Неверное определение намерения", 0) > 5:
        print("1. 🎯 Улучшить LLM prompt для определения намерений")
        print("   • Добавить больше примеров в INTENT_SYS")
        print("   • Уточнить граничные случаи между intent типами")
        print()
    
    if issue_types.get("Неверный результат поиска", 0) > 5:
        print("2. 🔍 Улучшить cross-language matching")
        print("   • Реализовать поддержку title_alias в retrievers")
        print("   • Добавить перевод запросов между языками")
        print()
    
    if issue_types.get("Не найдены обязательные книги", 0) > 5:
        print("3. 📚 Улучшить поиск похожих названий")
        print("   • Расширить fuzzy matching на названия книг")
        print("   • Добавить поиск по частичным совпадениям")
        print()
    
    if issue_types.get("Не запросил уточнение", 0) > 3:
        print("4. ❓ Улучшить детекцию конфликтов")
        print("   • Добавить проверки конфликтов для всех intent типов")
        print("   • Расширить логику _check_author_title_conflicts")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "test_results.json"
    generate_report(results_file)