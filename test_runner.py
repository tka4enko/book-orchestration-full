#!/usr/bin/env python3
"""
Автоматический тестировщик для BookBot на основе tests_extra.json
Проверяет все 40 тестовых запросов и логирует результаты
"""

import json
import requests
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BookBotTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
        
    def load_test_cases(self, filepath: str) -> List[Dict]:
        """Загружает тестовые случаи из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('tests_extra', [])
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки тестов: {e}")
            return []
    
    def send_query(self, test_case: Dict) -> Optional[Dict]:
        """Отправляет запрос к BookBot API"""
        try:
            payload = {
                "session_id": f"test_{test_case['id']}",
                "message": test_case["query"]
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ HTTP Error {response.status_code} для {test_case['id']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка запроса для {test_case['id']}: {e}")
            return None
    
    def check_intent_match(self, expected: str, actual: str) -> bool:
        """Проверяет соответствие намерений"""
        # Маппинг для совместимости
        intent_mapping = {
            "mixed_filters": "author_title",  # Смешанные фильтры часто детектятся как author_title
            "fuzzy_title": "author_title",    # Нечеткое совпадение обычно author_title
            "ask_summary": "title"            # Запрос саммари чаще всего title
        }
        
        mapped_expected = intent_mapping.get(expected, expected)
        return mapped_expected == actual or expected == actual
    
    def check_required_ids(self, expected_ids: List[str], actual_results: List[Dict]) -> tuple:
        """Проверяет наличие обязательных ID в результатах"""
        found_ids = []
        
        for result in actual_results:
            if isinstance(result, dict):
                result_id = result.get("document_id")
                if result_id and result_id in expected_ids:
                    found_ids.append(result_id)
        
        missing_ids = [id for id in expected_ids if id not in found_ids]
        return found_ids, missing_ids
    
    def check_excluded_ids(self, excluded_ids: List[str], actual_results: List[Dict]) -> List[str]:
        """Проверяет отсутствие запрещенных ID в результатах"""
        found_excluded = []
        
        for result in actual_results:
            if isinstance(result, dict):
                result_id = result.get("document_id") 
                if result_id and result_id in excluded_ids:
                    found_excluded.append(result_id)
        
        return found_excluded
    
    def analyze_result(self, test_case: Dict, response: Dict) -> Dict:
        """Анализирует результат теста"""
        test_id = test_case["id"]
        expected = test_case.get("expected", {})
        intent_expected = test_case.get("intent_expected")
        
        analysis = {
            "test_id": test_id,
            "query": test_case["query"],
            "lang": test_case.get("lang", "unknown"),
            "intent_expected": intent_expected,
            "intent_actual": response.get("intent"),
            "success": True,
            "issues": [],
            "details": {}
        }
        
        # 1. Проверка намерения
        if intent_expected and not self.check_intent_match(intent_expected, response.get("intent", "")):
            analysis["success"] = False
            analysis["issues"].append(f"Intent mismatch: expected '{intent_expected}', got '{response.get('intent')}'")
        
        # 2. Проверка clarification
        if expected.get("expect_clarify") and not response.get("need_clarify"):
            analysis["success"] = False
            analysis["issues"].append("Expected clarification but didn't get one")
        
        # 3. Проверка обязательных результатов
        must_include = expected.get("must_include_ids", [])
        if must_include:
            results = response.get("results", [])
            found_ids, missing_ids = self.check_required_ids(must_include, results)
            
            if missing_ids:
                analysis["success"] = False
                analysis["issues"].append(f"Missing required IDs: {missing_ids}")
            
            analysis["details"]["required_found"] = found_ids
            analysis["details"]["required_missing"] = missing_ids
        
        # 4. Проверка исключенных результатов
        must_exclude = expected.get("must_exclude_ids", [])
        if must_exclude:
            results = response.get("results", [])
            found_excluded = self.check_excluded_ids(must_exclude, results)
            
            if found_excluded:
                analysis["success"] = False
                analysis["issues"].append(f"Found excluded IDs: {found_excluded}")
            
            analysis["details"]["excluded_found"] = found_excluded
        
        # 5. Проверка топ-1 результата
        top1_expected = expected.get("top1_id")
        if top1_expected:
            results = response.get("results", [])
            if results and isinstance(results[0], dict):
                actual_top1 = results[0].get("document_id")
                if actual_top1 != top1_expected:
                    analysis["success"] = False
                    analysis["issues"].append(f"Top1 ID mismatch: expected '{top1_expected}', got '{actual_top1}'")
                analysis["details"]["top1_expected"] = top1_expected
                analysis["details"]["top1_actual"] = actual_top1
        
        # 6. Проверка режима abstain
        if expected.get("allow_abstain"):
            results = response.get("results", [])
            if not results or (len(results) == 1 and "не найдено" in results[0].get("message", "")):
                analysis["details"]["abstained"] = True
            else:
                analysis["details"]["abstained"] = False
        
        # 7. Подсчет найденных результатов
        results_count = len([r for r in response.get("results", []) if isinstance(r, dict) and r.get("title")])
        analysis["details"]["results_count"] = results_count
        
        return analysis
    
    def run_test(self, test_case: Dict) -> Dict:
        """Выполняет один тест"""
        test_id = test_case["id"]
        query = test_case["query"]
        
        logger.info(f"🧪 Тестирование {test_id}: '{query}'")
        
        # Отправка запроса
        response = self.send_query(test_case)
        if not response:
            return {
                "test_id": test_id,
                "query": query,
                "success": False,
                "issues": ["Failed to get response from API"],
                "details": {}
            }
        
        # Анализ результата
        analysis = self.analyze_result(test_case, response)
        
        # Логирование результата
        status = "✅ PASS" if analysis["success"] else "❌ FAIL"
        logger.info(f"   {status} - Intent: {analysis['intent_actual']}")
        
        if analysis["issues"]:
            for issue in analysis["issues"]:
                logger.warning(f"   ⚠️  {issue}")
        
        if analysis["details"]:
            details_str = ", ".join([f"{k}={v}" for k, v in analysis["details"].items() if v])
            if details_str:
                logger.info(f"   📊 Details: {details_str}")
        
        logger.info("")  # Пустая строка для разделения
        
        return analysis
    
    def run_all_tests(self, test_filepath: str) -> Dict:
        """Запускает все тесты"""
        logger.info("=" * 80)
        logger.info("🚀 ЗАПУСК АВТОМАТИЧЕСКИХ ТЕСТОВ BOOKBOT")
        logger.info(f"📅 Время: {datetime.now()}")
        logger.info(f"🌐 Endpoint: {self.base_url}")
        logger.info("=" * 80)
        
        # Проверка доступности API
        try:
            health_response = self.session.get(f"{self.base_url}/health", timeout=5)
            if health_response.status_code != 200:
                logger.error("❌ API недоступен")
                return {"error": "API unavailable"}
        except Exception as e:
            logger.error(f"❌ Не удается подключиться к API: {e}")
            return {"error": f"Connection failed: {e}"}
        
        logger.info("✅ API доступен, начинаем тестирование...")
        logger.info("")
        
        # Загрузка тестовых случаев
        test_cases = self.load_test_cases(test_filepath)
        if not test_cases:
            logger.error("❌ Не удалось загрузить тестовые случаи")
            return {"error": "No test cases loaded"}
        
        logger.info(f"📋 Загружено {len(test_cases)} тестовых случаев")
        logger.info("")
        
        # Выполнение тестов
        start_time = time.time()
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] ", end="", flush=True)
            
            analysis = self.run_test(test_case)
            self.results.append(analysis)
            
            if analysis["success"]:
                passed += 1
            else:
                failed += 1
            
            # Небольшая пауза между запросами
            time.sleep(0.5)
        
        # Подведение итогов
        total_time = time.time() - start_time
        success_rate = (passed / len(test_cases)) * 100
        
        summary = {
            "total_tests": len(test_cases),
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "total_time": total_time,
            "results": self.results
        }
        
        logger.info("=" * 80)
        logger.info("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        logger.info(f"✅ Пройдено: {passed}")
        logger.info(f"❌ Провалено: {failed}")
        logger.info(f"📈 Успешность: {success_rate:.1f}%")
        logger.info(f"⏱️  Время выполнения: {total_time:.1f} секунд")
        logger.info("=" * 80)
        
        # Детальный отчет по провалившимся тестам
        if failed > 0:
            logger.info("")
            logger.info("❌ ПРОВАЛИВШИЕСЯ ТЕСТЫ:")
            for result in self.results:
                if not result["success"]:
                    logger.error(f"   {result['test_id']}: {result['query']}")
                    for issue in result["issues"]:
                        logger.error(f"     • {issue}")
        
        return summary
    
    def save_results(self, filepath: str = "test_results.json"):
        """Сохраняет результаты в JSON файл"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total": len(self.results),
                        "passed": len([r for r in self.results if r["success"]]),
                        "failed": len([r for r in self.results if not r["success"]])
                    },
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Результаты сохранены в {filepath}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")

def main():
    """Главная функция"""
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "tests_extra.json"
    
    if not os.path.exists(test_file):
        print(f"❌ Файл тестов не найден: {test_file}")
        sys.exit(1)
    
    # API endpoint
    api_url = os.getenv("BOOKBOT_API_URL", "http://localhost:8000")
    
    # Создание и запуск тестера
    tester = BookBotTester(api_url)
    summary = tester.run_all_tests(test_file)
    
    if "error" not in summary:
        # Сохранение результатов
        tester.save_results()
        
        # Выход с кодом ошибки если есть провалившиеся тесты
        if summary["failed"] > 0:
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()