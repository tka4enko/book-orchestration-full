"""
Debug Reporter - Compact query execution analysis system
Provides easy-to-read summaries of search pipeline execution
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
    from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OPENAI_MODEL_CHAT = None
    OPENAI_API_KEY = None

logger = logging.getLogger(__name__)

@dataclass
class SearchStep:
    """Individual search step tracking"""
    name: str
    query: str
    success: bool
    results_count: int
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DebugSession:
    """Complete debug session for one query"""
    original_query: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Intent detection details
    intent: str = ""
    intent_confidence: str = ""
    intent_method: str = ""  # "function" or "llm" 
    intent_details: Dict[str, Any] = field(default_factory=dict)  # Details of intent detection
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Query processing
    simple_query_check: bool = False
    mixed_filters_used: bool = False
    predicate_filter_used: bool = False
    
    # Search queries
    bm25_query: str = ""
    vector_query: str = ""
    
    # Search steps
    steps: List[SearchStep] = field(default_factory=list)
    
    # Final result
    final_results_count: int = 0
    execution_time_ms: float = 0.0
    final_response: str = ""
    final_books: List[Dict[str, Any]] = field(default_factory=list)  # List of final books returned
    
    # Filtering details
    filtered_results: List[Dict[str, Any]] = field(default_factory=list)  # What was filtered and why
    
    # Analysis
    llm_analysis: str = ""
    success: bool = True
    issues: List[str] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    search_metrics: Dict[str, float] = field(default_factory=dict)

class DebugReporter:
    """Compact debug reporting system with LLM analysis"""
    
    def __init__(self):
        self.enabled = os.getenv('DEBUG_REPORTER_ENABLED', 'false').lower() == 'true'
        self.llm_analysis_enabled = os.getenv('DEBUG_LLM_ANALYSIS', 'false').lower() == 'true'
        self.current_session: Optional[DebugSession] = None
        
        # LLM for analysis (optional)
        self.llm = None
        if self.llm_analysis_enabled and HAS_OPENAI and OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=OPENAI_MODEL_CHAT,
                    temperature=0,
                    max_tokens=300,
                    api_key=OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for debug analysis: {e}")
                self.llm_analysis_enabled = False

    def start_session(self, query: str, session_id: str = "default") -> None:
        """Start new debug session"""
        if not self.enabled:
            return
            
        self.current_session = DebugSession(
            original_query=query,
            session_id=session_id
        )
        logger.debug(f"🔍 DEBUG SESSION STARTED: {query} (session: {session_id})")
        
    def set_intent(self, intent: str, filters: Dict[str, Any], confidence: str = "", 
                  method: str = "", details: Dict[str, Any] = None) -> None:
        """Record intent detection results"""
        if not self.enabled or not self.current_session:
            return
            
        self.current_session.intent = intent
        self.current_session.filters = filters.copy()
        self.current_session.intent_confidence = confidence
        self.current_session.intent_method = method
        self.current_session.intent_details = details.copy() if details else {}
        
    def add_search_step(self, name: str, query: str, success: bool, 
                       results_count: int, reason: str = "", **details) -> None:
        """Add search step to tracking"""
        if not self.enabled or not self.current_session:
            return
            
        step = SearchStep(
            name=name,
            query=query,
            success=success,
            results_count=results_count,
            reason=reason,
            details=details
        )
        self.current_session.steps.append(step)
        
    def add_issue(self, issue: str) -> None:
        """Add issue to current session"""
        if not self.enabled or not self.current_session:
            return
        self.current_session.issues.append(issue)
        
    def set_query_processing(self, simple_query: bool = False, mixed_filters: bool = False, 
                           predicate_filter: bool = False) -> None:
        """Record query processing flags"""
        if not self.enabled or not self.current_session:
            return
        self.current_session.simple_query_check = simple_query
        self.current_session.mixed_filters_used = mixed_filters
        self.current_session.predicate_filter_used = predicate_filter
        
    def set_search_queries(self, bm25_query: str = "", vector_query: str = "") -> None:
        """Record actual search queries used"""
        if not self.enabled or not self.current_session:
            return
        if bm25_query:
            self.current_session.bm25_query = bm25_query
        if vector_query:
            self.current_session.vector_query = vector_query
            
    def add_filtered_result(self, result_info: Dict[str, Any]) -> None:
        """Record filtering decision"""
        if not self.enabled or not self.current_session:
            return
        self.current_session.filtered_results.append(result_info)
        
    def set_final_response(self, response: str) -> None:
        """Record final response message"""
        if not self.enabled or not self.current_session:
            return
        self.current_session.final_response = response
    
    def set_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics"""
        if not self.enabled or not self.current_session:
            return
            
        self.current_session.performance_metrics = metrics.copy()
    
    def set_search_metrics(self, metrics: Dict[str, float]) -> None:
        """Record search metrics"""
        if not self.enabled or not self.current_session:
            return
            
        self.current_session.search_metrics = metrics.copy()
        
    def set_final_books(self, books: List[Dict[str, Any]]) -> None:
        """Record final books returned to user"""
        if not self.enabled or not self.current_session:
            return
        self.current_session.final_books = books.copy() if books else []
        
    def finalize_session(self, final_results_count: int, execution_time_ms: float = 0.0) -> None:
        """Finalize session and generate report"""
        if not self.enabled:
            return
            
        if not self.current_session:
            logger.warning("🔍 DEBUG: finalize_session called but no current_session!")
            return
            
        logger.debug(f"🔍 DEBUG SESSION FINALIZING: {self.current_session.original_query}")
            
        self.current_session.final_results_count = final_results_count
        self.current_session.execution_time_ms = execution_time_ms
        self.current_session.success = final_results_count > 0 and len(self.current_session.issues) == 0
        
        # Generate LLM analysis if enabled
        if self.llm_analysis_enabled and self.llm:
            self._generate_llm_analysis()
            
        # Print compact report
        self._print_compact_report()
        
        # Reset for next query
        self.current_session = None
        
    def _generate_llm_analysis(self) -> None:
        """Generate LLM analysis of the debug session"""
        if not self.current_session or not self.llm:
            return
            
        try:
            session_summary = {
                "query": self.current_session.original_query,
                "intent": self.current_session.intent,
                "filters": self.current_session.filters,
                "steps": [
                    {
                        "name": step.name,
                        "success": step.success,
                        "results": step.results_count,
                        "reason": step.reason
                    }
                    for step in self.current_session.steps
                ],
                "final_results": self.current_session.final_results_count,
                "issues": self.current_session.issues
            }
            
            system_prompt = """You are a search system analyzer. Analyze the search execution and provide brief insights.
Focus on:
1. Whether the intent detection was appropriate
2. Why search steps succeeded or failed
3. Potential improvements
4. Root cause of issues

Be concise (2-3 sentences max). Use emojis for readability.

IMPORTANT: Always respond in Russian language."""

            user_prompt = f"Analyze this search execution:\n\n{json.dumps(session_summary, indent=2)}"
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            self.current_session.llm_analysis = response.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            self.current_session.llm_analysis = "❌ Анализ не удался"
            
    def _print_compact_report(self) -> None:
        """Печать расширенного отчета отладки в терминал"""
        if not self.current_session:
            return
            
        session = self.current_session
        
        # Статус выполнения
        status = "✅ УСПЕШНО" if session.success else "❌ ОШИБКА"
        
        print(f"\n{'='*100}")
        print(f"🔍 ОТЧЕТ ОТЛАДКИ | {status} | {session.execution_time_ms:.0f}ms")
        print(f"{'='*100}")
        
        # Запрос пользователя
        print(f"📝 Запрос пользователя: '{session.original_query}'")
        
        # ============= АНАЛИЗ НАМЕРЕНИЯ =============
        print(f"\n🧠 АНАЛИЗ НАМЕРЕНИЯ:")
        print(f"   💡 Определенное намерение: {session.intent} ({session.intent_confidence} уверенность)")
        print(f"   🔧 Метод определения: {session.intent_method or 'не указан'}")
        
        if session.intent_details:
            print(f"   📋 Детали анализа:")
            for key, value in session.intent_details.items():
                print(f"      • {key}: {value}")
        
        # Флаги обработки запроса
        print(f"   🔍 Простой запрос: {'Да' if session.simple_query_check else 'Нет'}")
        print(f"   ⚙️  Mixed filters: {'Использовались' if session.mixed_filters_used else 'Не использовались'}")
        print(f"   🎯 Predicate filter: {'Применялся' if session.predicate_filter_used else 'Не применялся'}")
        
        # Извлеченные фильтры
        if session.filters:
            print(f"\n🎯 ИЗВЛЕЧЕННЫЕ ФИЛЬТРЫ:")
            for key, value in session.filters.items():
                if value and value != [] and value != {}:
                    if isinstance(value, list):
                        print(f"   • {key}: [{', '.join(map(str, value))}]")
                    else:
                        print(f"   • {key}: {value}")
        
        # ============= ПОИСКОВЫЕ ЗАПРОСЫ =============
        print(f"\n🔎 ПОИСКОВЫЕ ЗАПРОСЫ:")
        if session.bm25_query:
            print(f"   📊 BM25 запрос: '{session.bm25_query}'")
        if session.vector_query:
            print(f"   🧮 Vector запрос: '{session.vector_query}'")
        if not session.bm25_query and not session.vector_query:
            print(f"   ❓ Поисковые запросы не записаны")
        
        # ============= ПОИСКОВЫЙ КОНВЕЙЕР =============
        print(f"\n🔎 ПОИСКОВЫЙ КОНВЕЙЕР:")
        if not session.steps:
            print(f"   ❓ Шаги поиска не записаны")
        else:
            for i, step in enumerate(session.steps, 1):
                icon = "✅" if step.success else "❌"
                results_info = f"({step.results_count} результатов)" if step.success else f"({step.reason})"
                query_display = step.query[:50] + "..." if len(step.query) > 50 else step.query
                print(f"  {i}. {icon} {step.name}: {results_info}")
                print(f"     🔍 Запрос: '{query_display}'")
                
                # Детальная информация если есть
                if step.details:
                    for key, value in step.details.items():
                        if key == 'similarity_score':
                            print(f"     📈 Схожесть: {value}")
                        elif key == 'threshold':
                            print(f"     🎯 Порог: {value}")
                        elif key == 'boost_score':
                            print(f"     ⚡ Буст: {value}")
                        elif key == 'search_type':
                            print(f"     🔧 Тип поиска: {value}")
                        elif key == 'intent':
                            print(f"     🧠 Намерение: {value}")
                        else:
                            print(f"     📋 {key}: {value}")
        
        # ============= ФИЛЬТРАЦИЯ РЕЗУЛЬТАТОВ =============
        if session.filtered_results:
            print(f"\n🎛️  ФИЛЬТРАЦИЯ РЕЗУЛЬТАТОВ:")
            for i, result in enumerate(session.filtered_results, 1):
                action = result.get('action', 'неизвестно')
                reason = result.get('reason', 'причина не указана')
                book_info = result.get('book_info', {})
                title = book_info.get('title', 'Без названия')
                author = book_info.get('author', 'Неизвестный автор')
                score = result.get('score', 'N/A')
                print(f"  {i}. {action}: '{title}' by {author} (оценка: {score})")
                print(f"     🔍 Причина: {reason}")
        
        # ============= МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ =============
        print(f"\n⏱️  МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
        print(f"   🕐 Общее время выполнения: {session.execution_time_ms:.1f}ms")
        
        # Детальные метрики поиска (если доступны)
        if session.search_metrics:
            print(f"   📊 Детальные метрики поиска:")
            for metric_name, value in session.search_metrics.items():
                if metric_name == "bm25_search_ms":
                    print(f"      🔤 BM25 поиск: {value:.1f}ms")
                elif metric_name == "vector_books_search_ms":
                    print(f"      📚 Vector books поиск: {value:.1f}ms")
                elif metric_name == "vector_content_search_ms":
                    print(f"      📄 Vector content поиск: {value:.1f}ms")
                elif metric_name == "results_filtering_ms":
                    print(f"      🎛️  Фильтрация результатов: {value:.1f}ms")
                elif metric_name == "initialization_ms":
                    print(f"      ⚙️  Инициализация: {value:.1f}ms")
                elif metric_name == "total_search_time_ms":
                    print(f"      🔍 Общее время поиска: {value:.1f}ms")
                else:
                    print(f"      📊 {metric_name}: {value:.1f}ms")
        
        # Метрики производительности этапов (если доступны)
        if session.performance_metrics:
            print(f"   📈 Метрики производительности этапов:")
            for metric_name, value in session.performance_metrics.items():
                if metric_name == "detect_intent_ms":
                    print(f"      🧠 Определение намерения: {value:.1f}ms")
                elif metric_name == "route_search_ms":
                    print(f"      🔍 Маршрутизация поиска: {value:.1f}ms")
                elif metric_name == "core_search_ms":
                    print(f"      🔍 Основной поиск: {value:.1f}ms")
                elif metric_name == "post_search_processing_ms":
                    print(f"      ⚙️  Пост-обработка: {value:.1f}ms")
                elif metric_name == "answer_generation_ms":
                    print(f"      💬 Генерация ответа: {value:.1f}ms")
                else:
                    print(f"      📊 {metric_name}: {value:.1f}ms")
            
            # Добавляем детальную разбивку времени
            print(f"\n   🔍 ДЕТАЛЬНАЯ РАЗБИВКА ВРЕМЕНИ:")
            
            # Определение намерения
            detect_intent_time = session.performance_metrics.get("detect_intent_ms", 0)
            if detect_intent_time > 0:
                print(f"      🧠 Определение намерения: {detect_intent_time:.1f}ms")
            
            # Маршрутизация поиска
            route_search_time = session.performance_metrics.get("route_search_ms", 0)
            core_search_time = session.performance_metrics.get("core_search_ms", 0)
            post_processing_time = session.performance_metrics.get("post_search_processing_ms", 0)
            
            if route_search_time > 0:
                print(f"      🔍 Маршрутизация поиска (общее): {route_search_time:.1f}ms")
                if core_search_time > 0:
                    print(f"         ├─ Основной поиск: {core_search_time:.1f}ms")
                    if session.search_metrics:
                        bm25_time = session.search_metrics.get("bm25_search_ms", 0)
                        vector_books_time = session.search_metrics.get("vector_books_search_ms", 0)
                        vector_content_time = session.search_metrics.get("vector_content_search_ms", 0)
                        filtering_time = session.search_metrics.get("results_filtering_ms", 0)
                        init_time = session.search_metrics.get("initialization_ms", 0)
                        
                        if init_time > 0:
                            print(f"         │  ├─ Инициализация: {init_time:.1f}ms")
                        if bm25_time > 0:
                            print(f"         │  ├─ BM25 поиск: {bm25_time:.1f}ms")
                        if vector_books_time > 0:
                            print(f"         │  ├─ Vector books: {vector_books_time:.1f}ms")
                        if vector_content_time > 0:
                            print(f"         │  ├─ Vector content: {vector_content_time:.1f}ms")
                        if filtering_time > 0:
                            print(f"         │  └─ Фильтрация: {filtering_time:.1f}ms")
                    
                    # Проверяем разницу между core_search и total_search_time
                    total_search_time = session.search_metrics.get("total_search_time_ms", 0)
                    if total_search_time > 0 and abs(core_search_time - total_search_time) > 100:
                        overhead_time = core_search_time - total_search_time
                        print(f"         │  └─ Накладные расходы: {overhead_time:.1f}ms")
                
                if post_processing_time > 0:
                    print(f"         └─ Пост-обработка: {post_processing_time:.1f}ms")
            
            # Генерация ответа
            answer_time = session.performance_metrics.get("answer_generation_ms", 0)
            if answer_time > 0:
                print(f"      💬 Генерация ответа: {answer_time:.1f}ms")
            
            # Проверяем общую сумму
            total_measured = detect_intent_time + route_search_time + answer_time
            if total_measured > 0 and abs(session.execution_time_ms - total_measured) > 100:
                unaccounted_time = session.execution_time_ms - total_measured
                print(f"\n   ⚠️  НЕУЧТЕННОЕ ВРЕМЯ: {unaccounted_time:.1f}ms")
                print(f"      (возможно, время инициализации, сетевые задержки, etc.)")
        
        # ============= ФИНАЛЬНЫЙ РЕЗУЛЬТАТ =============
        results_icon = "✅" if session.final_results_count > 0 else "❌"
        print(f"\n🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {results_icon} {session.final_results_count} книг найдено")
        
        # Показать все найденные книги
        if session.final_books:
            print(f"\n📚 НАЙДЕННЫЕ КНИГИ:")
            for i, book in enumerate(session.final_books, 1):
                title = book.get('title', 'Без названия')
                author = book.get('author', 'Неизвестный автор')
                year = book.get('year', '')
                year_str = f" ({year})" if year else ""
                print(f"  {i}. \"{title}\" by {author}{year_str}")
                
                # Показать краткое описание если есть
                summary = book.get('summary', '')
                if summary:
                    summary_preview = summary[:80] + "..." if len(summary) > 80 else summary
                    print(f"     💡 {summary_preview}")
        
        if session.final_response:
            response_preview = session.final_response[:100] + "..." if len(session.final_response) > 100 else session.final_response
            print(f"\n   💬 Ответ пользователю: '{response_preview}'")
        
        # ============= ПРОБЛЕМЫ =============
        if session.issues:
            print(f"\n⚠️  ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ:")
            for i, issue in enumerate(session.issues, 1):
                print(f"  {i}. {issue}")
        
        # ============= LLM АНАЛИЗ =============
        if session.llm_analysis:
            print(f"\n🤖 АНАЛИЗ ИИ: {session.llm_analysis}")
        
        print(f"{'='*100}\n")

# Global instance
debug_reporter = DebugReporter()

# Convenience functions
def start_debug(query: str, session_id: str = "default"):
    """Start debug session"""
    debug_reporter.start_session(query, session_id)
    
def set_debug_intent(intent: str, filters: Dict[str, Any], confidence: str = "", 
                    method: str = "", details: Dict[str, Any] = None):
    """Record intent detection"""
    debug_reporter.set_intent(intent, filters, confidence, method, details)
    
def add_debug_step(name: str, query: str, success: bool, results_count: int, reason: str = "", **details):
    """Add search step"""
    debug_reporter.add_search_step(name, query, success, results_count, reason, **details)
    
def add_debug_issue(issue: str):
    """Add debug issue"""
    debug_reporter.add_issue(issue)
    
# Новые convenience functions
def set_debug_query_processing(simple_query: bool = False, mixed_filters: bool = False, predicate_filter: bool = False):
    """Записать флаги обработки запроса"""
    debug_reporter.set_query_processing(simple_query, mixed_filters, predicate_filter)
    
def set_debug_search_queries(bm25_query: str = "", vector_query: str = ""):
    """Записать поисковые запросы"""
    debug_reporter.set_search_queries(bm25_query, vector_query)
    
def add_debug_filtered_result(result_info: Dict[str, Any]):
    """Записать решение о фильтрации"""
    debug_reporter.add_filtered_result(result_info)
    
def set_debug_final_response(response: str):
    """Записать финальный ответ пользователю"""
    debug_reporter.set_final_response(response)

def set_debug_final_books(books: List[Dict[str, Any]]):
    """Записать финальные книги"""
    debug_reporter.set_final_books(books)

def finalize_debug(final_results_count: int, execution_time_ms: float = 0.0):
    """Finalize debug session"""
    debug_reporter.finalize_session(final_results_count, execution_time_ms)

def set_performance_metrics(metrics: Dict[str, float]):
    """Record performance metrics"""
    debug_reporter.set_performance_metrics(metrics)

def set_search_metrics(metrics: Dict[str, float]):
    """Record search metrics"""
    debug_reporter.set_search_metrics(metrics)