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
            self.current_session.llm_analysis = "❌ Analysis failed"
            
    def _print_compact_report(self) -> None:
        """Print extended debug report to terminal"""
        if not self.current_session:
            return
            
        session = self.current_session
        
        # Execution status
        status = "✅ SUCCESS" if session.success else "❌ ERROR"
        
        print(f"\n{'='*100}")
        print(f"🔍 DEBUG REPORT | {status} | {session.execution_time_ms:.0f}ms")
        print(f"{'='*100}")
        
        # User query
        print(f"📝 User query: '{session.original_query}'")
        
        # ============= INTENT ANALYSIS =============
        print(f"\n🧠 INTENT ANALYSIS:")
        print(f"   💡 Detected intent: {session.intent} ({session.intent_confidence} confidence)")
        print(f"   🔧 Detection method: {session.intent_method or 'not specified'}")
        
        if session.intent_details:
            print(f"   📋 Analysis details:")
            for key, value in session.intent_details.items():
                print(f"      • {key}: {value}")
        
        # Query processing flags
        print(f"   🔍 Simple query: {'Yes' if session.simple_query_check else 'No'}")
        print(f"   ⚙️  Mixed filters: {'Used' if session.mixed_filters_used else 'Not used'}")
        print(f"   🎯 Predicate filter: {'Applied' if session.predicate_filter_used else 'Not applied'}")
        
        # Extracted filters
        if session.filters:
            print(f"\n🎯 EXTRACTED FILTERS:")
            for key, value in session.filters.items():
                if value and value != [] and value != {}:
                    if isinstance(value, list):
                        print(f"   • {key}: [{', '.join(map(str, value))}]")
                    else:
                        print(f"   • {key}: {value}")
        
        # ============= SEARCH QUERIES =============
        print(f"\n🔎 SEARCH QUERIES:")
        if session.bm25_query:
            print(f"   📊 BM25 query: '{session.bm25_query}'")
        if session.vector_query:
            print(f"   🧮 Vector query: '{session.vector_query}'")
        if not session.bm25_query and not session.vector_query:
            print(f"   ❓ Search queries not recorded")
        
        # ============= SEARCH PIPELINE =============
        print(f"\n🔎 SEARCH PIPELINE:")
        if not session.steps:
            print(f"   ❓ Search steps not recorded")
        else:
            for i, step in enumerate(session.steps, 1):
                icon = "✅" if step.success else "❌"
                results_info = f"({step.results_count} results)" if step.success else f"({step.reason})"
                query_display = step.query[:50] + "..." if len(step.query) > 50 else step.query
                print(f"  {i}. {icon} {step.name}: {results_info}")
                print(f"     🔍 Query: '{query_display}'")
                
                # Detailed information if available
                if step.details:
                    for key, value in step.details.items():
                        if key == 'similarity_score':
                            print(f"     📈 Similarity: {value}")
                        elif key == 'threshold':
                            print(f"     🎯 Threshold: {value}")
                        elif key == 'boost_score':
                            print(f"     ⚡ Boost: {value}")
                        elif key == 'search_type':
                            print(f"     🔧 Search type: {value}")
                        elif key == 'intent':
                            print(f"     🧠 Intent: {value}")
                        else:
                            print(f"     📋 {key}: {value}")
        
        # ============= RESULTS FILTERING =============
        if session.filtered_results:
            print(f"\n🎛️  RESULTS FILTERING:")
            for i, result in enumerate(session.filtered_results, 1):
                action = result.get('action', 'unknown')
                reason = result.get('reason', 'reason not specified')
                book_info = result.get('book_info', {})
                title = book_info.get('title', 'Untitled')
                author = book_info.get('author', 'Unknown author')
                score = result.get('score', 'N/A')
                print(f"  {i}. {action}: '{title}' by {author} (score: {score})")
                print(f"     🔍 Reason: {reason}")
        
        # ============= PERFORMANCE METRICS =============
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"   🕐 Total execution time: {session.execution_time_ms:.1f}ms")
        
        # Detailed search metrics (if available)
        if session.search_metrics:
            print(f"   📊 Detailed search metrics:")
            for metric_name, value in session.search_metrics.items():
                if metric_name == "bm25_search_ms":
                    print(f"      🔤 BM25 search: {value:.1f}ms")
                elif metric_name == "vector_books_search_ms":
                    print(f"      📚 Vector books search: {value:.1f}ms")
                elif metric_name == "vector_content_search_ms":
                    print(f"      📄 Vector content search: {value:.1f}ms")
                elif metric_name == "results_filtering_ms":
                    print(f"      🎛️  Filtering results: {value:.1f}ms")
                elif metric_name == "initialization_ms":
                    print(f"      ⚙️  Initialization: {value:.1f}ms")
                elif metric_name == "total_search_time_ms":
                    print(f"      🔍 Total search time: {value:.1f}ms")
                else:
                    print(f"      📊 {metric_name}: {value:.1f}ms")
        
        # Stage performance metrics (if available)
        if session.performance_metrics:
            print(f"   📈 Stage performance metrics:")
            for metric_name, value in session.performance_metrics.items():
                if metric_name == "detect_intent_ms":
                    print(f"      🧠 Intent detection: {value:.1f}ms")
                elif metric_name == "route_search_ms":
                    print(f"      🔍 Search routing: {value:.1f}ms")
                elif metric_name == "core_search_ms":
                    print(f"      🔍 Core search: {value:.1f}ms")
                elif metric_name == "post_search_processing_ms":
                    print(f"      ⚙️  Post-processing: {value:.1f}ms")
                elif metric_name == "answer_generation_ms":
                    print(f"      💬 Answer generation: {value:.1f}ms")
                else:
                    print(f"      📊 {metric_name}: {value:.1f}ms")
            
            # Add detailed time breakdown
            print(f"\n   🔍 DETAILED TIME BREAKDOWN:")
            
            # Intent detection
            detect_intent_time = session.performance_metrics.get("detect_intent_ms", 0)
            if detect_intent_time > 0:
                print(f"      🧠 Intent detection: {detect_intent_time:.1f}ms")
            
            # Search routing
            route_search_time = session.performance_metrics.get("route_search_ms", 0)
            core_search_time = session.performance_metrics.get("core_search_ms", 0)
            post_processing_time = session.performance_metrics.get("post_search_processing_ms", 0)
            
            if route_search_time > 0:
                print(f"      🔍 Search routing (total): {route_search_time:.1f}ms")
                if core_search_time > 0:
                    print(f"         ├─ Core search: {core_search_time:.1f}ms")
                    if session.search_metrics:
                        bm25_time = session.search_metrics.get("bm25_search_ms", 0)
                        vector_books_time = session.search_metrics.get("vector_books_search_ms", 0)
                        vector_content_time = session.search_metrics.get("vector_content_search_ms", 0)
                        filtering_time = session.search_metrics.get("results_filtering_ms", 0)
                        init_time = session.search_metrics.get("initialization_ms", 0)
                        
                        if init_time > 0:
                            print(f"         │  ├─ Initialization: {init_time:.1f}ms")
                        if bm25_time > 0:
                            print(f"         │  ├─ BM25 search: {bm25_time:.1f}ms")
                        if vector_books_time > 0:
                            print(f"         │  ├─ Vector books: {vector_books_time:.1f}ms")
                        if vector_content_time > 0:
                            print(f"         │  ├─ Vector content: {vector_content_time:.1f}ms")
                        if filtering_time > 0:
                            print(f"         │  └─ Filtering: {filtering_time:.1f}ms")
                    
                    # Check difference between core_search and total_search_time
                    total_search_time = session.search_metrics.get("total_search_time_ms", 0)
                    if total_search_time > 0 and abs(core_search_time - total_search_time) > 100:
                        overhead_time = core_search_time - total_search_time
                        print(f"         │  └─ Overhead: {overhead_time:.1f}ms")
                
                if post_processing_time > 0:
                    print(f"         └─ Post-processing: {post_processing_time:.1f}ms")
            
            # Answer generation
            answer_time = session.performance_metrics.get("answer_generation_ms", 0)
            if answer_time > 0:
                print(f"      💬 Answer generation: {answer_time:.1f}ms")
            
            # Check total sum
            total_measured = detect_intent_time + route_search_time + answer_time
            if total_measured > 0 and abs(session.execution_time_ms - total_measured) > 100:
                unaccounted_time = session.execution_time_ms - total_measured
                print(f"\n   ⚠️  UNACCOUNTED TIME: {unaccounted_time:.1f}ms")
                print(f"      (possibly initialization time, network delays, etc.)")
        
        # ============= FINAL RESULT =============
        results_icon = "✅" if session.final_results_count > 0 else "❌"
        print(f"\n🎯 FINAL RESULT: {results_icon} {session.final_results_count} books found")
        
        # Show all found books
        if session.final_books:
            print(f"\n📚 FOUND BOOKS:")
            for i, book in enumerate(session.final_books, 1):
                title = book.get('title', 'Untitled')
                author = book.get('author', 'Unknown author')
                year = book.get('year', '')
                year_str = f" ({year})" if year else ""
                print(f"  {i}. \"{title}\" by {author}{year_str}")
                
                # Show brief description if available
                summary = book.get('summary', '')
                if summary:
                    summary_preview = summary[:80] + "..." if len(summary) > 80 else summary
                    print(f"     💡 {summary_preview}")
        
        if session.final_response:
            response_preview = session.final_response[:100] + "..." if len(session.final_response) > 100 else session.final_response
            print(f"\n   💬 User response: '{response_preview}'")
        
        # ============= ISSUES =============
        if session.issues:
            print(f"\n⚠️  DETECTED ISSUES:")
            for i, issue in enumerate(session.issues, 1):
                print(f"  {i}. {issue}")
        
        # ============= LLM ANALYSIS =============
        if session.llm_analysis:
            print(f"\n🤖 AI ANALYSIS: {session.llm_analysis}")
        
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
    
# New convenience functions
def set_debug_query_processing(simple_query: bool = False, mixed_filters: bool = False, predicate_filter: bool = False):
    """Record query processing flags"""
    debug_reporter.set_query_processing(simple_query, mixed_filters, predicate_filter)
    
def set_debug_search_queries(bm25_query: str = "", vector_query: str = ""):
    """Record search queries"""
    debug_reporter.set_search_queries(bm25_query, vector_query)
    
def add_debug_filtered_result(result_info: Dict[str, Any]):
    """Record filtering decision"""
    debug_reporter.add_filtered_result(result_info)
    
def set_debug_final_response(response: str):
    """Record final response to user"""
    debug_reporter.set_final_response(response)

def set_debug_final_books(books: List[Dict[str, Any]]):
    """Record final books"""
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