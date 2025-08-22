from typing import Optional, Dict, Any, List
import logging
import time
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY
from .simple_retriever import SimpleVectorRetriever
from .simple_llm_filter import SimpleLLMFilter

logger = logging.getLogger(__name__)

class SimpleSearchState(BaseModel):
    """Состояние для простого поиска без сложной логики"""
    session_id: str
    message: str
    # Результаты поиска
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    # Результаты после LLM фильтрации
    filtered_results: List[Dict[str, Any]] = Field(default_factory=list)
    # Анализ от LLM
    intent: Optional[str] = None
    analysis: Optional[str] = None
    negative_filters: Optional[str] = None
    uncertainty_note: Optional[str] = None
    # Финальный ответ
    final_response: Optional[str] = None
    # Метрики производительности
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    # Ошибки
    error: Optional[str] = None

# Инициализация компонентов
simple_retriever = SimpleVectorRetriever()
llm_filter = SimpleLLMFilter()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

def search_step(state: SimpleSearchState) -> SimpleSearchState:
    """Шаг 1: Простой векторный поиск"""
    logger.info(f"🔍 [Step 1] Простой поиск для: '{state.message}'")
    
    start_time = time.time()
    
    try:
        # Выполняем асинхронный поиск с помощью concurrent.futures
        import concurrent.futures
        import asyncio
        
        def run_search():
            # Создаем новый event loop в отдельном потоке
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(simple_retriever.search(state.message, k=5))
            finally:
                loop.close()
        
        # Выполняем в отдельном потоке
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_search)
            search_results = future.result()
        
        state.search_results = search_results
        state.performance_metrics['search_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 1] Найдено {len(search_results)} результатов за {state.performance_metrics['search_time']:.2f}с")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 1] Ошибка поиска: {e}")
        state.error = f"Ошибка поиска: {e}"
        state.performance_metrics['search_time'] = time.time() - start_time
        return state

def filter_step(state: SimpleSearchState) -> SimpleSearchState:
    """Шаг 2: LLM фильтрация и анализ"""
    logger.info(f"🧠 [Step 2] LLM фильтрация {len(state.search_results)} результатов")
    
    start_time = time.time()
    
    try:
        # Если есть ошибка на предыдущем шаге, пропускаем
        if state.error:
            logger.warning("⚠️ [Step 2] Пропускаем из-за ошибки на предыдущем шаге")
            return state
            
        # Если нет результатов поиска, пропускаем фильтрацию
        if not state.search_results:
            logger.info("ℹ️ [Step 2] Нет результатов для фильтрации")
            state.filtered_results = []
            state.intent = "no_results"
            state.analysis = "Результаты поиска не найдены"
            state.performance_metrics['filter_time'] = time.time() - start_time
            return state
        
        # Выполняем LLM фильтрацию
        import concurrent.futures
        import asyncio
        
        def run_filter():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(llm_filter.filter_and_analyze(state.message, state.search_results))
            finally:
                loop.close()
        
        # Выполняем в отдельном потоке
        llm_filter_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_filter)
            filter_result = future.result()
        llm_filter_time = time.time() - llm_filter_start
        logger.info(f"⚡ LLM filter за {llm_filter_time:.3f}с")
        
        # Обновляем состояние
        state.filtered_results = filter_result['filtered_results']
        state.intent = filter_result['intent']
        state.analysis = filter_result['analysis']
        state.negative_filters = filter_result.get('negative_filters', '')
        state.uncertainty_note = filter_result.get('uncertainty_note', '')
        state.performance_metrics['filter_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 2] LLM анализ: intent='{state.intent}', "
                   f"отфильтровано {len(state.filtered_results)}/{len(state.search_results)} за {state.performance_metrics['filter_time']:.2f}с")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 2] Ошибка LLM фильтрации: {e}")
        # В случае ошибки возвращаем все результаты
        state.filtered_results = state.search_results
        state.intent = "filter_error"
        state.analysis = f"Ошибка фильтрации: {e}"
        state.performance_metrics['filter_time'] = time.time() - start_time
        return state

def format_step(state: SimpleSearchState) -> SimpleSearchState:
    """Шаг 3: Форматирование финального ответа"""
    logger.info(f"💬 [Step 3] Форматирование ответа для {len(state.filtered_results)} результатов")
    
    start_time = time.time()
    
    try:
        # Если есть критическая ошибка, возвращаем ошибку
        if state.error and not state.filtered_results:
            state.final_response = f"Извините, произошла ошибка при поиске: {state.error}"
            state.performance_metrics['format_time'] = time.time() - start_time
            return state
        
        # Если нет результатов
        if not state.filtered_results:
            state.final_response = _format_no_results_response(state.message)
            state.performance_metrics['format_time'] = time.time() - start_time
            return state
        
        # Генерируем ответ с помощью LLM
        import concurrent.futures
        import asyncio
        
        def run_format():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(_generate_formatted_response(
                    state.message, 
                    state.filtered_results, 
                    state.intent,
                    state.analysis,
                    state.uncertainty_note
                ))
            finally:
                loop.close()
        
        # Выполняем в отдельном потоке
        llm_format_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_format)
            formatted_response = future.result()
        llm_format_time = time.time() - llm_format_start
        logger.info(f"⚡ LLM format за {llm_format_time:.3f}с")
        
        state.final_response = formatted_response
        state.performance_metrics['format_time'] = time.time() - start_time
        
        logger.info(f"✅ [Step 3] Ответ сформирован за {state.performance_metrics['format_time']:.2f}с")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ [Step 3] Ошибка форматирования: {e}")
        # Fallback к простому форматированию
        state.final_response = _format_simple_response(state.filtered_results)
        state.performance_metrics['format_time'] = time.time() - start_time
        return state

async def _generate_formatted_response(query: str, results: List[Dict], intent: str, analysis: str, uncertainty_note: str = "") -> str:
    """Генерирует форматированный ответ с помощью LLM"""
    
    # Минимальная информация для быстрого форматирования с content
    results_info = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Unknown')
        author = result.get('author', 'Unknown')
        content = result.get('content', '')
        
        result_text = f"""{i}. "{title}" - {author}"""
        
        # Добавляем только год если есть
        metadata = result.get('metadata', {})
        year = metadata.get('year', '')
        if year:
            result_text += f" ({year})"
        
        # Добавляем полный content 
        if content:
            result_text += f"\n   {content}"
        
        results_info.append(result_text)
    
    results_text = "\n\n".join(results_info)
    
    # Краткий мотивирующий промпт для чтения
    system_prompt = """Ты опытный библиотекарь. Создай краткий и мотивирующий ответ на русском языке.

ВАЖНО: Ответ должен быть коротким (200-300 символов)!

Для каждой книги напиши:
1. "📖 «Название» — Автор (год)"
2. Краткое описание сюжета (1-2 предложения)
3. Почему стоит прочесть

Будь лаконичным, но вдохновляющим!"""

    user_prompt = f"""Пользователь ищет: "{query}"

Найденные книги:
{results_text}

Создай вдохновляющий ответ, который мотивирует к чтению!"""

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        logger.info(f"📝 LLM сгенерированный ответ: {response.content[:200]}...")
        return response.content
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации ответа: {e}")
        logger.info("🔄 Используем fallback форматирование")
        return _format_simple_response(results)

def _format_simple_response(results: List[Dict]) -> str:
    """Простое форматирование без LLM в случае ошибки"""
    if not results:
        return "К сожалению, подходящих книг не найдено."
    
    lines = [f"Найдено {len(results)} книг(и):"]
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'Unknown')
        author = result.get('author', 'Unknown')
        
        line = f"{i}. \"{title}\" - {author}"
        
        # Добавляем год если есть
        metadata = result.get('metadata', {})
        year = metadata.get('year', '')
        if year:
            line += f" ({year})"
            
        lines.append(line)
    
    return "\n".join(lines)

def _format_no_results_response(query: str) -> str:
    """Форматирует ответ когда результатов нет"""
    return f"""К сожалению, не удалось найти книги по запросу "{query}".

Попробуйте:
- Проверить правильность написания названия или автора
- Использовать более общие термины
- Попробовать поиск по жанру или теме

Могу помочь с другим запросом!"""

# Создание графа
def create_simple_search_graph():
    """Создает граф для простого поиска"""
    
    workflow = StateGraph(SimpleSearchState)
    
    # Добавляем узлы
    workflow.add_node("search", search_step)
    workflow.add_node("filter", filter_step) 
    workflow.add_node("format", format_step)
    
    # Определяем поток
    workflow.set_entry_point("search")
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()

# Создаем граф
simple_search_graph = create_simple_search_graph()

async def process_simple_search(session_id: str, message: str) -> Dict[str, Any]:
    """
    Основная функция для обработки простого поиска
    
    Args:
        session_id: идентификатор сессии
        message: сообщение пользователя
        
    Returns:
        Dict с результатами поиска
    """
    logger.info(f"🚀 [Simple Search] Начинаем обработку: session_id={session_id}, query='{message}'")
    
    total_start_time = time.time()
    
    try:
        # Создаем начальное состояние
        initial_state = SimpleSearchState(
            session_id=session_id,
            message=message
        )
        
        # Выполняем граф
        final_state = simple_search_graph.invoke(initial_state)
        
        # Общее время
        total_time = time.time() - total_start_time
        
        # Обрабатываем результат (может быть dict или объект)
        if hasattr(final_state, 'performance_metrics'):
            final_state.performance_metrics['total_time'] = total_time
            performance_metrics = final_state.performance_metrics
            search_results = final_state.search_results
            filtered_results = final_state.filtered_results
            final_response = final_state.final_response
            intent = final_state.intent
            analysis = final_state.analysis
            uncertainty_note = getattr(final_state, 'uncertainty_note', '')
        else:
            # Если final_state это dict
            performance_metrics = final_state.get('performance_metrics', {})
            performance_metrics['total_time'] = total_time
            search_results = final_state.get('search_results', [])
            filtered_results = final_state.get('filtered_results', [])
            final_response = final_state.get('final_response', 'Ошибка обработки')
            intent = final_state.get('intent', 'error')
            analysis = final_state.get('analysis', 'Анализ недоступен')
            uncertainty_note = final_state.get('uncertainty_note', '')
        
        # Формируем результат
        result = {
            "response": final_response,
            "results": [
                {
                    "title": r.get('title', 'Unknown'),
                    "author": r.get('author', 'Unknown'),
                    "score": r.get('score', 0.0),
                    "collection": r.get('collection', 'unknown')
                }
                for r in filtered_results
            ],
            "intent": intent,
            "analysis": analysis,
            "uncertainty_note": uncertainty_note,
            "performance_metrics": performance_metrics,
            "search_stats": {
                "total_found": len(search_results),
                "after_filter": len(filtered_results),
                "total_time": total_time
            }
        }
        
        logger.info(f"🎯 [Simple Search] Завершено за {total_time:.2f}с: "
                   f"найдено {len(search_results)}, отфильтровано {len(filtered_results)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ [Simple Search] Критическая ошибка: {e}")
        
        return {
            "response": f"Извините, произошла ошибка при обработке запроса: {e}",
            "results": [],
            "intent": "error",
            "analysis": f"Критическая ошибка: {e}",
            "performance_metrics": {"total_time": time.time() - total_start_time},
            "search_stats": {
                "total_found": 0,
                "after_filter": 0,
                "total_time": time.time() - total_start_time
            }
        }