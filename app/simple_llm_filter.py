from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from .settings import OPENAI_MODEL_CHAT, OPENAI_API_KEY

logger = logging.getLogger(__name__)

class SimpleLLMFilter:
    """LLM-фильтр с системным промптом для анализа и фильтрации результатов поиска"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL_CHAT,
            temperature=0,  # Детерминированный результат
            api_key=OPENAI_API_KEY
        )
        logger.info("🧠 SimpleLLMFilter initialized")
    
    async def filter_and_analyze(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует запрос пользователя и фильтрует результаты поиска
        
        Args:
            query: оригинальный запрос пользователя
            search_results: результаты векторного поиска
            
        Returns:
            Dict с отфильтрованными результатами и метаданными анализа
        """
        logger.info(f"🧠 LLM анализ запроса: '{query}' для {len(search_results)} результатов")
        
        if not search_results:
            return {
                "filtered_results": [],
                "intent": "no_results",
                "analysis": "Нет результатов для анализа",
                "total_found": 0,
                "total_filtered": 0
            }
        
        try:
            # Подготавливаем данные для LLM
            results_summary = self._prepare_results_for_llm(search_results)
            
            # Создаем промпт
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, results_summary)
            
            # Вызываем LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Парсим ответ LLM
            analysis_result = self._parse_llm_response(response.content, search_results)
            
            logger.info(f"✅ LLM анализ: intent='{analysis_result['intent']}', "
                       f"filtered={analysis_result['total_filtered']}/{analysis_result['total_found']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Ошибка в LLM фильтре: {e}")
            # Возвращаем все результаты в случае ошибки
            return {
                "filtered_results": search_results,
                "intent": "error",
                "analysis": f"Ошибка анализа: {e}",
                "total_found": len(search_results),
                "total_filtered": len(search_results)
            }
    
    def _create_system_prompt(self) -> str:
        """Создает простой системный промпт для фильтрации"""
        return """Ты фильтр результатов поиска книг. Твоя задача - выбрать из предоставленного списка книги, которые соответствуют запросу пользователя.

ПРАВИЛА:
- Учитывай автора, название, содержание книг
- При сомнениях лучше включить, чем исключить
- Учитывай синонимы и варианты написания (Орwell/Оруэлл)
- Обрабатывай исключения: "детективы но не Агата Кристи"

ФОРМАТ ОТВЕТА:
{
  "filtered_indices": [список индексов подходящих результатов, например: [0, 2, 4]],
  "note": "краткое объяснение выбора"
}"""

    def _create_user_prompt(self, query: str, results_summary: str) -> str:
        """Создает пользовательский промпт с запросом и результатами"""
        return f"""Запрос: "{query}"

Результаты:
{results_summary}

Выбери подходящие книги."""

    def _prepare_results_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
        """Подготавливает результаты поиска для передачи в LLM"""
        summary_lines = []
        
        for i, result in enumerate(search_results):
            title = result.get('title', 'Unknown')
            author = result.get('author', 'Unknown')
            
            # Краткое содержание
            content = result.get('content', '')
            content_preview = content[:200] + '...' if len(content) > 200 else content
            
            # Основная информация с content preview
            summary_line = f"""[{i}] "{title}" by {author}"""
            if content_preview:
                summary_line += f"\n   Content preview: {content_preview}"
            
            summary_lines.append(summary_line)
        
        return "\n".join(summary_lines)

    def _parse_llm_response(self, llm_response: str, original_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Парсит ответ LLM и возвращает отфильтрованные результаты"""
        logger.info(f"🔍 Парсинг ответа LLM: {llm_response[:200]}...")
        
        try:
            import json
            
            # Очищаем ответ от возможных markdown блоков
            clean_response = llm_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            # Парсим JSON
            parsed = json.loads(clean_response)
            
            filtered_indices = parsed.get('filtered_indices', [])
            note = parsed.get('note', 'Фильтрация выполнена')
            
            # Фильтруем результаты по индексам
            filtered_results = []
            for idx in filtered_indices:
                if 0 <= idx < len(original_results):
                    filtered_results.append(original_results[idx])
                else:
                    logger.warning(f"⚠️ Невалидный индекс в filtered_indices: {idx}")
            
            return {
                "filtered_results": filtered_results,
                "intent": "filtered",
                "analysis": note,
                "negative_filters": "",
                "uncertainty_note": "",
                "total_found": len(original_results),
                "total_filtered": len(filtered_results)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON ответа LLM: {e}")
            logger.error(f"Ответ LLM: {llm_response}")
            
            # Fallback: возвращаем все результаты
            return {
                "filtered_results": original_results,
                "intent": "parse_error",
                "analysis": f"Не удалось распарсить ответ LLM: {e}",
                "negative_filters": "",
                "uncertainty_note": "",
                "total_found": len(original_results),
                "total_filtered": len(original_results)
            }
        
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при парсинге ответа LLM: {e}")
            
            return {
                "filtered_results": original_results,
                "intent": "error", 
                "analysis": f"Ошибка обработки: {e}",
                "negative_filters": "",
                "uncertainty_note": "",
                "total_found": len(original_results),
                "total_filtered": len(original_results)
            }