# 🔍 Debug Reporter System

Компактная система анализа выполнения поисковых запросов с AI-анализом.

## ✨ Что делает система

Показывает **компактный отчет** после каждого запроса:
- 🧠 **Intent Detection**: какой intent определился и почему
- 🔎 **Search Pipeline**: каждый шаг поиска (BM25, vector, фильтры)
- 📊 **Results**: количество результатов на каждом этапе
- ⚠️ **Issues**: конфликты, ошибки, проблемы
- 🤖 **AI Analysis**: умные инсайты от LLM (опционально)

## 🚀 Быстрый старт

1. **Включить систему**:
   ```bash
   # В .env файле:
   DEBUG_REPORTER_ENABLED=true
   DEBUG_LLM_ANALYSIS=true  # Опционально для AI анализа
   ```

2. **Перезапустить приложение**:
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Тестировать**:
   ```bash
   python test_debug.py
   ```

## 📊 Пример отчета

```
================================================================================
🔍 DEBUG REPORT | ✅ SUCCESS | 245ms
================================================================================
📝 Query: 'Animal Farm by George Orwell'
🧠 Intent: author_title high
🎯 Filters: title:animal farm | author:george orwell

🔎 SEARCH PIPELINE:
  1. ✅ Hybrid Search (BM25 + Vector): (3 results) | query='Animal Farm by George Orwell'
     📊 search_type=hybrid | intent=author_title
  2. ✅ Predicate Filter: (2 results) | filter by title, author
  3. ✅ Author+Title Fuzzy Match: (1 results) | query='fuzzy validation'

🎯 Final Results: ✅ 1 books found

⚠️  Issues:
   - Title similarity only 85%, consider improving matching

🤖 Analysis: ✅ Perfect author+title detection. Hybrid search found relevant 
matches, predicate filter worked correctly. Minor title matching could be 
improved but overall excellent execution. 🎯
================================================================================
```

## ⚙️ Конфигурация

| Переменная | Значения | Описание |
|-----------|----------|----------|
| `DEBUG_REPORTER_ENABLED` | `true`/`false` | Включить/выключить всю систему |
| `DEBUG_LLM_ANALYSIS` | `true`/`false` | AI анализ результатов |

## 🎯 Что отслеживается

- **Intent Detection**: Mixed filters → author_title downgrade
- **BM25 Search**: Keyword matching, результаты
- **Vector Search**: Semantic search, сходство
- **Predicate Filters**: ISBN, язык, год, исключения
- **Fuzzy Matching**: Author-title валидация
- **Conflicts**: ISBN vs author, title mismatches
- **Performance**: Время выполнения, количество результатов

## 🔧 Интеграция

Система автоматически встроена в `orchestrator.py`:

```python
# Автоматически начинается в node_detect_intent
start_debug(state.message, state.session_id)

# Записывается intent
set_debug_intent(state.intent, state.filters, confidence)

# Отслеживаются шаги поиска
add_debug_step("Hybrid Search", query, success, results_count)

# Записываются проблемы
add_debug_issue("Author conflict detected")

# Финализируется в node_answer
finalize_debug(final_results_count, execution_time)
```

## 🚫 Отключение для продакшн

**Способ 1**: Убрать переменную
```bash
# Удалить из .env:
# DEBUG_REPORTER_ENABLED=true
```

**Способ 2**: Явно отключить
```bash
# В .env:
DEBUG_REPORTER_ENABLED=false
```

**Нулевые накладные расходы** когда отключено - все функции сразу возвращают `return` если `enabled=false`.

## 🧪 Тестирование

```bash
# Базовое тестирование
python test_debug.py

# С AI анализом
python test_debug_llm.py

# Ручное тестирование
DEBUG_REPORTER_ENABLED=true python -c "
from app.orchestrator import ChatState, graph
result = graph.invoke(ChatState(session_id='test', message='leadership books'))
"
```

## 🔍 Troubleshooting

**Отчеты не появляются?**
- Проверить `DEBUG_REPORTER_ENABLED=true` в .env
- Перезапустить приложение
- Проверить что запросы идут через orchestrator.py

**AI анализ не работает?**
- Проверить `OPENAI_API_KEY` в .env
- Установить `DEBUG_LLM_ANALYSIS=true`
- Проверить логи на API ошибки

**Система влияет на производительность?**
- При `DEBUG_REPORTER_ENABLED=false` - нет влияния
- При включенной системе - минимальное влияние (~1-5ms)
- LLM анализ добавляет ~200-500ms (но не блокирует основной поток)

---

**🎯 Цель**: Быстро видеть что происходит в поисковой системе и сразу понимать где проблемы!