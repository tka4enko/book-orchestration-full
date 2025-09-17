# Agent Flow & Prompt Usage

Документ фиксирует текущую реализацию узлов в `orchestrator_chat_agent.py`, а также
используемые LLM-промпты. Структура отражает договорённости из обсуждения.

## 1. Intent Detection (`app/smart_intent_system.py`)

1. Узел `intent` получает текущее сообщение и контекст (`build_intent_context`).
2. Модуль `smart_intent_system` вызывает ChatOpenAI с промптом `ENHANCED_INTENT_PROMPT`.
   - Промпт жёстко задаёт валидные интенты: `CHAT`, `RECOMMEND`, `SEARCH`,
     `ANALYTICS`, `CLARIFY`.
   - В правилах описаны ключевые сценарии (подтверждение после рекомендаций →
     `SEARCH`, низкая уверенность → `CLARIFY`).
3. Модель отвечает JSON-структурой, откуда берём намерение, уверенность, причину.
4. Если `confidence < 0.6`, узел принудительно переключается в `CLARIFY`.

## 2. Simple Search (`app/smart_simple_orchestrator.py`)

1. Запуск только при явном интенте `SEARCH` или когда пользователь текстом
   подтверждает одну из рекомендаций.
2. Этапы:
   - `SimpleVectorRetriever.search` — векторный поиск по коллекциям.
   - `SimpleLLMFilter.filter_and_analyze` — фильтрация и короткий анализ.
   - Форматирование ответа через ChatOpenAI (промпт в функции `_format_response`).
3. Результат содержит:
   - `response` — текстовое сообщение для пользователя.
   - `results` — массив карточек (title, author, score, metadata).
   - `performance_metrics` — времена этапов.
4. Если результатов нет — возвращается вежливое сообщение "не удалось найти...".
   Дополнительные узлы не запускаются.

## 3. Recommendations (`app/smart_recommendation_orchestrator.py`)

1. Запуск только при интенте `RECOMMEND`.
2. Используется тот же векторный поиск для получения кандидатов.
3. Промпт для LLM строится в `_rank_candidates_with_llm`:
   - Передаём список кандидатов (title, author, description).
   - Фиксируем нежелательные сущности из контекста (жанры/авторы, которые
     пользователь отверг ранее).
   - Просим вернуть JSON: `response` + `books[{title, author, reason}]`.
4. Если LLM не вернул JSON — используется fallback: простое перечисление книг.
5. Подтверждение пользователя текстом добавляется в `focus_entities` и может
   инициировать `SEARCH` на следующем шаге.

## 4. Analytics (`app/smart_analytics_orchestrator.py`)

1. Запуск только при интенте `ANALYTICS`.
2. Снова используем `SimpleVectorRetriever` для выборки релевантных книг.
3. Промпт в `_build_analytics_report` просит модель сформировать JSON с полями:
   `summary`, `stats`, `highlights`.
4. При успехе возвращается аналитическая сводка. При ошибке — текст с просьбой
   уточнить запрос.

## 5. Chat Response (`node_smart_chat` в `app/orchestrator_chat_agent.py`)

1. Запускается при интенте `CHAT`.
2. Оценивает настроение (`_analyze_user_satisfaction`) и формирует ответ через
   ChatOpenAI ("эмпатический" промпт внутри узла).
3. Не предлагает дальнейших действий — просто поддерживает диалог.

## 6. Clarify (`node_clarify` в `app/orchestrator_chat_agent.py`)

1. Запускается только при интенте `CLARIFY`.
2. Составляет уточняющий вопрос на основе контекста (последние сообщения,
   попытки уточнений). Чипы не используются — задаётся один конкретный вопрос.

## 7. Transition Manager (`app/smart_transition_system.py`)

- Принимает решение, какой узел вызвать после `intent`:
  - `SEARCH` → `simple_search`
  - `RECOMMEND` → `recommendations`
  - `ANALYTICS` → `analytics`
  - `CLARIFY` → `clarify`
  - Остальные → `chat_response`
- Если `needs_clarification = True`, узел принудительно переводит поток в
  `clarify`, даже если было другое намерение.

## Пошаговый поток запроса

1. Пользователь отправляет сообщение.
2. Узел `intent` (LLM) классифицирует запрос.
3. `smart_transition_manager` выбирает следующий узел.
4. Выбранный узел (search/recommend/analytics/chat/clarify) выполняет свою
   логику и мутирует общее состояние.
5. Итоговое сообщение возвращается пользователю.

## Промпты (сводка)

| Назначение                         | Файл                                 | Константа / функция            |
|-----------------------------------|--------------------------------------|--------------------------------|
| Определение интента               | `app/smart_intent_system.py`         | `ENHANCED_INTENT_PROMPT`       |
| Форматирование ответа поиска      | `app/smart_simple_orchestrator.py`   | `_format_response`             |
| Ranking рекомендаций              | `app/smart_recommendation_orchestrator.py` | `_rank_candidates_with_llm` |
| Аналитическая сводка              | `app/smart_analytics_orchestrator.py`| `_build_analytics_report`      |
| Ответ в режиме small talk         | `app/orchestrator_chat_agent.py`     | промпт внутри `node_smart_chat`|
| Уточняющий вопрос                 | `app/orchestrator_chat_agent.py`     | промпт внутри `node_clarify`   |

Документ отражает текущую реализацию без создания новых веток логики — мы лишь
зафиксировали поведение и промпты, которые уже используются в коде.
