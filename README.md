# 📚 BookBot Full - Интеллектуальная система поиска книг с чат-интерфейсом

**BookBot Full** - это продвинутая AI-система для поиска книг, которая объединяет мощный гибридный поиск, интеллектуальное определение намерений, многоуровневую детекцию дубликатов и современный чат-интерфейс с поддержкой WebSocket для реального времени.

## 🎯 Ключевые особенности

### 🔍 Интеллектуальный поиск
- **Гибридный поиск** - оптимальное сочетание BM25 (ключевые слова) и векторного поиска (семантика)
- **Умное определение намерений** - автоматическое распознавание типа запроса (ISBN, автор, жанр, тема)
- **Адаптивные пороги схожести** - динамическая настройка на основе языка и типа запроса
- **Диверсификация результатов** - предотвращение доминирования одной книги в результатах

### 🧠 AI-возможности
- **LangGraph оркестрация** - сложные пайплайны обработки запросов
- **Контекстный чат** - поддержка истории диалога и контекстных ответов
- **LLM-анализ метаданных** - автоматическое извлечение информации о книгах
- **Детекция языка** - многоязычная поддержка с русским приоритетом

### 💬 Чат-интерфейс
- **Два режима работы** - классический поиск и интерактивный чат
- **WebSocket поддержка** - общение в реальном времени
- **Умные чипсы** - предлагаемые действия и продолжения диалога
- **Контекстные советы** - направление пользователя к поиску книг

### 🛡️ Защита от дубликатов
- **4-уровневая детекция** - хеш файла, ISBN, метаданные, содержимое
- **Гибкая конфигурация** - настраиваемые пороги и принудительная загрузка
- **Хранение хешей** - постоянное отслеживание загруженных файлов

## 🏗️ Архитектура системы

```
bookbot_full/
├── app/
│   ├── main.py                      # FastAPI сервер + WebSocket
│   ├── orchestrator.py              # Классический поисковый пайплайн
│   ├── orchestrator_with_chat.py    # Чат-пайплайн с контекстом
│   ├── retrievers.py               # Гибридные алгоритмы поиска
│   ├── ingest.py                   # Загрузка и обработка документов
│   ├── loaders.py                  # Загрузчики файлов (PDF, DOCX, TXT)
│   ├── metadata.py                 # Обработка метаданных
│   ├── metadata_llm.py             # LLM для анализа контента
│   ├── duplicate_detection.py      # Многоуровневая детекция дубликатов
│   ├── file_hash_store.py          # Хранилище хешей файлов
│   ├── utils_isbn.py               # ISBN нормализация и валидация
│   ├── settings.py                 # Конфигурация системы
│   └── static/
│       ├── index.html              # Основной веб-интерфейс
│       └── chat_test.html          # Тестовый чат-интерфейс
└── requirements.txt                # Python зависимости
```

## 💻 Технологический стек

### Backend
- **FastAPI** - высокопроизводительный веб-фреймворк
- **LangChain** - фреймворк для LLM приложений  
- **LangGraph** - граф-оркестрация AI пайплайнов
- **ChromaDB** - векторная база данных с персистентностью
- **BM25** - алгоритм ранжирования по ключевым словам
- **WebSocket** - реальное время для чата

### AI Модели
- **OpenAI GPT-4o-mini** - языковая модель для чата и анализа
- **text-embedding-3-small** - векторные эмбеддинги
- **LangSmith** - мониторинг и трассировка LLM

### Обработка данных
- **PyPDF2** - обработка PDF документов
- **python-docx** - обработка Word документов
- **isbnlib** - работа с ISBN номерами
- **langdetect** - детекция языков

## 🚀 Быстрый старт

### 1. Установка

```bash
git clone <repository-url>
cd bookbot_full
pip install -r requirements.txt
```

### 2. Настройка окружения

Создайте `.env` файл:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-small

# База данных
CHROMA_DIR=.chroma

# Сервер
HOST=127.0.0.1
PORT=8000

# LangSmith (опционально)
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Настройки поиска (опционально)
MAX_CHUNKS_PER_BOOK=2
CHUNKS_PER_BOOK_IN_CONTENT=2
CONTENT_SEARCH_EXPAND_K=20

# JSON конфигурации (опционально)
SIMILARITY_THRESHOLDS={"author": 0.8, "title": 0.7, "topic": 0.4, "free_text": 0.5}
BM25_THRESHOLDS={"author": 0.1, "title": 0.1, "topic": 0.6, "free_text": 0.4}
LANGUAGE_THRESHOLD_MODIFIERS={"same_language": -0.1, "different_language": 0.0}
```

### 3. Запуск

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Доступ

- **Веб-интерфейс**: http://127.0.0.1:8000
- **Чат-тест**: http://127.0.0.1:8000/chat_test
- **API документация**: http://127.0.0.1:8000/docs
- **Health check**: http://127.0.0.1:8000/health

## 📖 Использование

### Веб-интерфейсы

#### Основной интерфейс (`/`)
- Стандартный поиск с форматированными результатами
- Поддержка всех типов запросов
- JSON ответы с детальной информацией

#### Чат-интерфейс (`/chat_test`)
- Диалоговый режим с историей
- WebSocket соединение в реальном времени
- Контекстные подсказки и чипсы
- Умное переключение между чатом и поиском

### API Эндпоинты

#### Поиск книг
```bash
POST /chat
Content-Type: application/json

{
  "session_id": "unique_session_id",
  "message": "найди книги Орвелла"
}
```

Ответ:
```json
{
  "session_id": "unique_session_id",
  "intent": "author",
  "results": [{
    "message": "1. «1984» от George Orwell - Антиутопия о тотальном контроле",
    "intent": "author"
  }]
}
```

#### Чат через WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat_test');
ws.send(JSON.stringify({
  "session_id": "test_session",
  "message": "привет, посоветуй что почитать"
}));
```

Ответ:
```json
{
  "reply": "Привет! Что тебя интересует?",
  "cards": [],
  "chips": [
    {"text": "Детектив", "action": "search"},
    {"text": "Фэнтези", "action": "search"}
  ],
  "intent": "chat"
}
```

#### Загрузка книг
```bash
POST /ingest
Content-Type: multipart/form-data

file: [PDF/DOCX/TXT файл]
meta: {"title": "Название", "author": "Автор", "prefer_llm": true}
max_chunks: 50
force_ingest: false
```

С детекцией дубликатов:
```json
{
  "status": "duplicate_detected",
  "message": "Document rejected: ISBN 9781234567890 already exists",
  "duplicate_level": "isbn",
  "duplicate_confidence": 1.0,
  "existing_document_id": "existing-doc-123"
}
```

#### Просмотр коллекций
```bash
# Книги (мастер-записи)
GET /vector-store/collection/books?limit=10

# Контент (фрагменты)  
GET /vector-store/collection/content?limit=10

# Конкретная книга
GET /vector-store/collection/books/chunks?document_id=book-id-123

# Поиск в коллекции
GET /vector-store/collection/books/search?q=Орвелл&k=5

# Отдельный фрагмент
GET /vector-store/chunk/books/book-id-123
```

## 🔍 Алгоритмы поиска

### Гибридный поиск (OptimizedThresholdRetriever)

Многоэтапный пайплайн с приоритетами:

```
1. BM25 поиск по ключевым словам (приоритет)
   ├─ Точные совпадения в метаданных
   ├─ Фильтрация по BM25-порогам
   └─ Исключение из векторного поиска

2. Векторный поиск (дополнение)
   ├─ Семантический поиск в книгах
   ├─ Диверсифицированный поиск в контенте  
   └─ Фильтрация по similarity-порогам

3. Умная дедупликация
   ├─ Группировка по книгам
   ├─ Объединение мастер + лучший фрагмент
   └─ Приоритет мастер-записей
```

### Определение намерений

LLM анализирует запросы и контекст для определения типа:

```python
# Точный поиск
"isbn": "978-1234567890"        # ISBN номер
"author": "Стивен Кинг"         # Автор
"title": "1984"                 # Название
"author_title": "Кинг Сияние"   # Автор + название

# Семантический поиск  
"genre": "детектив"             # Жанр
"topic": "психология"           # Тема
"free_text": "интересная книга" # Свободный текст

# Чат
"greeting": "привет"            # Приветствие
"clarify": "посоветуй жанр"     # Нужен совет
"chat": "как дела?"             # Общение
```

### Контекстный анализ в чате

Система анализирует историю диалога:

```python
# Обработка согласий
История: "assistant: Как насчет фэнтези или детективов?"
Пользователь: "давай"
→ Предлагает выбор между вариантами

История: "assistant: Попробуй Стивена Кинга"
Пользователь: "хорошо"
→ Ищет книги Кинга

# Направление к книгам
После 4+ сообщений без книжных тем
→ Мягко предлагает книги
→ При согласии переключается на подбор
```

### Адаптивные пороги

```python
# Базовые пороги по намерениям
SIMILARITY_THRESHOLDS = {
    "author": 0.8,        # Высокий - точный поиск
    "isbn": 0.9,          # Очень высокий
    "title": 0.7,         # Высокий
    "topic": 0.4,         # Низкий - семантика
    "genre": 0.5,         # Средний
    "free_text": 0.5      # Универсальный
}

# Модификаторы по языку
LANGUAGE_THRESHOLD_MODIFIERS = {
    "same_language": -0.1,      # Снижение для одного языка
    "different_language": 0.0,  # Без изменений
    "unknown_language": -0.05   # Мягкое снижение
}

# Результирующий порог = базовый + модификатор языка
```

## 🛡️ Детекция дубликатов

### 4-уровневая система

```python
Level 1: Хеш файла (SHA-256)
├─ Проверка идентичности файлов
├─ Confidence: 1.0 при совпадении
└─ Останов при обнаружении

Level 2: ISBN проверка
├─ Нормализация ISBN-10/13
├─ Поиск в базе книг
└─ Confidence: 1.0 при совпадении

Level 3: Метаданные (автор + название)
├─ Каноническая нормализация
├─ Точное и нечеткое сравнение
└─ Confidence: 0.9+ при высокой схожести

Level 4: Схожесть контента
├─ Хеш содержимого + семплирование
├─ n-gram анализ (Jaccard similarity)
└─ Confidence: 0.95+ при схожести
```

### Настройки детекции

```python
# Пороги блокировки
confidence_threshold = 0.8   # Минимальная уверенность для блока
force_ingest = False        # Принудительная загрузка

# Стратегия остановки
- Level 1 (файл): немедленная остановка
- Level 2 (ISBN): остановка при confidence >= 0.9  
- Level 3 (метаданные): остановка при confidence >= 0.9
- Level 4 (контент): финальная проверка
```

## 📊 Структура данных

### Коллекция "books" (мастер-записи)

```python
{
  "document_id": "unique-uuid",
  "title": "Название книги", 
  "author": "Имя Автора",
  "isbn13": "9781234567890",
  "isbn10": "1234567890", 
  "summary": "Краткое описание...",
  "primary_genre": "Фантастика",
  "secondary_genres": ["Антиутопия", "Классика"],
  "main_topics": ["Тоталитаризм", "Контроль"],
  "mentioned_topics": ["Политика", "Общество"],
  "language": "en",
  "year": 1949,
  "is_master_chunk": true,
  "title_canonical": "название книги",
  "author_canonical": "имя автора",
  "file_type": "pdf"
}
```

### Коллекция "content" (фрагменты)

```python
{
  "document_id": "parent-book-uuid",
  "title": "Название книги",
  "author": "Имя Автора", 
  "language": "en",
  "idx": 0  // индекс фрагмента
}
```

### Формат мастер-текста

Обогащенный текст для семантического поиска:
```
Название книги — Полное описание — Автор: Имя Автора — Год: 1949 — Жанр: Фантастика, Антиутопия — Темы: Тоталитаризм, Контроль, Политика
```

## ⚙️ Конфигурация

### Параметры поиска (settings.py)

```python
# Размеры фрагментов
CHUNK_SIZE = 1200          # Размер фрагмента текста
CHUNK_OVERLAP = 120        # Перекрытие между фрагментами  
MAX_CHUNKS = 80           # Максимум фрагментов на книгу

# Параметры алгоритмов
BM25_K = 8                # Результатов от BM25
VEC_BOOKS_K = 8          # Результатов из книг
VEC_CONTENT_K = 6        # Результатов из контента
RRF_K = 60               # Reciprocal Rank Fusion

# Диверсификация
MAX_CHUNKS_PER_BOOK = 2             # Макс фрагментов на книгу
CHUNKS_PER_BOOK_IN_CONTENT = 2      # Фрагментов на книгу в контенте
CONTENT_SEARCH_EXPAND_K = 20        # Расширенный поиск для отбора

# Фильтрация
MIN_SIMILARITY_THRESHOLD = 0.6      # Устаревший глобальный порог
```

### Динамические настройки через ENV

```bash
# JSON конфигурации - позволяют гибкую настройку без перезапуска
SIMILARITY_THRESHOLDS='{"author": 0.8, "title": 0.7, "topic": 0.4}'
BM25_THRESHOLDS='{"isbn": 0.05, "author": 0.1, "topic": 0.6}'
LANGUAGE_THRESHOLD_MODIFIERS='{"same_language": -0.1, "different_language": 0.0}'
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `OPENAI_API_KEY` | Ключ OpenAI API | **обязательно** |
| `OPENAI_MODEL_CHAT` | Чат модель | `gpt-4o-mini` |
| `OPENAI_MODEL_EMBED` | Embedding модель | `text-embedding-3-small` |
| `CHROMA_DIR` | Папка ChromaDB | `.chroma` |
| `HOST` | Хост сервера | `127.0.0.1` |
| `PORT` | Порт сервера | `8000` |
| `LANGSMITH_TRACING` | Трассировка LangSmith | `false` |
| `LANGSMITH_PROJECT` | Проект LangSmith | - |
| `LANGSMITH_API_KEY` | Ключ LangSmith | - |
| `LANGSMITH_ENDPOINT` | Эндпоинт LangSmith | - |

## 🎨 Пользовательский интерфейс

### Основные компоненты

#### 1. Основной поиск (`index.html`)
- Форма поиска с автофокусом
- Результаты с детальной информацией
- Форма загрузки файлов с настройками
- Просмотр коллекций векторной базы

#### 2. Чат-интерфейс (`chat_test.html`) 
- WebSocket подключение
- История сообщений с прокруткой
- Умные чипсы-кнопки для действий
- Индикация состояния соединения
- Отладочная информация

### Интерактивные элементы

```javascript
// Чипсы для быстрых действий
chips: [
  {"text": "Детектив", "action": "search"},
  {"text": "Фэнтези", "action": "search"},
  {"text": "Поговорим о другом", "action": "chat"}
]

// Типы действий
action: "search" // Поиск книг
action: "chat"   // Продолжение диалога
```

## 🔧 API Reference

### Основные эндпоинты

#### `GET /` - Главная страница
Возвращает HTML интерфейс для поиска

#### `GET /chat_test` - Тестовый чат
Возвращает HTML интерфейс чата с WebSocket

#### `GET /health` - Проверка здоровья
```json
{"ok": true}
```

#### `POST /chat` - Поиск книг
Основной эндпоинт для поиска через REST API

Запрос:
```json
{
  "session_id": "string",
  "message": "string"  
}
```

Ответ:
```json
{
  "session_id": "string",
  "message": "string", 
  "intent": "author|title|isbn|genre|topic|free_text|clarify",
  "filters": {},
  "results": [
    {
      "title": "string",
      "author": "string", 
      "isbn13": "string",
      "summary": "string",
      "message": "string",  // Форматированный ответ
      "intent": "string"
    }
  ],
  "need_clarify": false,
  "clarify_question": "string"
}
```

#### `POST /ingest` - Загрузка документов
Загрузка книг с автоматической обработкой

Параметры (multipart/form-data):
- `file`: PDF/DOCX/TXT файл
- `meta`: JSON с метаданными (опционально)
- `prefer_llm`: "true"/"false" - использовать LLM для анализа
- `max_chunks`: число - максимум фрагментов
- `force_ingest`: "true"/"false" - принудительная загрузка

Успешный ответ:
```json
{
  "status": "success",
  "message": "Document 'Title' ingested successfully", 
  "book_id": "book:hash16",
  "document_id": "uuid",
  "chunks": 25,
  "metadata": {
    "title": "string",
    "author": "string",
    "language": "ru", 
    "primary_genre": "string",
    "isbn13": "string"
  },
  "duplicate_checks": []
}
```

Дублирование:
```json
{
  "status": "duplicate_detected",
  "message": "Document rejected: reason",
  "duplicate_level": "file_hash|isbn|metadata|content_similarity",
  "duplicate_confidence": 0.95,
  "existing_document_id": "uuid",
  "duplicate_checks": [
    {
      "level": "string",
      "is_duplicate": true,
      "reason": "string", 
      "confidence": 0.95
    }
  ]
}
```

#### `WebSocket /ws/chat_test` - Чат в реальном времени

Сообщение от клиента:
```json
{
  "session_id": "string",
  "message": "string"
}
```

Ответ сервера:
```json
{
  "reply": "string",           // Ответ бота
  "cards": [],                 // Карточки (будущая функция)
  "chips": [                   // Предлагаемые действия
    {
      "text": "string", 
      "action": "search|chat"
    }
  ],
  "intent": "search|chat",     // Тип обработки
  "debug": {                   // Отладочная информация
    "intent": "string",
    "mode": "chat_test",
    "processing_time": "string"
  }
}
```

### Коллекции векторной базы

#### `GET /vector-store/collection/{name}` - Обзор коллекции
Параметры:
- `name`: "books" | "content"
- `limit`: число результатов (1-200, по умолчанию 3)

#### `GET /vector-store/collection/{name}/chunks` - Фрагменты документа  
Параметры:
- `name`: имя коллекции
- `document_id`: ID документа (опционально)
- `offset`: смещение (по умолчанию 0)
- `limit`: лимит (1-200, по умолчанию 50)

#### `GET /vector-store/collection/{name}/search` - Поиск в коллекции
Параметры:
- `name`: имя коллекции 
- `q`: поисковый запрос
- `k`: количество результатов (1-50, по умолчанию 5)

#### `GET /vector-store/chunk/{name}/{chunk_id}` - Конкретный фрагмент
Возвращает полную информацию о фрагменте

## 🚀 Производительность и оптимизации

### Алгоритмические оптимизации

```python
# 1. Условный векторный поиск
# Пропуск векторного поиска когда BM25 дал хорошие результаты
if intent in ["isbn", "author"] and high_quality_bm25_matches > 0:
    skip_vector_search = True

# 2. Исключение дубликатов в векторном поиске  
# BM25 результаты исключаются из векторного поиска
for doc, score in vector_results:
    if doc.document_id not in bm25_found_ids:
        keep_result(doc, score)

# 3. Диверсификация контента
# Ограничение фрагментов на книгу для разнообразия
for book_id, chunks in grouped_chunks.items():
    best_chunks = sorted(chunks, key=score)[:CHUNKS_PER_BOOK_IN_CONTENT]

# 4. Умная дедупликация
# Объединение мастер-записи с лучшим фрагментом
merged_doc.page_content = master.content + "\n\n" + best_chunk.content
```

### Кэширование и переиспользование

- **Embedding переиспользование** - векторный поиск использует уже вычисленные эмбеддинги
- **BM25 построение** - единоразовое построение индекса при запуске  
- **Персистентная база** - ChromaDB сохраняет данные между перезапусками
- **Хеш-хранилище** - быстрая проверка дубликатов файлов

### Масштабирование

- **Асинхронная загрузка** - `asyncio.to_thread` для блокирующих операций
- **WebSocket pool** - поддержка множественных чат-сессий
- **Модульная архитектура** - легкое расширение компонентов
- **Конфигурируемые лимиты** - настройка под объем данных

## 🔒 Безопасность

### Валидация данных
- **Pydantic модели** - строгая типизация API
- **Размер файлов** - ограничения на загрузку 
- **Форматы файлов** - поддержка только разрешенных типов
- **SQL injection** - параметризованные запросы к ChromaDB

### Детекция аномалий
- **Дублирование** - предотвращение спам-загрузок
- **Пустые файлы** - проверка содержимого
- **Некорректные ISBN** - валидация через isbnlib
- **Подозрительные запросы** - логирование и мониторинг

### Логирование и аудит
```python
# Детальное логирование всех операций
logger.info("🔍 [retrievers.py] Starting hybrid search...")
logger.warning("🚨 DUPLICATE DETECTED: ISBN already exists")
logger.error("❌ [main.py] Chat request failed")

# Трассировка через LangSmith
LANGSMITH_TRACING=true  # Отслеживание LLM вызовов
```

## 🧪 Разработка и отладка

### Структура логов

```
🔥 CHAT REQUEST STARTED          # Начало обработки
🧠 node_detect_intent            # Определение намерения  
🔍 node_route_search             # Поиск документов
💬 node_answer                   # Формирование ответа
🎉 CHAT REQUEST COMPLETED        # Завершение
```

### Уровни детализации

```python
# Логирование поискового пайплайна
🎯 OptimizedThresholdRetriever   # Основной алгоритм
  🔤 Step 1: BM25 search         # BM25 поиск
  📊 Step 2: Vector search       # Векторный поиск  
  🔍 Step 3: Content search      # Поиск в контенте
  🧹 Smart deduplication         # Дедупликация

# Детекция дубликатов
🔍 Level 1: File hash check      # Проверка хеша
🔍 Level 2: ISBN check           # Проверка ISBN
🔍 Level 3: Metadata check       # Проверка метаданных
🔍 Level 4: Content similarity   # Проверка содержимого
```

### Отладочные эндпоинты

```bash
# Статус коллекций
GET /vector-store/collection/books
GET /vector-store/collection/content

# Поиск по коллекциям
GET /vector-store/collection/books/search?q=test&k=5

# Отдельные документы
GET /vector-store/chunk/books/book-id-123
```

### Настройка отладки

```env
# Включение детального логирования
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot_debug

# Снижение порогов для тестирования  
SIMILARITY_THRESHOLDS='{"free_text": 0.3}'
BM25_THRESHOLDS='{"free_text": 0.2}'
```

## 🤝 Расширение системы

### Добавление новых форматов файлов

```python
# В loaders.py
def load_text_from_file(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    
    # Добавить новый формат
    if ext == ".epub":
        return load_epub(path), "epub"
    
    # Существующие форматы
    if ext in (".txt", ".md", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), ext.lstrip(".")
```

### Новые алгоритмы поиска

```python
# В retrievers.py
class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Кастомная логика поиска
        return documents

def custom_hybrid_search(intent: str = "free_text"):
    return CustomRetriever()
```

### Дополнительные детекторы дубликатов

```python
# В duplicate_detection.py
def level5_semantic_similarity(text: str) -> DuplicateDetectionResult:
    """Level 5: Семантическая схожесть через эмбеддинги"""
    # Векторное сравнение содержимого
    return DuplicateDetectionResult(...)

# Обновить основную функцию
def detect_duplicates(...):
    # Существующие уровни 1-4
    result5 = level5_semantic_similarity(full_text) 
    results.append(result5)
```

### Кастомные намерения

```python
# В orchestrator.py или orchestrator_with_chat.py
INTENT_SYS = """
Добавить новые намерения:
- series: поиск серий книг
- publisher: поиск по издательству  
- year_range: поиск по годам
"""

# В node_route_search добавить обработку
if state.intent == "series":
    # Логика поиска серий
```

## 📈 Мониторинг и метрики

### LangSmith интеграция

```env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=bookbot_production
LANGSMITH_API_KEY=your_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Отслеживаемые метрики:
- Время выполнения LLM вызовов
- Токены использованные для анализа
- Успешность определения намерений
- Качество поисковых результатов

### Логи производительности

```python
# Время выполнения этапов
logger.info("🔍 BM25 search completed in 0.15s")
logger.info("📊 Vector search completed in 0.32s")  
logger.info("🧹 Deduplication completed in 0.08s")
logger.info("🎯 Total search time: 0.55s")

# Статистика результатов
logger.info("📊 BM25 found: 12 → passed threshold: 3")
logger.info("📚 Vector found: 24 → excluded: 2 → kept: 22 → passed: 7")
logger.info("🎯 FINAL RESULTS: 8 documents after deduplication")
```

### Основные KPI

- **Точность поиска** - релевантность результатов запросу
- **Время отклика** - скорость обработки запросов
- **Покрытие коллекции** - процент находимых книг
- **Дублирование** - эффективность детекции
- **Использование ресурсов** - потребление OpenAI API

## ❓ Устранение проблем

### Частые проблемы

#### 1. Нет результатов поиска
```bash
# Проверка коллекций
GET /vector-store/collection/books

# Снижение порогов
SIMILARITY_THRESHOLDS='{"free_text": 0.2}'
```

#### 2. Медленная работа
```python
# Проверка размера коллекции
collection.count()  # Если > 10000 - оптимизировать

# Уменьшение параметров поиска
VEC_BOOKS_K = 5
CONTENT_SEARCH_EXPAND_K = 10
```

#### 3. Проблемы с загрузкой
```python
# Принудительная загрузка
force_ingest = True

# Отладка детекции дубликатов
logger.info("Duplicate checks: %s", duplicate_results)
```

#### 4. WebSocket отключения
```javascript
// Переподключение
ws.onclose = function() {
    setTimeout(connectWebSocket, 1000);
};
```

### Диагностические команды

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Тестовый поиск
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"тест"}'

# Статус коллекций
curl "http://localhost:8000/vector-store/collection/books?limit=1"
```

## 📄 Лицензия

MIT License - свободное использование и модификация.

## 🆘 Поддержка

1. **Проверьте конфигурацию** - `.env` файл и ключи API
2. **Изучите логи** - детальное логирование всех операций  
3. **Тестируйте компоненты** - используйте диагностические эндпоинты
4. **Создайте Issue** - для сообщения о багах и предложений

---

**BookBot Full** - современная AI-система для интеллектуального поиска книг с продвинутыми возможностями чата и гибридными алгоритмами! 📚✨🤖