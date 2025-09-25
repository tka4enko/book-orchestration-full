# 📊 BookBot Full - Карта проекта

## 🎯 Зоны ответственности

### **ЯДРО РАНТАЙМА**
```
🧠 AGENTS & ORCHESTRATORS (Основная логика)
├── orchestrator_chat_agent.py      # Главный LangGraph чат-агент
├── orchestrator_with_chat.py       # Альтернативный чат-оркестратор
├── orchestrator.py                 # Классический поисковый граф
├── simple_orchestrator.py          # Упрощенный поиск
├── smart_recommendation_orchestrator.py  # Персонализированные рекомендации
├── smart_analytics_orchestrator.py # Аналитика коллекции
└── secure_orchestrator.py          # Безопасный вариант

🔍 SEARCH & RETRIEVAL (Поиск и извлечение)
├── retrievers.py                   # Гибридный BM25 + vector retriever
├── simple_retriever.py             # Простой векторный поиск
└── simple_llm_filter.py            # LLM-фильтрация результатов

📥 INGESTION (Загрузка и обработка)
├── ingest.py                       # Пакетная загрузка документов
├── loaders.py                      # Загрузчики файлов (PDF/DOCX/TXT)
├── duplicate_detection.py          # 4-уровневое детектирование дубликатов
├── file_hash_store.py              # Хранение хешей файлов
├── metadata.py                     # Извлечение метаданных
└── metadata_llm.py                 # LLM-анализ метаданных
```

### **ИНФРАСТРУКТУРА**
```
⚙️ CONFIGURATION & UTILS
├── settings.py                     # Настройки и константы
├── secure_config.py                # Безопасные конфигурации
├── utils_isbn.py                   # Работа с ISBN
├── mixed_filters_parser.py         # Парсинг сложных запросов
├── error_handler.py                # Обработка ошибок
└── security.py                     # Механизмы безопасности

🌐 WEB INTERFACES
├── main.py                         # FastAPI сервер (основной)
├── main_secure.py                  # Защищенный FastAPI
└── static/                         # HTML интерфейсы
    ├── chat_agent.html             # Интерфейс чат-агента
    ├── chat_test.html              # Тестовый чат
    ├── simple_chat.html            # Простой чат
    └── index.html                  # Главная страница
```

### **ТЕСТЫ И ДОКУМЕНТАЦИЯ**
```
📋 DOCUMENTATION
├── README.md                       # Общее описание
├── CLAUDE.md                       # Инструкции для Claude Code
└── docs/
    ├── agent_flow.md               # Схема работы агента
    └── chat_agent_improvement_guide.md  # Руководство по улучшению

🧪 TESTS
├── tests/
│   └── test_chat_agent_conversations.py  # Тесты разговоров
└── logs/                           # Логи выполнения
```

## 🏗️ Архитектурные слои

### **Слой 1: Интерфейсы**
- `main.py`, `main_secure.py` - API точки входа
- `static/` - Web UI компоненты

### **Слой 2: Агенты**
- `orchestrator_chat_agent.py` - Главный LangGraph агент
- Остальные orchestrator_*.py - Специализированные агенты

### **Слой 3: Сервисы**
- `retrievers.py`, `simple_retriever.py` - Поисковые сервисы
- `smart_*_orchestrator.py` - Сервисы рекомендаций/аналитики

### **Слой 4: Данные**
- `ingest.py`, `loaders.py` - Обработка данных
- `metadata*.py` - Извлечение метаданных
- `duplicate_detection.py` - Валидация данных

### **Слой 5: Инфраструктура**
- `settings.py` - Конфигурация
- `utils_*.py` - Утилиты
- `security.py` - Безопасность

## 🎯 Проблемы текущей структуры

❌ **Все в одной папке** - сложно навигировать
❌ **Промпты в коде** - сложно версионировать
❌ **Нет явных контрактов** - импорты везде
❌ **Мало тестов** - только один файл тестов
❌ **Конфиги размазаны** - в settings.py и по коду
❌ **Нет автоматизации** - ручной стиль кода

## 🚀 Предлагаемая структура

```
bookbot_full/
├── app/
│   ├── core/           # Общие модели, utils
│   ├── agents/         # LangGraph подграфы
│   ├── services/       # Бизнес-логика
│   ├── interfaces/     # API, CLI
│   └── infra/          # Настройки, хранилища
├── config/             # YAML/TOML конфиги
├── prompts/            # Шаблоны промптов
├── static/             # Статические файлы
├── tests/              # Структурированные тесты
└── docs/               # Документация
```

## 📋 План миграции

1. ✅ **Анализ текущего состояния** (этот файл)
2. 🔄 **Создание новой структуры папок**
3. 🔄 **Перемещение файлов по зонам ответственности**
4. 🔄 **Извлечение схем и контрактов**
5. 🔄 **Вынос промптов и конфигов**
6. 🔄 **Разделение на подграфы**
7. 🔄 **Реструктуризация тестов**
8. 🔄 **Автоматизация (pyproject.toml)**
9. 🔄 **Документирование архитектуры**

---
*Сгенерировано для рефакторинга BookBot Full*