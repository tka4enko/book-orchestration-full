"""Analytics helper producing trend-style answers from catalogue snippets."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .settings import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from .simple_retriever import SimpleVectorRetriever

logger = logging.getLogger(__name__)

_retriever = SimpleVectorRetriever()
_analytics_llm = ChatOpenAI(
    model=OPENAI_MODEL_CHAT,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    max_tokens=380,
)


async def process_smart_analytics(
    session_id: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate comprehensive analytics for the entire collection."""

    logger.info("📊 [analytics] session=%s message='%s'", session_id, message)
    total_start = time.time()

    analytics_start = time.time()

    # Use comprehensive analytics instead of limited search
    analytics_data = await _retriever.get_analytics_summary(message)

    # Generate analytical response based on full data
    report = _build_comprehensive_analytics_report(message, analytics_data)

    analytics_time = time.time() - analytics_start
    total_time = time.time() - total_start

    metrics = {
        "analytics_time": analytics_time,
        "total_time": total_time,
    }

    return {
        "response": report.get("summary"),
        "analytics_stats": report.get("stats", {}),
        "results": report.get("highlights", []),
        "performance_metrics": metrics,
    }


def _build_comprehensive_analytics_report(query: str, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build analytics report from comprehensive data"""
    try:
        total_books = analytics_data.get("total_books", 0)

        if total_books == 0:
            return {
                "summary": "No books found in the collection for analysis.",
                "stats": {},
                "highlights": []
            }

        # Generate response based on query type
        query_lower = query.lower()

        # Check for specific analytics requests
        if any(word in query_lower for word in ["сколько", "количество", "count", "how many"]):
            summary = f"Collection contains {total_books} books. "

            genres_data = analytics_data.get("genres", {})
            if genres_data:
                top_genre = max(genres_data.items(), key=lambda x: x[1])
                summary += f"Most popular genre: {top_genre[0]} ({top_genre[1]} books)."

        elif any(word in query_lower for word in ["жанр", "genre", "категори"]):
            genres_data = analytics_data.get("genres", {})
            if genres_data:
                top_genres = list(genres_data.items())[:5]
                genres_list = [f"{genre} ({count})" for genre, count in top_genres]
                summary = f"Top genres in collection: {', '.join(genres_list)}"
            else:
                summary = "Genre information not available."

        elif any(word in query_lower for word in ["автор", "author", "писател"]):
            authors_data = analytics_data.get("authors", {})
            if authors_data:
                top_authors = list(authors_data.items())[:5]
                authors_list = [f"{author} ({count})" for author, count in top_authors]
                summary = f"Top authors in collection: {', '.join(authors_list)}"
            else:
                summary = "Author information not available."

        elif any(word in query_lower for word in ["язык", "language", "languages"]):
            languages_data = analytics_data.get("languages", {})
            if languages_data:
                langs_list = [f"{lang} ({count})" for lang, count in languages_data.items()]
                summary = f"Languages in collection: {', '.join(langs_list)}"
            else:
                summary = "Language information not available."

        else:
            # General overview
            summary = f"Analysis of collection with {total_books} books. "

            genres_data = analytics_data.get("genres", {})
            authors_data = analytics_data.get("authors", {})

            if genres_data:
                top_genre = max(genres_data.items(), key=lambda x: x[1])
                summary += f"Predominant genre: {top_genre[0]}. "

            if authors_data:
                top_author = max(authors_data.items(), key=lambda x: x[1])
                summary += f"Most books by: {top_author[0]} ({top_author[1]} books)."

        # Prepare stats for detailed display
        stats = {
            "total_books": total_books,
            "top_genres": dict(list(analytics_data.get("genres", {}).items())[:3]),
            "top_authors": dict(list(analytics_data.get("authors", {}).items())[:3]),
            "languages": analytics_data.get("languages", {})
        }

        # Create highlights for display
        highlights = []
        genres_data = analytics_data.get("genres", {})
        for genre, count in list(genres_data.items())[:3]:
            highlights.append(f"Genre '{genre}': {count} books")

        return {
            "summary": summary,
            "stats": stats,
            "highlights": highlights
        }

    except Exception as e:
        logger.error(f"❌ Error building comprehensive analytics report: {e}")
        return {
            "summary": f"Analytics processing error: {str(e)}",
            "stats": {},
            "highlights": []
        }

def _build_analytics_report(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "summary": (
                f"По запросу «{query}» не удалось собрать статистику. Попробуйте задать другой вопрос "
                "или уточните, какие данные вам нужны."
            ),
            "stats": {},
            "highlights": [],
        }

    payload = []
    for item in results[:10]:
        metadata = item.get("metadata", {})
        payload.append(
            {
                "title": item.get("title") or metadata.get("title"),
                "author": item.get("author") or metadata.get("author"),
                "summary": metadata.get("summary") or (item.get("content") or ""),
                "genres": metadata.get("primary_genre"),
                "topics": metadata.get("main_topics"),
            }
        )

    system_prompt = (
        "You analyse catalogue snippets and produce a short analytics recap. "
        "Use only the supplied data. Return JSON with keys: summary (string), "
        "stats (object with up to 3 bullet facts), highlights (list of strings)."
    )
    user_prompt = (
        f"User analytics request: {query}\n"
        f"Candidate data: {json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        response = _analytics_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = json.loads(response.content)
        if isinstance(parsed, dict) and parsed.get("summary"):
            return parsed
    except Exception as exc:  # pragma: no cover - LLM failure
        logger.error("❌ [analytics] analysis failed: %s", exc)

    highlights = [
        f"{item.get('title') or 'Книга'} — {item.get('author') or 'автор неизвестен'}"
        for item in payload[:3]
    ]
    return {
        "summary": (
            "Удалось найти несколько релевантных книг, но точную статистику собрать не получилось. "
            "Попробуйте уточнить вопрос или указать конкретный период."),
        "stats": {},
        "highlights": highlights,
    }
