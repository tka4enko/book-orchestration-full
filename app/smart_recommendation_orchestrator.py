"""Smart recommendation orchestrator with context-aware personalized recommendations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .settings import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from .simple_retriever import SimpleVectorRetriever

logger = logging.getLogger(__name__)

_retriever = SimpleVectorRetriever()
_recommendation_llm = ChatOpenAI(
    model=OPENAI_MODEL_CHAT,
    temperature=0.3,
    api_key=OPENAI_API_KEY,
    max_tokens=500,
)

# Recommendation prompts and configurations
PREFERENCE_EXTRACTION_PROMPT = """You are an expert at understanding user preferences from conversation context.

Analyze the conversation history and extract user preferences and dislikes about books.

EXTRACT:
1. LIKES: Genres, authors, themes, book types user shows interest in
2. DISLIKES: What user explicitly doesn't want or shows negative sentiment toward
3. CONTEXT: Current mood, situation, reading goals mentioned

CONVERSATION HISTORY:
{chat_history}

CURRENT REQUEST: {current_message}

Return JSON:
{{
    "likes": {{
        "genres": ["genre1", "genre2"],
        "authors": ["author1", "author2"],
        "themes": ["theme1", "theme2"],
        "book_types": ["fiction", "non-fiction", "classics"]
    }},
    "dislikes": {{
        "genres": ["disliked_genre"],
        "authors": ["disliked_author"],
        "themes": ["theme_to_avoid"]
    }},
    "context": {{
        "mood": "relaxing reading",
        "situation": "bedtime reading",
        "goal": "learn something new"
    }},
    "recommendation_type": "mood-based|discovery|trending|similar|author-based"
}}
"""

SMART_RECOMMENDATION_PROMPT = """You are an expert book curator. You MUST recommend ONLY books from the available database provided below.

USER PREFERENCES:
{user_preferences}

AVAILABLE BOOKS IN DATABASE (you can ONLY recommend from this list):
{books_data}

BOOKS TO EXCLUDE (already seen/mentioned):
{exclude_books}

CRITICAL RULES:
1. You can ONLY recommend books that appear in the "AVAILABLE BOOKS IN DATABASE" list above
2. DO NOT invent or hallucinate book titles or authors that are not in the provided list
3. Select 3-5 books maximum from the available list that best match user preferences
4. Skip books that appear in the "exclude" list
5. If no suitable books are found in the database, return empty recommendations list

SELECTION CRITERIA:
- Match user's preferred genres, authors, themes from available books
- Avoid user's dislikes if specified
- Provide variety when possible
- Include reasoning based on user preferences and book metadata

Return JSON (use exact titles and authors from the provided database):
{{
    "recommendations": [
        {{
            "title": "EXACT_TITLE_FROM_DATABASE",
            "author": "EXACT_AUTHOR_FROM_DATABASE",
            "genre": "primary_genre_from_database",
            "reasoning": "Why this specific book from our database fits user preferences (reference genres, topics, summary)",
            "confidence": 0.9
        }}
    ],
    "summary": "Explanation of selection strategy based on available database books and user preferences"
}}

IMPORTANT: Use the exact "title" and "author" fields from the database entries above. Pay attention to the "summary", "topics", and "all_genres" fields to make better matches with user preferences.
"""


async def process_smart_recommendations(
    session_id: str,
    current_message: str,
    chat_history: List[Dict[str, Any]],
    user_preferences: Optional[Dict[str, Any]] = None,
    exclude_books: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Generate smart recommendations based on user preferences and context."""

    logger.info("💡 [smart_recommendations] session=%s message='%s'", session_id, current_message)
    total_start = time.time()

    try:
        # Step 1: Extract/update user preferences from chat context
        preferences_start = time.time()
        if not user_preferences:
            user_preferences = await _extract_user_preferences(current_message, chat_history)
        preferences_time = time.time() - preferences_start

        # Step 2: Get available books data
        data_start = time.time()
        books_data = await _get_books_for_recommendations(user_preferences, exclude_books or set())
        data_time = time.time() - data_start

        # Step 3: Generate smart recommendations
        rec_start = time.time()
        recommendations = await _generate_recommendations(
            user_preferences, books_data, exclude_books or set()
        )
        rec_time = time.time() - rec_start

        # Step 4: Format response
        response_text = _format_recommendation_response(recommendations, user_preferences)

        total_time = time.time() - total_start

        metrics = {
            "preferences_extraction_time": preferences_time,
            "data_retrieval_time": data_time,
            "recommendation_generation_time": rec_time,
            "total_time": total_time,
        }

        return {
            "response": response_text,
            "recommendations": recommendations.get("recommendations", []),
            "user_preferences": user_preferences,
            "recommendation_stats": {
                "total_recommendations": len(recommendations.get("recommendations", [])),
                "strategy": recommendations.get("summary", ""),
            },
            "performance_metrics": metrics,
        }

    except Exception as e:
        logger.error("❌ [smart_recommendations] Processing failed: %s", e)
        return {
            "response": "Sorry, couldn't generate personalized recommendations right now. Try asking for specific genres or authors!",
            "recommendations": [],
            "user_preferences": user_preferences or {},
            "error": str(e),
            "performance_metrics": {"total_time": time.time() - total_start},
        }


async def _extract_user_preferences(
    current_message: str, chat_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract user preferences from conversation context."""
    try:
        # Format chat history
        history_text = ""
        if chat_history:
            recent_history = chat_history[-10:]  # Last 10 messages
            history_lines = []
            for entry in recent_history:
                role = entry.get("role", "user")
                content = entry.get("content", "")
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

        prompt = PREFERENCE_EXTRACTION_PROMPT.format(
            chat_history=history_text or "No previous conversation",
            current_message=current_message
        )

        response = _recommendation_llm.invoke([
            SystemMessage(content="You are a book preference analyst."),
            HumanMessage(content=prompt)
        ])

        # Parse JSON response
        try:
            preferences = json.loads(response.content)
            logger.info("🎯 [preferences] Extracted: %s", preferences)
            return preferences
        except json.JSONDecodeError:
            logger.warning("⚠️ [preferences] Failed to parse LLM response, using defaults")
            return _get_default_preferences()

    except Exception as e:
        logger.error("❌ [preferences] Extraction failed: %s", e)
        return _get_default_preferences()


async def _get_books_for_recommendations(
    user_preferences: Dict[str, Any], exclude_books: Set[str]
) -> List[Dict[str, Any]]:
    """Get books data suitable for recommendations."""
    try:
        # Get comprehensive book data from database
        all_books = await _retriever.get_all_books_metadata()

        if not all_books:
            return []

        # Filter out excluded books
        filtered_books = []
        for book in all_books:
            book_title = book.get("title", "")
            if book_title not in exclude_books:
                filtered_books.append(book)

        # Limit to reasonable number for LLM processing
        return filtered_books[:50]  # Top 50 books for analysis

    except Exception as e:
        logger.error("❌ [books_data] Failed to get books: %s", e)
        return []


async def _generate_recommendations(
    user_preferences: Dict[str, Any],
    books_data: List[Dict[str, Any]],
    exclude_books: Set[str]
) -> Dict[str, Any]:
    """Generate personalized recommendations using LLM."""
    try:
        if not books_data:
            return {
                "recommendations": [],
                "summary": "No books available for recommendations"
            }

        # Prepare data for LLM with comprehensive book information
        books_summary = []
        for i, book in enumerate(books_data[:30]):  # Limit for prompt size
            # Extract comprehensive metadata
            title = book.get("title", "Unknown")
            author = book.get("author", "Unknown")
            primary_genre = book.get("primary_genre", "Unknown")

            # Get secondary genres if available
            secondary_genres = book.get("secondary_genres", "[]")
            try:
                import json
                if secondary_genres and secondary_genres != "[]":
                    sec_genres = json.loads(secondary_genres)
                    all_genres = [primary_genre] + (sec_genres if isinstance(sec_genres, list) else [])
                else:
                    all_genres = [primary_genre]
            except:
                all_genres = [primary_genre]

            # Get summary and topics
            summary = book.get("summary", "")
            topics = book.get("main_topics", "")
            language = book.get("language", "Unknown")
            year = book.get("year", "Unknown")

            book_info = {
                "index": i,  # Add index for easier reference
                "title": title,
                "author": author,
                "primary_genre": primary_genre,
                "all_genres": all_genres[:3],  # Limit to 3 genres
                "language": language,
                "year": year,
                "topics": topics if topics else "Not specified",
                "summary": summary[:300] + "..." if len(summary) > 300 else summary or "No summary available"
            }
            books_summary.append(book_info)

        prompt = SMART_RECOMMENDATION_PROMPT.format(
            user_preferences=json.dumps(user_preferences, ensure_ascii=False, indent=2),
            books_data=json.dumps(books_summary, ensure_ascii=False, indent=2),
            exclude_books=list(exclude_books) if exclude_books else []
        )

        response = _recommendation_llm.invoke([
            SystemMessage(content="You are an expert book curator and recommendation specialist."),
            HumanMessage(content=prompt)
        ])

        # Parse recommendations
        try:
            recommendations = json.loads(response.content)

            # Validate that all recommendations exist in our database
            validated_recommendations = _validate_recommendations(
                recommendations.get("recommendations", []),
                books_data
            )

            # Update recommendations with validated list
            recommendations["recommendations"] = validated_recommendations

            logger.info("📚 [recommendations] Generated %d recommendations, %d validated",
                       len(recommendations.get("recommendations", [])),
                       len(validated_recommendations))

            # If no valid recommendations found, use fallback
            if not validated_recommendations:
                logger.warning("⚠️ [recommendations] No valid recommendations found, using fallback")
                return _get_fallback_recommendations(books_data, user_preferences)

            return recommendations
        except json.JSONDecodeError:
            logger.warning("⚠️ [recommendations] Failed to parse LLM response")
            return _get_fallback_recommendations(books_data, user_preferences)

    except Exception as e:
        logger.error("❌ [recommendations] Generation failed: %s", e)
        return _get_fallback_recommendations(books_data, user_preferences)


def _format_recommendation_response(
    recommendations: Dict[str, Any], user_preferences: Dict[str, Any]
) -> str:
    """Format recommendations into a user-friendly response."""
    try:
        recs = recommendations.get("recommendations", [])
        if not recs:
            return "I couldn't find suitable recommendations right now. Try being more specific about what you'd like to read!"

        response_parts = [
            "Here are my personalized recommendations for you:\n"
        ]

        for i, rec in enumerate(recs[:5], 1):  # Max 5 recommendations
            title = rec.get("title", "Unknown Title")
            author = rec.get("author", "Unknown Author")
            reasoning = rec.get("reasoning", "Great choice for you")

            response_parts.append(f"{i}. **{title}** by {author}")
            response_parts.append(f"   _{reasoning}_\n")

        # Add strategy explanation
        strategy = recommendations.get("summary", "")
        if strategy:
            response_parts.append(f"My recommendation strategy: {strategy}")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error("❌ [format_response] Failed: %s", e)
        return "Here are some books you might enjoy! (Formatting error occurred)"


def _validate_recommendations(
    recommendations: List[Dict[str, Any]],
    available_books: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate that all recommendations exist in the available books database"""
    valid_recommendations = []

    # Create lookup set for faster matching
    available_titles = set()
    title_to_book = {}

    for book in available_books:
        title = book.get("title", "").strip()
        author = book.get("author", "").strip()
        if title:
            # Create normalized lookup key
            lookup_key = f"{title.lower()}|{author.lower()}"
            available_titles.add(lookup_key)
            title_to_book[lookup_key] = book

    logger.info(f"🔍 [validation] Checking {len(recommendations)} recommendations against {len(available_titles)} available books")

    for rec in recommendations:
        rec_title = rec.get("title", "").strip()
        rec_author = rec.get("author", "").strip()

        if not rec_title:
            logger.warning(f"⚠️ [validation] Skipping recommendation with empty title")
            continue

        # Create normalized lookup key for recommendation
        lookup_key = f"{rec_title.lower()}|{rec_author.lower()}"

        if lookup_key in available_titles:
            # Valid recommendation - use exact data from database
            original_book = title_to_book[lookup_key]
            validated_rec = {
                "title": original_book.get("title", rec_title),
                "author": original_book.get("author", rec_author),
                "genre": original_book.get("primary_genre", rec.get("genre", "Unknown")),
                "reasoning": rec.get("reasoning", "Good match from our collection"),
                "confidence": rec.get("confidence", 0.8)
            }
            valid_recommendations.append(validated_rec)
            logger.info(f"✅ [validation] Valid: '{rec_title}' by {rec_author}")
        else:
            logger.warning(f"❌ [validation] Invalid recommendation: '{rec_title}' by {rec_author} - NOT in database")

    logger.info(f"📊 [validation] Validated {len(valid_recommendations)}/{len(recommendations)} recommendations")
    return valid_recommendations


def _get_default_preferences() -> Dict[str, Any]:
    """Return default preferences when extraction fails."""
    return {
        "likes": {
            "genres": [],
            "authors": [],
            "themes": [],
            "book_types": []
        },
        "dislikes": {
            "genres": [],
            "authors": [],
            "themes": []
        },
        "context": {
            "mood": "general reading",
            "situation": "leisure",
            "goal": "entertainment"
        },
        "recommendation_type": "discovery"
    }


def _get_fallback_recommendations(
    books_data: List[Dict[str, Any]], user_preferences: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate intelligent fallback recommendations when LLM fails."""
    try:
        logger.info("🔄 [fallback] Generating fallback recommendations from %d books", len(books_data))

        # Get user preference data for scoring
        liked_genres = set()
        liked_authors = set()
        disliked_genres = set()

        if user_preferences:
            likes = user_preferences.get("likes", {})
            dislikes = user_preferences.get("dislikes", {})

            liked_genres = set(g.lower() for g in likes.get("genres", []))
            liked_authors = set(a.lower() for a in likes.get("authors", []))
            disliked_genres = set(g.lower() for g in dislikes.get("genres", []))

        # Score and rank books
        scored_books = []
        for book in books_data[:50]:  # Consider more books for better selection
            score = _score_book_for_preferences(book, liked_genres, liked_authors, disliked_genres)
            if score > 0:  # Only include books with positive score
                scored_books.append((book, score))

        # Sort by score (descending) and ensure diversity
        scored_books.sort(key=lambda x: x[1], reverse=True)

        # Select diverse recommendations
        recommendations = []
        seen_genres = set()
        seen_authors = set()

        for book, score in scored_books:
            if len(recommendations) >= 5:
                break

            genre = book.get("primary_genre", "Unknown").lower()
            author = book.get("author", "Unknown").lower()
            title = book.get("title", "Unknown")

            # Prioritize diversity while maintaining quality
            if len(recommendations) < 3 or genre not in seen_genres:
                reasoning = _generate_fallback_reasoning(book, score, user_preferences)

                recommendations.append({
                    "title": title,
                    "author": book.get("author", "Unknown"),
                    "genre": book.get("primary_genre", "Unknown"),
                    "reasoning": reasoning,
                    "confidence": min(0.8, 0.5 + score * 0.3)  # Scale confidence based on score
                })

                seen_genres.add(genre)
                seen_authors.add(author)

        # If still no recommendations, get any available books
        if not recommendations and books_data:
            for book in books_data[:3]:
                recommendations.append({
                    "title": book.get("title", "Unknown"),
                    "author": book.get("author", "Unknown"),
                    "genre": book.get("primary_genre", "Unknown"),
                    "reasoning": "Available book from our collection",
                    "confidence": 0.5
                })

        logger.info("📚 [fallback] Generated %d fallback recommendations", len(recommendations))

        return {
            "recommendations": recommendations,
            "summary": "Curated selection from available books using preference analysis"
        }

    except Exception as e:
        logger.error("❌ [fallback_recommendations] Failed: %s", e)
        return {
            "recommendations": [],
            "summary": "Unable to generate recommendations"
        }


def _score_book_for_preferences(
    book: Dict[str, Any],
    liked_genres: set,
    liked_authors: set,
    disliked_genres: set
) -> float:
    """Score a book based on user preferences"""
    score = 0.5  # Base score

    # Check genre preferences
    primary_genre = book.get("primary_genre", "").lower()
    if primary_genre in liked_genres:
        score += 0.4
    elif primary_genre in disliked_genres:
        score -= 0.3

    # Check author preferences
    author = book.get("author", "").lower()
    if author in liked_authors:
        score += 0.3

    # Bonus for books with summaries and topics
    if book.get("summary"):
        score += 0.1
    if book.get("main_topics"):
        score += 0.1

    return max(0, score)  # Don't return negative scores


def _generate_fallback_reasoning(
    book: Dict[str, Any],
    score: float,
    user_preferences: Dict[str, Any]
) -> str:
    """Generate reasoning for fallback recommendation"""
    try:
        genre = book.get("primary_genre", "Unknown")
        author = book.get("author", "Unknown")

        if user_preferences and score > 0.7:
            return f"Good match for your preferences - {genre} genre from our collection"
        elif book.get("summary"):
            return f"Interesting {genre.lower()} book with detailed content"
        else:
            return f"Quality {genre.lower()} book from our collection"

    except:
        return "Selected book from our collection"