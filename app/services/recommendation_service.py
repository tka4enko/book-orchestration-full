"""Smart recommendation orchestrator with context-aware personalized recommendations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from ..infra.settings import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from .vector_retriever import SimpleVectorRetriever

logger = logging.getLogger(__name__)

_retriever = SimpleVectorRetriever()
_recommendation_llm = ChatOpenAI(
    model=OPENAI_MODEL_CHAT,
    temperature=0.3,
    api_key=OPENAI_API_KEY,
    max_tokens=500,
)

# Recommendation prompts and configurations
PREFERENCE_EXTRACTION_PROMPT = """You are an intelligent conversation analyst. Extract what the user wants right now.

Analyze ONLY user messages. Ignore all assistant responses completely.

CONVERSATION HISTORY:
{chat_history}

CURRENT REQUEST: {current_message}

Extract the user's preferences from their current request. Each new request represents their current interests - do not combine with previous requests unless they explicitly say "more" or "also".

Return only what the user wants now as valid JSON:
{{
    "likes": {{
        "genres": [],
        "authors": [],
        "themes": [],
        "book_types": []
    }},
    "dislikes": {{
        "genres": [],
        "authors": [],
        "themes": []
    }},
    "context": {{
        "mood": "general reading",
        "situation": "leisure",
        "goal": "entertainment"
    }},
    "recommendation_type": "discovery"
}}
"""

SMART_RECOMMENDATION_PROMPT = """You are an expert book curator. Recommend books that match user preferences from the available database.

USER PREFERENCES:
{user_preferences}

AVAILABLE BOOKS IN DATABASE (you can ONLY recommend from this list):
{books_data}

BOOKS TO EXCLUDE (already seen/mentioned):
{exclude_books}

CRITICAL RULES:
1. Recommend books that match user preferences from the available database
2. DO NOT invent or hallucinate book titles or authors that are not in the provided list
3. Prioritize preference matching over quantity
4. Skip books that appear in the "exclude" list
5. BE HONEST: If no matching books are found, return empty recommendations list

STRICT MATCHING CRITERIA:
- For genre preferences: book's primary_genre or all_genres must contain exact match
- For theme preferences: book's topics must contain similar themes
- NO creative interpretation or forced connections
- If no exact/similar matches found → return empty array
- Quality over quantity - recommend only what truly fits
- Empty recommendations are acceptable and honest when appropriate

SELECTION CRITERIA:
- IF user has specific preferences → match preferred genres, authors, themes
- IF user has no specific preferences (empty arrays) → provide diverse discovery selection
- Avoid user's dislikes if specified
- Provide variety when possible
- Include reasoning based on ACTUAL user preferences (not invented ones)

Return JSON (use exact titles and authors from the provided database):
{{
    "recommendations": [
        {{
            "title": "EXACT_TITLE_FROM_DATABASE",
            "author": "EXACT_AUTHOR_FROM_DATABASE",
            "genre": "primary_genre_from_database",
            "reasoning": "Why this book was selected - reference book's actual qualities, NOT invented user preferences",
            "confidence": 0.9
        }}
    ],
    "summary": "Specific explanation of why these books were selected from our database based on ACTUAL user preferences or book qualities. For users with no preferences: 'Selected diverse high-quality books including classic literature and modern fiction with engaging themes.' For users with specific preferences: 'Selected sci-fi books that match your stated interest in space exploration and dystopian themes.' DO NOT mention authors not in the database."
}}

CRITICAL REASONING RULES:
1. Use the exact "title" and "author" fields from the database entries above
2. Pay attention to the "summary", "topics", and "all_genres" fields to make better matches with user preferences
3. In the "summary" field, NEVER mention authors like "Agatha Christie", "J.K. Rowling" or any other authors that are NOT in the provided database
4. Base your explanation ONLY on the actual books and authors present in the database list above

REASONING FOR EMPTY PREFERENCES:
- IF user preferences are empty/minimal → reasoning should focus on book's intrinsic qualities
- Example: "Classic dystopian novel exploring surveillance and freedom themes"
- NOT: "matches user's preference for dystopian fiction" (when user never said they prefer dystopian)
- Focus on: book quality, interesting themes, well-regarded author, diverse selection

REASONING FOR SPECIFIC PREFERENCES:
- IF user has clear preferences → reference those specific preferences
- Example: "Matches your interest in sci-fi and space exploration themes"
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
        # Step 1: Use provided user preferences (extracted in node_recommendations)
        preferences_start = time.time()
        if not user_preferences:
            logger.warning("⚠️ [smart_recommendations] No user preferences provided, using defaults")
            user_preferences = _get_default_preferences()
        preferences_time = time.time() - preferences_start

        # Step 2: Get available books data
        data_start = time.time()
        books_data = await _get_books_for_recommendations(user_preferences, exclude_books or set())
        data_time = time.time() - data_start

        # Step 3: Generate smart recommendations
        rec_start = time.time()
        logger.info("🎯 [smart_recommendations] Passing preferences to LLM: %s", user_preferences)
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


@traceable(name="extract_user_preferences")
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
            for i, entry in enumerate(recent_history):
                role = entry.get("role", "user")
                content = entry.get("content", "")
                # Add simple chronological marker
                timestamp = f"[{i+1:02d}]"
                history_lines.append(f"{timestamp} {role}: {content}")
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

@traceable(name="get_books_for_recommendations")
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


@traceable(name="generate_recommendations")
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

            # If no valid recommendations found, check if this was an honest empty response
            if not validated_recommendations:
                # Check if LLM honestly said there are no suitable books
                if _is_honest_empty_response(recommendations):
                    logger.info("✅ [recommendations] LLM honestly reported no suitable books available")
                    return recommendations  # Respect LLM's honest decision
                else:
                    logger.warning("⚠️ [recommendations] No valid recommendations found, FALLBACK DISABLED FOR TESTING")
                    return recommendations  # Return empty instead of fallback

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
            # Check if user had specific theme/genre preferences
            likes = user_preferences.get("likes", {})
            themes = likes.get("themes", [])
            genres = likes.get("genres", [])

            if themes or genres:
                criteria = []
                if themes:
                    criteria.extend(themes)
                if genres:
                    criteria.extend(genres)
                criteria_text = ", ".join(criteria)
                return f"I couldn't find any more {criteria_text} books in our database. Try asking for a different genre or theme!"
            else:
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

        # Add strategy explanation (but avoid showing hallucinated preferences)
        strategy = recommendations.get("summary", "")
        if strategy:
            # Check if user preferences are mostly empty to avoid showing hallucinated preferences
            likes = user_preferences.get("likes", {})
            has_real_preferences = (
                likes.get("genres") or
                likes.get("authors") or
                likes.get("themes") or
                likes.get("book_types")
            )

            if has_real_preferences:
                response_parts.append(f"My recommendation strategy: {strategy}")
            else:
                # For users with no explicit preferences, use neutral strategy text
                response_parts.append("My recommendation strategy: Selected diverse high-quality books with engaging themes and well-regarded authors.")

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
    """Return conservative default preferences - empty arrays to avoid false assumptions."""
    return {
        "likes": {
            "genres": [],  # Conservative: no assumed preferences
            "authors": [],  # Conservative: no assumed preferences
            "themes": [],  # Conservative: no assumed preferences
            "book_types": []  # Conservative: no assumed preferences
        },
        "dislikes": {
            "genres": [],  # Conservative: no assumed dislikes
            "authors": [],  # Conservative: no assumed dislikes
            "themes": []  # Conservative: no assumed dislikes
        },
        "context": {
            "mood": "general reading",  # Safe default
            "situation": "leisure",  # Safe default
            "goal": "entertainment"  # Safe default
        },
        "recommendation_type": "discovery"  # Discovery = exploration, safe default
    }


def _is_honest_empty_response(recommendations: Dict[str, Any]) -> bool:
    """Check if LLM honestly said there are no suitable books."""
    recs = recommendations.get("recommendations", [])
    summary = recommendations.get("summary", "").lower()

    # Empty recommendations + summary indicating honest limitation
    if not recs and any(phrase in summary for phrase in [
        "no books", "no more", "unfortunately", "not available",
        "no suitable", "no matching", "database", "sorry"
    ]):
        return True

    return False


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
            "summary": "Diverse selection from our book collection for discovery and exploration"
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