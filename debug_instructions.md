# Debug Reporter System

## Quick Start

1. **Enable Debug Reports**:
   ```bash
   # Add to your .env file:
   DEBUG_REPORTER_ENABLED=true
   DEBUG_LLM_ANALYSIS=true  # Optional: AI analysis of results
   ```

2. **Restart application**:
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Run queries** - debug reports will appear after each search

## Sample Debug Report

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
  2. ✅ Author+Title Fuzzy Match: (1 results) | query='fuzzy match validation'
     📊 title_similarity=95 | author_similarity=100

🎯 Final Results: ✅ 1 books found

🤖 Analysis: ✅ Perfect author+title detection. Hybrid search found relevant matches, fuzzy validation confirmed exact match. System working optimally for natural language queries.
================================================================================
```

## Configuration Options

| Setting | Values | Description |
|---------|--------|-------------|
| `DEBUG_REPORTER_ENABLED` | `true`/`false` | Enable/disable entire debug system |
| `DEBUG_LLM_ANALYSIS` | `true`/`false` | Enable AI analysis of search results |

## What Gets Tracked

- **Intent Detection**: What intent was detected and why
- **Search Steps**: Each search operation (BM25, vector, filters)
- **Filtering**: Predicate filters, negative filters, year ranges
- **Issues**: Conflicts, mismatches, failed searches
- **Performance**: Execution time, result counts
- **AI Analysis**: Smart insights about search effectiveness

## For Production

**Disable before production**:
```bash
# In .env file:
DEBUG_REPORTER_ENABLED=false
```

Or remove the environment variables entirely. The system has zero overhead when disabled.

## Troubleshooting

**No debug reports appearing?**
- Check `DEBUG_REPORTER_ENABLED=true` in .env
- Restart application after changing .env
- Verify queries are going through orchestrator.py

**LLM analysis not working?**
- Check `OPENAI_API_KEY` is set correctly
- Verify `DEBUG_LLM_ANALYSIS=true`
- Check logs for API errors