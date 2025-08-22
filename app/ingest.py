import os, uuid, hashlib, json, logging
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .settings import CHROMA_DIR, OPENAI_MODEL_EMBED, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS
from .loaders import load_text_from_file
from .metadata import build_master_meta
from .metadata_llm import extract_metadata_llm
from .utils_isbn import normalize_isbn
from .duplicate_detection import detect_duplicates, should_skip_ingestion, calculate_file_hash
from .file_hash_store import get_file_hash_store

logger = logging.getLogger(__name__)

def embeddings():
    return OpenAIEmbeddings(model=OPENAI_MODEL_EMBED, api_key=OPENAI_API_KEY)

def books_store():
    return Chroma(collection_name="books", persist_directory=CHROMA_DIR, embedding_function=embeddings())

def content_store():
    return Chroma(collection_name="content", persist_directory=CHROMA_DIR, embedding_function=embeddings())

def _stable_id(prefix: str, seed: str) -> str:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return f"{prefix}:{h[:16]}"

def split_text_sampled(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, max_chunks=MAX_CHUNKS) -> List[str]:
    logger.info(f"🔧 Splitting {len(text or '')} chars with chunk_size={chunk_size}, overlap={chunk_overlap}, max_chunks={max_chunks}")
    
    raw_chunks = []
    n = len(text or "")
    if n <= 0:
        logger.info("⚠️ Empty text, returning empty chunks")
        return []
    
    # Fix potential infinite loop: ensure chunk_overlap < chunk_size
    if chunk_overlap >= chunk_size:
        logger.warning(f"⚠️ chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}), adjusting overlap")
        chunk_overlap = max(0, chunk_size - 1)
    
    start = 0
    iteration = 0
    while start < n:
        iteration += 1
        if iteration > 10000:  # Safety break
            logger.error(f"❌ Infinite loop detected! Breaking at iteration {iteration}")
            break
            
        end = min(n, start + chunk_size)
        raw_chunks.append(text[start:end])
        
        # Move start forward, ensuring progress
        new_start = end - chunk_overlap
        if new_start <= start:  # Prevent infinite loop
            new_start = start + max(1, chunk_size - chunk_overlap)
        start = new_start
        
        if start >= n: 
            break
            
        if iteration % 100 == 0:
            logger.info(f"🔄 Processing chunk {iteration}, position {start}/{n}")
    
    logger.info(f"✅ Created {len(raw_chunks)} raw chunks")
    
    if len(raw_chunks) <= max_chunks:
        return raw_chunks
    
    logger.info(f"📉 Sampling down from {len(raw_chunks)} to {max_chunks} chunks")
    # sample head/middle/tail
    keep = max_chunks
    head_n = max(keep // 3, 1)
    tail_n = max(keep // 3, 1)
    mid_n = keep - head_n - tail_n
    head = raw_chunks[:head_n]
    tail = raw_chunks[-tail_n:] if tail_n > 0 else []
    mid = []
    if mid_n > 0:
        region = raw_chunks[head_n: len(raw_chunks) - tail_n]
        step = max(len(region) // mid_n, 1)
        mid = [region[i] for i in range(0, len(region), step)][:mid_n]
    
    final_chunks = head + mid + tail
    logger.info(f"✅ Final sampled chunks: {len(final_chunks)} (head:{len(head)}, mid:{len(mid)}, tail:{len(tail)})")
    return final_chunks

def _scalarize_meta(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
    return out

def ingest_one(file_path: str, meta_json: Dict | None = None, force_ingest: bool = False) -> Dict:
    """
    Ingest a single document with multi-level duplicate detection
    
    Args:
        file_path: Path to the file to ingest
        meta_json: Optional metadata override
        force_ingest: If True, skip duplicate detection and force ingestion
    
    Returns:
        Dict with ingestion result and duplicate detection info
    """
    logger.info(f"📖 [ingest.py] Starting ingestion: {file_path}")
    logger.info(f"📖 Loading text from file: {file_path}")
    full_text, file_type = load_text_from_file(file_path)
    logger.info(f"✅ Text loaded. Type: {file_type}, Length: {len(full_text)} chars")
    
    base = dict(meta_json or {})
    base.setdefault("file_type", file_type)
    base.setdefault("document_id", str(uuid.uuid4()))
    base.setdefault("full_text", full_text)

    prefer_llm = meta_json is None or bool(meta_json.get("prefer_llm", True))
    if prefer_llm:
        logger.info("🧠 Extracting metadata using LLM (OpenAI API call)...")
        llm_meta = extract_metadata_llm(full_text)
        base = {**llm_meta, **base}
        logger.info(f"✅ LLM metadata extracted: {llm_meta.get('title', 'Unknown title')}")

    logger.info("🔍 Processing ISBN...")
    raw_isbn = base.get("isbn13") or base.get("isbn") or base.get("isbn10")
    if raw_isbn and not base.get("isbn13"):
        norm = normalize_isbn(raw_isbn) or {}
        base["isbn13"] = norm.get("isbn13")
        base["isbn10"] = norm.get("isbn10")

    # DUPLICATE DETECTION - Multi-level checks
    duplicate_results = []
    if not force_ingest:
        logger.info("🛡️ Running multi-level duplicate detection...")
        hash_store = get_file_hash_store()
        file_hash_mappings = hash_store.get_all_hashes()
        
        duplicate_results = detect_duplicates(file_path, base, full_text, file_hash_mappings)
        should_skip, skip_reason = should_skip_ingestion(duplicate_results)
        
        if should_skip:
            logger.error(f"🚨 INGESTION BLOCKED: {skip_reason}")
            
            # Find the duplicate that caused the block
            duplicate_result = next((r for r in duplicate_results if r.is_duplicate and r.confidence >= 0.8), None)
            
            return {
                "status": "duplicate_detected",
                "message": f"Document rejected: {skip_reason}",
                "duplicate_level": duplicate_result.level if duplicate_result else "unknown",
                "duplicate_confidence": duplicate_result.confidence if duplicate_result else 0.0,
                "existing_document_id": duplicate_result.existing_doc_id if duplicate_result else None,
                "duplicate_checks": [{"level": r.level, "is_duplicate": r.is_duplicate, "reason": r.reason, "confidence": r.confidence} for r in duplicate_results],
                "file_path": file_path,
                "detected_title": base.get("title", "Unknown"),
                "detected_author": base.get("author", "Unknown")
            }
    else:
        logger.warning("⚠️ FORCE INGESTION: Skipping duplicate detection")

    logger.info("📊 Building master metadata...")
    master_meta = build_master_meta(base)
    
    # Build enriched master_text for better semantic search - AFTER all metadata is complete
    logger.info("🔍 Creating enriched master_text for semantic search...")
    logger.info(f"📋 Available metadata: title='{master_meta.get('title')}', author='{master_meta.get('author')}', year={master_meta.get('year')}, genre='{master_meta.get('primary_genre')}'")
    
    # Start with title and full summary (no truncation)
    parts = [master_meta["title"]]
    logger.info(f"📝 Step 1 - Title: Added '{master_meta['title']}'")
    
    if master_meta.get("summary"):
        parts.append(master_meta["summary"])  # Full summary, no [:400] truncation
        logger.info(f"📝 Step 2 - Summary: Added summary ({len(master_meta['summary'])} chars)")
    else:
        logger.warning("⚠️ Step 2 - Summary: NO SUMMARY FOUND!")
    
    # Add author for author-based searches
    if master_meta.get("author"):
        author_part = f"Author: {master_meta['author']}"
        parts.append(author_part)
        logger.info(f"📝 Step 3 - Author: Added '{author_part}'")
    else:
        logger.warning("⚠️ Step 3 - Author: NO AUTHOR FOUND!")
    
    # Add year for year-based searches  
    if master_meta.get("year"):
        year_part = f"Year: {master_meta['year']}"
        parts.append(year_part)
        logger.info(f"📝 Step 4 - Year: Added '{year_part}'")
    else:
        logger.warning("⚠️ Step 4 - Year: NO YEAR FOUND!")
    
    # Add genres for genre-based searches
    genres = []
    if master_meta.get("primary_genre"):
        genres.append(master_meta["primary_genre"])
        logger.info(f"📝 Step 5a - Primary genre: '{master_meta['primary_genre']}'")
    
    if master_meta.get("secondary_genres"):
        secondary = master_meta["secondary_genres"]
        logger.info(f"📝 Step 5b - Secondary genres raw: {secondary} (type: {type(secondary)})")
        if isinstance(secondary, list):
            genres.extend(secondary)
            logger.info(f"📝 Step 5b - Added list: {secondary}")
        elif isinstance(secondary, str):
            try:
                import json
                parsed = json.loads(secondary)
                if isinstance(parsed, list):
                    genres.extend(parsed)
                    logger.info(f"📝 Step 5b - Parsed JSON list: {parsed}")
                else:
                    genres.append(str(secondary))
                    logger.info(f"📝 Step 5b - Added string as-is: {secondary}")
            except:
                genres.append(str(secondary))
                logger.info(f"📝 Step 5b - JSON parse failed, added as string: {secondary}")
        else:
            genres.append(str(secondary))
            logger.info(f"📝 Step 5b - Added other type as string: {secondary}")
    
    if genres:
        genre_part = f"Genre: {', '.join(genres)}"
        parts.append(genre_part)
        logger.info(f"📝 Step 5 - Genres: Added '{genre_part}'")
    else:
        logger.warning("⚠️ Step 5 - Genres: NO GENRES FOUND!")
    
    # Add topics for topic-based searches
    topics = []
    if master_meta.get("main_topics"):
        main_topics = master_meta["main_topics"]
        logger.info(f"📝 Step 6a - Main topics raw: {main_topics} (type: {type(main_topics)})")
        if isinstance(main_topics, list):
            topics.extend(main_topics)
        elif isinstance(main_topics, str):
            try:
                import json
                parsed = json.loads(main_topics)
                if isinstance(parsed, list):
                    topics.extend(parsed)
                else:
                    topics.append(str(main_topics))
            except:
                topics.append(str(main_topics))
        else:
            topics.append(str(main_topics))
    
    if master_meta.get("mentioned_topics"):
        mentioned = master_meta["mentioned_topics"]
        logger.info(f"📝 Step 6b - Mentioned topics raw: {mentioned} (type: {type(mentioned)})")
        if isinstance(mentioned, list):
            topics.extend(mentioned)
        elif isinstance(mentioned, str):
            try:
                import json
                parsed = json.loads(mentioned)
                if isinstance(parsed, list):
                    topics.extend(parsed)
                else:
                    topics.append(str(mentioned))
            except:
                topics.append(str(mentioned))
        else:
            topics.append(str(mentioned))
    
    if topics:
        topics_part = f"Topics: {', '.join(topics[:10])}"  # Limit to first 10 topics to avoid too long text
        parts.append(topics_part)
        logger.info(f"📝 Step 6 - Topics: Added '{topics_part}' (from {len(topics)} total topics)")
    else:
        logger.warning("⚠️ Step 6 - Topics: NO TOPICS FOUND!")
    
    master_text = " — ".join(parts)
    logger.info(f"🔧 All parts before joining: {parts}")
    logger.info(f"✅ Enriched master_text created ({len(master_text)} chars):")
    logger.info(f"📄 FULL MASTER TEXT: {master_text}")
    
    # Log what will be sent to Chroma
    logger.info(f"📤 About to send to Chroma books store:")
    logger.info(f"    - Text length: {len(master_text)} chars")
    logger.info(f"    - Text preview: {master_text[:100]}...")
    logger.info(f"    - Book ID: {_stable_id('book', master_meta['document_id'])}")

    logger.info("📚 Adding book to books vector store...")
    bstore = books_store()
    book_id = _stable_id("book", master_meta["document_id"])
    
    # Final validation before sending to Chroma
    scalarized_meta = _scalarize_meta(master_meta)
    logger.info(f"🔍 FINAL VALIDATION before Chroma:")
    logger.info(f"    📋 Book ID: {book_id}")
    logger.info(f"    📄 Master text length: {len(master_text)} chars")
    logger.info(f"    🏷️ is_master_chunk: {scalarized_meta.get('is_master_chunk')}")
    logger.info(f"    📖 Title: {scalarized_meta.get('title')}")
    logger.info(f"    👤 Author: {scalarized_meta.get('author')}")
    logger.info(f"    🎭 Genre: {scalarized_meta.get('primary_genre')}")
    logger.info(f"    📅 Year: {scalarized_meta.get('year')}")
    logger.info(f"    📄 Master text preview: {master_text[:150]}...")
    
    bstore.add_texts(texts=[master_text], metadatas=[scalarized_meta], ids=[book_id])
    logger.info(f"✅ Book added to store with ID: {book_id}")
    
    # Verify what was actually stored
    logger.info(f"🔍 VERIFICATION: Reading back from Chroma...")
    try:
        coll = bstore._collection
        stored_data = coll.get(ids=[book_id], include=["documents", "metadatas"])
        if stored_data["documents"]:
            stored_text = stored_data["documents"][0]
            stored_meta = stored_data["metadatas"][0] if stored_data["metadatas"] else {}
            logger.info(f"    ✅ Successfully stored and retrieved")
            logger.info(f"    📄 Stored text length: {len(stored_text)} chars")
            logger.info(f"    📄 Stored text preview: {stored_text[:150]}...")
            logger.info(f"    🏷️ Stored is_master_chunk: {stored_meta.get('is_master_chunk')}")
        else:
            logger.error(f"    ❌ Failed to retrieve stored document!")
    except Exception as e:
        logger.error(f"    ❌ Verification failed: {e}")

    logger.info("✂️ Splitting text into chunks...")
    cstore = content_store()
    max_chunks = int(base.get("max_chunks", MAX_CHUNKS))
    chunks = split_text_sampled(full_text, max_chunks=max_chunks)
    logger.info(f"✅ Text split into {len(chunks)} chunks (max: {max_chunks})")
    
    if chunks:
        logger.info("💾 Adding chunks to content vector store...")
        c_ids, c_metas = [], []
        for i, ch in enumerate(chunks):
            cid = f"chunk:{master_meta['document_id']}:{i}"
            meta = {"document_id": master_meta["document_id"], "title": master_meta["title"], "author": master_meta.get("author"), "language": master_meta.get("language"), "idx": i}
            c_ids.append(cid); c_metas.append(_scalarize_meta(meta))
        cstore.add_texts(texts=chunks, metadatas=c_metas, ids=c_ids)
        logger.info(f"✅ All {len(chunks)} chunks added to content store")
    
    # Store file hash for future duplicate detection
    if not force_ingest:
        logger.info("💾 Storing file hash for duplicate detection...")
        file_hash = calculate_file_hash(file_path)
        if file_hash:
            hash_store = get_file_hash_store()
            hash_store.add_hash(master_meta["document_id"], file_hash, file_path)

    # Clear BM25 cache since we added new documents
    from .retrievers import clear_bm25_cache
    clear_bm25_cache()
    
    logger.info("✨ Ingest process completed successfully!")
    
    # Return success result
    return {
        "status": "success", 
        "message": f"Document '{master_meta.get('title', 'Unknown')}' ingested successfully",
        "book_id": book_id, 
        "document_id": master_meta["document_id"],
        "chunks": len(chunks), 
        "metadata": {
            "title": master_meta.get("title"),
            "author": master_meta.get("author"),
            "language": master_meta.get("language"),
            "primary_genre": master_meta.get("primary_genre"),
            "isbn13": master_meta.get("isbn13")
        },
        "duplicate_checks": [{"level": r.level, "is_duplicate": r.is_duplicate, "reason": r.reason, "confidence": r.confidence} for r in duplicate_results] if duplicate_results else [],
        "file_path": file_path
    }

def ingest_batch(paths: List[str], metas: List[Dict] | None = None) -> List[Dict]:
    results, metas = [], (metas or [None]*len(paths))
    for p, m in zip(paths, metas): results.append(ingest_one(p, m))
    return results
