import os, uuid, hashlib, json, logging
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .settings import CHROMA_DIR, OPENAI_MODEL_EMBED, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS, DEBUG_INGEST_STAGES
from .loaders import load_text_from_file
from .metadata import build_master_meta
from .metadata_llm import extract_metadata_llm, extract_basic_metadata, enhance_metadata_from_chunks
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
        chunk_text = text[start:end]
        
        # Skip tiny chunks that are mostly overlap
        if len(chunk_text) < chunk_size // 3 and len(raw_chunks) > 0:
            logger.info(f"⚠️ Skipping tiny chunk of {len(chunk_text)} chars (< {chunk_size // 3})")
            break
            
        raw_chunks.append(chunk_text)
        
        # If we reached the end, stop
        if end >= n:
            break
        
        # Move start forward, ensuring progress
        new_start = end - chunk_overlap
        if new_start <= start:  # Prevent infinite loop
            new_start = start + max(1, chunk_size - chunk_overlap)
        start = new_start
            
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

    text_length = len(full_text or "")
    is_long_book = text_length > 9000
    
    if DEBUG_INGEST_STAGES:
        logger.info(f"🔬 DEBUG: Text length: {text_length}, Long book: {is_long_book}")
    
    if is_long_book:
        logger.info(f"📚 LONG BOOK DETECTED ({text_length} chars) - Using 3-stage analysis")
        if DEBUG_INGEST_STAGES:
            logger.info("🔬 DEBUG: Starting Stage 1 - Basic metadata extraction")
        logger.info("🧠 STAGE 1: Extracting basic metadata from start+end...")
        llm_meta = extract_basic_metadata(full_text)
        base = {**llm_meta, **base}
        logger.info(f"✅ Stage 1 completed: {llm_meta.get('title', 'Unknown title')}")
        if DEBUG_INGEST_STAGES:
            summary = llm_meta.get('summary') or ''
            logger.info(f"🔬 DEBUG: Stage 1 result - Title: {llm_meta.get('title')}, Author: {llm_meta.get('author')}, Summary length: {len(summary)}")
    else:
        logger.info(f"📖 SHORT BOOK ({text_length} chars) - Using single-stage analysis")
        if DEBUG_INGEST_STAGES:
            logger.info("🔬 DEBUG: Using single-stage LLM analysis")
        logger.info("🧠 Extracting metadata using LLM (OpenAI API call)...")
        llm_meta = extract_metadata_llm(full_text)
        base = {**llm_meta, **base}
        logger.info(f"✅ LLM metadata extracted: {llm_meta.get('title', 'Unknown title')}")
        if DEBUG_INGEST_STAGES:
            summary = llm_meta.get('summary') or ''
            logger.info(f"🔬 DEBUG: Single-stage result - Title: {llm_meta.get('title')}, Author: {llm_meta.get('author')}, Summary length: {len(summary)}")

    logger.info("🔍 Processing ISBN...")
    raw_isbn = base.get("isbn") or base.get("isbn13") or base.get("isbn10")
    if raw_isbn and not base.get("isbn"):
        norm = normalize_isbn(raw_isbn) or {}
        base["isbn"] = norm.get("isbn")

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
    if DEBUG_INGEST_STAGES:
        logger.info("🔬 DEBUG: Starting Stage 2 - Building master metadata and content chunks")
    master_meta = build_master_meta(base)
    if DEBUG_INGEST_STAGES:
        logger.info(f"🔬 DEBUG: Master metadata built - Document ID: {master_meta.get('document_id')}")
    
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
    
    # Add ISBN for ISBN-based searches
    if master_meta.get("isbn"):
        isbn_part = f"ISBN: {master_meta['isbn']}"
        parts.append(isbn_part)
        logger.info(f"📝 Step 6 - ISBN: Added '{isbn_part}'")
    else:
        logger.warning("⚠️ Step 6 - ISBN: NO ISBN FOUND!")

    # Add topics for topic-based searches (main + mentioned, with deduplication)
    main_topics_list = []
    mentioned_topics_list = []
    
    if master_meta.get("main_topics"):
        main_topics = master_meta["main_topics"]
        logger.info(f"📝 Step 7a - Main topics raw: {main_topics} (type: {type(main_topics)})")
        if isinstance(main_topics, list):
            main_topics_list.extend(main_topics)
        elif isinstance(main_topics, str):
            try:
                import json
                parsed = json.loads(main_topics)
                if isinstance(parsed, list):
                    main_topics_list.extend(parsed)
                else:
                    main_topics_list.append(str(main_topics))
            except:
                main_topics_list.append(str(main_topics))
        else:
            main_topics_list.append(str(main_topics))
    
    if master_meta.get("mentioned_topics"):
        mentioned = master_meta["mentioned_topics"]
        logger.info(f"📝 Step 7b - Mentioned topics raw: {mentioned} (type: {type(mentioned)})")
        if isinstance(mentioned, list):
            mentioned_topics_list.extend(mentioned)
        elif isinstance(mentioned, str):
            try:
                import json
                parsed = json.loads(mentioned)
                if isinstance(parsed, list):
                    mentioned_topics_list.extend(parsed)
                else:
                    mentioned_topics_list.append(str(mentioned))
            except:
                mentioned_topics_list.append(str(mentioned))
        else:
            mentioned_topics_list.append(str(mentioned))
    
    # Combine topics with deduplication (exact string matches only)
    main_topics_set = set(main_topics_list)
    mentioned_unique = [topic for topic in mentioned_topics_list if topic not in main_topics_set]
    
    # Create separate sections for main and mentioned topics
    if main_topics_list or mentioned_unique:
        topics_parts = []
        if main_topics_list:
            topics_parts.append(f"Topics: {', '.join(main_topics_list)}")
        if mentioned_unique:
            topics_parts.append(f"Mentioned topics: {', '.join(mentioned_unique)}")
        
        topics_part = " — ".join(topics_parts)
        parts.append(topics_part)
        logger.info(f"📝 Step 7 - Topics: Added '{topics_part}' (main: {len(main_topics_list)}, mentioned unique: {len(mentioned_unique)})")
    else:
        logger.warning("⚠️ Step 7 - Topics: NO TOPICS FOUND!")
    
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
        
        # STAGE 3: Enhanced metadata analysis for long books
        if is_long_book:
            if DEBUG_INGEST_STAGES:
                logger.info("🔬 DEBUG: Starting Stage 3 - Enhanced content analysis")
            logger.info("🧠 STAGE 3: Enhancing metadata from all chunks...")
            
            # Combine chunks for comprehensive analysis
            chunks_sample = chunks[:5]  # Use first 5 chunks to avoid token limits
            combined_chunks = " ".join(chunks_sample)
            
            logger.info(f"📄 Analyzing {len(chunks_sample)} chunks ({len(combined_chunks)} chars)")
            if DEBUG_INGEST_STAGES:
                logger.info(f"🔬 DEBUG: Chunks sample preview: {combined_chunks[:200]}...")
            
            enhanced_meta = enhance_metadata_from_chunks(combined_chunks)
            
            if enhanced_meta:
                logger.info("🔄 Updating master record with enhanced metadata...")
                if DEBUG_INGEST_STAGES:
                    logger.info(f"🔬 DEBUG: Enhanced metadata received - Summary: {bool(enhanced_meta.get('summary'))}, Genres: {enhanced_meta.get('primary_genre')}, Topics: {len(enhanced_meta.get('main_topics', []))}")
                
                # Update summary if we got a better one
                if enhanced_meta.get("summary"):
                    if DEBUG_INGEST_STAGES:
                        old_summary = master_meta.get('summary') or ''
                        logger.info(f"🔬 DEBUG: Updating summary from {len(old_summary)} to {len(enhanced_meta['summary'])} chars")
                    master_meta["summary"] = enhanced_meta["summary"]
                    logger.info(f"📝 Updated summary ({len(enhanced_meta['summary'])} chars)")
                
                # Merge genres (keep existing + add new)
                if enhanced_meta.get("primary_genre") and not master_meta.get("primary_genre"):
                    master_meta["primary_genre"] = enhanced_meta["primary_genre"]
                    logger.info(f"🎭 Updated primary genre: {enhanced_meta['primary_genre']}")
                
                # Merge secondary genres
                existing_genres = master_meta.get("secondary_genres") or []
                new_genres = enhanced_meta.get("secondary_genres") or []
                if isinstance(existing_genres, str):
                    existing_genres = []
                combined_genres = list(set(existing_genres + new_genres))
                if combined_genres:
                    master_meta["secondary_genres"] = combined_genres
                    logger.info(f"🎭 Updated secondary genres: {combined_genres}")
                
                # Merge topics
                existing_main = master_meta.get("main_topics") or []
                new_main = enhanced_meta.get("main_topics") or []
                if isinstance(existing_main, str):
                    existing_main = []
                combined_main = list(set(existing_main + new_main))
                if combined_main:
                    master_meta["main_topics"] = combined_main
                    logger.info(f"📚 Updated main topics: {combined_main[:5]}...")  # Show first 5
                
                existing_mentioned = master_meta.get("mentioned_topics") or []
                new_mentioned = enhanced_meta.get("mentioned_topics") or []
                if isinstance(existing_mentioned, str):
                    existing_mentioned = []
                combined_mentioned = list(set(existing_mentioned + new_mentioned))
                if combined_mentioned:
                    master_meta["mentioned_topics"] = combined_mentioned
                    logger.info(f"🏷️ Updated mentioned topics: {combined_mentioned[:5]}...")  # Show first 5
                
                # Rebuild master_text with enhanced metadata
                logger.info("🔧 Rebuilding master_text with enhanced metadata...")
                
                # Recreate enriched master_text (copy from earlier code)
                parts = [master_meta["title"]]
                
                if master_meta.get("summary"):
                    parts.append(master_meta["summary"])
                
                if master_meta.get("author"):
                    parts.append(f"Author: {master_meta['author']}")
                
                if master_meta.get("year"):
                    parts.append(f"Year: {master_meta['year']}")
                
                # Genres
                genres = []
                if master_meta.get("primary_genre"):
                    genres.append(master_meta["primary_genre"])
                if master_meta.get("secondary_genres"):
                    secondary = master_meta["secondary_genres"]
                    if isinstance(secondary, list):
                        genres.extend(secondary)
                if genres:
                    parts.append(f"Genre: {', '.join(genres)}")
                
                # ISBN
                if master_meta.get("isbn"):
                    parts.append(f"ISBN: {master_meta['isbn']}")
                
                # Topics with deduplication (same logic as regular ingest)
                topics_parts = []
                main_topics_list = master_meta.get("main_topics") or []
                if isinstance(main_topics_list, str):
                    try:
                        main_topics_list = json.loads(main_topics_list)
                    except:
                        main_topics_list = []
                if not isinstance(main_topics_list, list):
                    main_topics_list = []
                
                mentioned_topics_list = master_meta.get("mentioned_topics") or []
                if isinstance(mentioned_topics_list, str):
                    try:
                        mentioned_topics_list = json.loads(mentioned_topics_list)
                    except:
                        mentioned_topics_list = []
                if not isinstance(mentioned_topics_list, list):
                    mentioned_topics_list = []
                
                # Combine topics with deduplication (exact string matches only)
                main_topics_set = set(main_topics_list)
                mentioned_unique = [topic for topic in mentioned_topics_list if topic not in main_topics_set]
                
                # Create separate sections for main and mentioned topics
                if main_topics_list:
                    topics_parts.append(f"Topics: {', '.join(main_topics_list)}")
                if mentioned_unique:
                    topics_parts.append(f"Mentioned topics: {', '.join(mentioned_unique)}")
                
                if topics_parts:
                    topics_part = " — ".join(topics_parts)
                    parts.append(topics_part)
                
                enhanced_master_text = " — ".join(parts)
                
                # Update the master record in ChromaDB
                logger.info(f"📝 Updating master record with enhanced text ({len(enhanced_master_text)} chars)")
                if DEBUG_INGEST_STAGES:
                    logger.info(f"🔬 DEBUG: Final master_text preview: {enhanced_master_text[:300]}...")
                
                enhanced_scalarized_meta = _scalarize_meta(master_meta)
                enhanced_scalarized_meta["is_master_chunk"] = True
                
                bstore.add_texts(texts=[enhanced_master_text], metadatas=[enhanced_scalarized_meta], ids=[book_id])
                logger.info("✅ Master record updated with enhanced metadata")
                
                if DEBUG_INGEST_STAGES:
                    logger.info(f"🔬 DEBUG: Final metadata - Genres: {master_meta.get('primary_genre')}/{len(master_meta.get('secondary_genres', []))}, Topics: {len(master_meta.get('main_topics', []))}/{len(master_meta.get('mentioned_topics', []))}")
            
            logger.info("✅ Stage 3 completed - Enhanced analysis done")
    
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
        "content_preview": enhanced_master_text if (is_long_book and 'enhanced_master_text' in locals()) else master_text,  # What's stored in ChromaDB
        "isbn": master_meta.get("isbn"),  # Primary ISBN field
        "metadata": {
            "title": master_meta.get("title"),
            "author": master_meta.get("author"),
            "language": master_meta.get("language"),
            "primary_genre": master_meta.get("primary_genre"),
            "isbn": master_meta.get("isbn"),  # Primary ISBN
            "year": master_meta.get("year"),
            "summary": master_meta.get("summary")
        },
        "duplicate_checks": [{"level": r.level, "is_duplicate": r.is_duplicate, "reason": r.reason, "confidence": r.confidence} for r in duplicate_results] if duplicate_results else [],
        "file_path": file_path
    }

def ingest_batch(paths: List[str], metas: List[Dict] | None = None, force_ingest: bool = False) -> Dict:
    """
    Batch ingest multiple documents with comprehensive duplicate handling

    Args:
        paths: List of file paths to ingest
        metas: Optional list of metadata for each file
        force_ingest: If True, skip duplicate detection for all files

    Returns:
        Dict with batch summary: success_count, duplicate_count, error_count, results
    """
    logger.info(f"📦 [ingest.py] Starting batch ingestion of {len(paths)} files")

    results = []
    metas = metas or [None] * len(paths)

    success_count = 0
    duplicate_count = 0
    error_count = 0

    for i, (path, meta) in enumerate(zip(paths, metas)):
        logger.info(f"📄 [{i+1}/{len(paths)}] Processing: {path}")

        try:
            result = ingest_one(path, meta, force_ingest)
            results.append(result)

            # Count results by status
            if result.get("status") == "success":
                success_count += 1
                logger.info(f"✅ [{i+1}/{len(paths)}] SUCCESS: {result.get('metadata', {}).get('title', 'Unknown')}")
            elif result.get("status") == "duplicate_detected":
                duplicate_count += 1
                logger.warning(f"🚨 [{i+1}/{len(paths)}] DUPLICATE: {result.get('detected_title', 'Unknown')} - {result.get('message')}")
            else:
                error_count += 1
                logger.error(f"❌ [{i+1}/{len(paths)}] ERROR: {result.get('message', 'Unknown error')}")

        except Exception as e:
            error_count += 1
            error_result = {
                "status": "error",
                "message": f"Processing failed: {str(e)}",
                "file_path": path,
                "error_type": e.__class__.__name__
            }
            results.append(error_result)
            logger.error(f"💥 [{i+1}/{len(paths)}] EXCEPTION: {path} - {e}")

    # Summary
    total_files = len(paths)
    logger.info("=" * 50)
    logger.info(f"📦 BATCH INGESTION COMPLETED")
    logger.info(f"   Total files: {total_files}")
    logger.info(f"   ✅ Successful: {success_count}")
    logger.info(f"   🚨 Duplicates: {duplicate_count}")
    logger.info(f"   ❌ Errors: {error_count}")
    logger.info("=" * 50)

    return {
        "status": "batch_completed",
        "summary": {
            "total_files": total_files,
            "successful": success_count,
            "duplicates": duplicate_count,
            "errors": error_count
        },
        "results": results,
        "force_ingest": force_ingest
    }
