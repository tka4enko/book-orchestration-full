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
    master_text = master_meta["title"] if not master_meta.get("summary") else f"{master_meta['title']} — {master_meta['summary'][:400]}"

    logger.info("📚 Adding book to books vector store...")
    bstore = books_store()
    book_id = _stable_id("book", master_meta["document_id"])
    bstore.add_texts(texts=[master_text], metadatas=[_scalarize_meta(master_meta)], ids=[book_id])
    logger.info(f"✅ Book added to store with ID: {book_id}")

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
