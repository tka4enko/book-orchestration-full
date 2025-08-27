import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from .retrievers import books_store, content_store
from .metadata import canon
from .utils_isbn import normalize_isbn

logger = logging.getLogger(__name__)

class DuplicateDetectionResult:
    def __init__(self, is_duplicate: bool, level: str, reason: str, confidence: float, existing_doc_id: Optional[str] = None):
        self.is_duplicate = is_duplicate
        self.level = level  # "file_hash", "isbn", "metadata", "content_similarity"
        self.reason = reason
        self.confidence = confidence  # 0.0 to 1.0
        self.existing_doc_id = existing_doc_id
    
    def __str__(self):
        return f"Level {self.level}: {'DUPLICATE' if self.is_duplicate else 'UNIQUE'} ({self.confidence:.3f}) - {self.reason}"

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file contents"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to calculate file hash: {e}")
        return ""

def calculate_content_hash(text: str) -> str:
    """Calculate SHA-256 hash of text content"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def level1_file_hash_check(file_path: str, file_hash_store: Dict[str, str]) -> DuplicateDetectionResult:
    """Level 1: Check if exact same file was uploaded before by file hash"""
    logger.info("🔍 Level 1: File hash duplicate detection...")
    
    file_hash = calculate_file_hash(file_path)
    if not file_hash:
        return DuplicateDetectionResult(False, "file_hash", "Could not calculate file hash", 0.0)
    
    # Check if hash exists in our store
    for doc_id, stored_hash in file_hash_store.items():
        if stored_hash == file_hash:
            logger.warning(f"🚨 DUPLICATE DETECTED: Exact file hash match")
            return DuplicateDetectionResult(True, "file_hash", f"Identical file already exists", 1.0, doc_id)
    
    logger.info("✅ Level 1: File hash is unique")
    return DuplicateDetectionResult(False, "file_hash", "File hash is unique", 1.0)

def level2_isbn_check(base_metadata: Dict) -> DuplicateDetectionResult:
    """Level 2: Check ISBN duplicates in existing books"""
    logger.info("🔍 Level 2: ISBN duplicate detection...")
    
    # Get ISBN from metadata
    isbn = base_metadata.get("isbn")
    if not isbn:
        # Fallback to old format for backward compatibility
        raw_isbn = base_metadata.get("isbn13") or base_metadata.get("isbn10")
        if raw_isbn:
            norm = normalize_isbn(raw_isbn)
            if norm:
                isbn = norm.get("isbn")
    
    if not isbn:
        logger.info("ℹ️ Level 2: No ISBN found, skipping ISBN check")
        return DuplicateDetectionResult(False, "isbn", "No ISBN to check", 0.0)
    
    # Search existing books by ISBN
    books = books_store()
    try:
        # Search by ISBN in Chroma
        existing = books._collection.get(where={"isbn": isbn}, include=["metadatas"])
        if existing and existing.get("metadatas"):
            existing_meta = existing["metadatas"][0]
            existing_doc_id = existing_meta.get("document_id")
            existing_title = existing_meta.get("title", "Unknown")
            
            logger.warning(f"🚨 DUPLICATE DETECTED: ISBN {isbn} already exists")
            logger.warning(f"    Existing book: '{existing_title}' (doc_id: {existing_doc_id})")
            return DuplicateDetectionResult(True, "isbn", f"ISBN {isbn} already exists for '{existing_title}'", 1.0, existing_doc_id)
        
        logger.info(f"✅ Level 2: ISBN {isbn} is unique")
        return DuplicateDetectionResult(False, "isbn", f"ISBN {isbn} is unique", 1.0)
        
    except Exception as e:
        logger.warning(f"⚠️ Level 2: ISBN check failed: {e}")
        return DuplicateDetectionResult(False, "isbn", f"ISBN check failed: {e}", 0.0)

def level3_metadata_matching(base_metadata: Dict) -> DuplicateDetectionResult:
    """Level 3: Check title + author combination"""
    logger.info("🔍 Level 3: Metadata duplicate detection...")
    
    title = (base_metadata.get("title") or "").strip()
    author = (base_metadata.get("author") or "").strip()
    
    if not title or not author:
        logger.info("ℹ️ Level 3: Missing title or author, skipping metadata check")
        return DuplicateDetectionResult(False, "metadata", "Missing title or author", 0.0)
    
    # Canonicalize for comparison
    title_canon = canon(title)
    author_canon = canon(author)
    
    logger.info(f"    Searching for: '{title}' by '{author}'")
    logger.info(f"    Canonical: '{title_canon}' by '{author_canon}'")
    
    # Search existing books
    books = books_store()
    try:
        # Get all books and check manually (Chroma doesn't support complex where queries easily)
        all_books = books._collection.get(include=["metadatas"])
        
        if all_books and all_books.get("metadatas"):
            for existing_meta in all_books["metadatas"]:
                existing_title = existing_meta.get("title", "")
                existing_author = existing_meta.get("author", "")
                existing_doc_id = existing_meta.get("document_id")
                
                # Skip if missing data
                if not existing_title or not existing_author:
                    continue
                
                # Canonicalize existing
                existing_title_canon = canon(existing_title)
                existing_author_canon = canon(existing_author)
                
                # Check exact match
                if title_canon == existing_title_canon and author_canon == existing_author_canon:
                    logger.warning(f"🚨 DUPLICATE DETECTED: Exact title+author match")
                    logger.warning(f"    Existing: '{existing_title}' by '{existing_author}' (doc_id: {existing_doc_id})")
                    return DuplicateDetectionResult(True, "metadata", f"Exact match: '{existing_title}' by '{existing_author}'", 1.0, existing_doc_id)
                
                # Check very similar (fuzzy matching)
                title_similarity = _calculate_similarity(title_canon, existing_title_canon)
                author_similarity = _calculate_similarity(author_canon, existing_author_canon)
                
                # If both title and author are very similar (>90%), consider it a duplicate
                if title_similarity > 0.9 and author_similarity > 0.9:
                    confidence = (title_similarity + author_similarity) / 2
                    logger.warning(f"🚨 POTENTIAL DUPLICATE: Very similar title+author")
                    logger.warning(f"    Existing: '{existing_title}' by '{existing_author}' (similarity: {confidence:.3f})")
                    return DuplicateDetectionResult(True, "metadata", f"Very similar: '{existing_title}' by '{existing_author}' (similarity: {confidence:.3f})", confidence, existing_doc_id)
        
        logger.info("✅ Level 3: Title+Author combination is unique")
        return DuplicateDetectionResult(False, "metadata", "Title+Author combination is unique", 1.0)
        
    except Exception as e:
        logger.warning(f"⚠️ Level 3: Metadata check failed: {e}")
        return DuplicateDetectionResult(False, "metadata", f"Metadata check failed: {e}", 0.0)

def level4_content_similarity(full_text: str, content_hash_threshold: float = 0.95) -> DuplicateDetectionResult:
    """Level 4: Check content similarity using text hashes and sampling"""
    logger.info("🔍 Level 4: Content similarity detection...")
    
    if not full_text or len(full_text) < 100:
        logger.info("ℹ️ Level 4: Text too short for content similarity check")
        return DuplicateDetectionResult(False, "content_similarity", "Text too short", 0.0)
    
    # Calculate content hash
    content_hash = calculate_content_hash(full_text)
    content_size = len(full_text)
    
    # Get sample from text for comparison (first, middle, last parts)
    text_sample = _get_text_sample(full_text)
    
    logger.info(f"    Content hash: {content_hash[:16]}...")
    logger.info(f"    Content size: {content_size} chars")
    
    # Search existing content
    content = content_store()
    try:
        # Get all content chunks grouped by document
        all_content = content._collection.get(include=["documents", "metadatas"])
        
        if not all_content or not all_content.get("documents"):
            logger.info("✅ Level 4: No existing content to compare")
            return DuplicateDetectionResult(False, "content_similarity", "No existing content", 1.0)
        
        # Group content by document_id
        doc_contents = {}
        for doc, meta in zip(all_content["documents"], all_content["metadatas"]):
            doc_id = meta.get("document_id")
            title = meta.get("title", "Unknown")
            if doc_id:
                if doc_id not in doc_contents:
                    doc_contents[doc_id] = {"text": "", "title": title, "chunks": 0}
                doc_contents[doc_id]["text"] += doc + " "
                doc_contents[doc_id]["chunks"] += 1
        
        # Compare with each existing document
        for existing_doc_id, content_data in doc_contents.items():
            existing_text = content_data["text"].strip()
            existing_title = content_data["title"]
            
            if not existing_text or len(existing_text) < 100:
                continue
            
            # Calculate content similarity
            existing_hash = calculate_content_hash(existing_text)
            existing_sample = _get_text_sample(existing_text)
            
            # Exact content hash match
            if content_hash == existing_hash:
                logger.warning(f"🚨 DUPLICATE DETECTED: Identical content hash")
                logger.warning(f"    Existing: '{existing_title}' (doc_id: {existing_doc_id})")
                return DuplicateDetectionResult(True, "content_similarity", f"Identical content to '{existing_title}'", 1.0, existing_doc_id)
            
            # Sample similarity check
            sample_similarity = _calculate_similarity(text_sample, existing_sample)
            if sample_similarity > content_hash_threshold:
                logger.warning(f"🚨 POTENTIAL DUPLICATE: Very similar content")
                logger.warning(f"    Existing: '{existing_title}' (similarity: {sample_similarity:.3f})")
                return DuplicateDetectionResult(True, "content_similarity", f"Very similar content to '{existing_title}' (similarity: {sample_similarity:.3f})", sample_similarity, existing_doc_id)
        
        logger.info("✅ Level 4: Content is unique")
        return DuplicateDetectionResult(False, "content_similarity", "Content is unique", 1.0)
        
    except Exception as e:
        logger.warning(f"⚠️ Level 4: Content similarity check failed: {e}")
        return DuplicateDetectionResult(False, "content_similarity", f"Content check failed: {e}", 0.0)

def _get_text_sample(text: str, sample_size: int = 1000) -> str:
    """Get representative sample from text (beginning + middle + end)"""
    if len(text) <= sample_size:
        return text
    
    third = sample_size // 3
    beginning = text[:third]
    middle_start = len(text) // 2 - third // 2
    middle = text[middle_start:middle_start + third]
    end = text[-third:]
    
    return beginning + " " + middle + " " + end

def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple character-based similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Simple Jaccard similarity on character n-grams
    def get_ngrams(text: str, n: int = 3) -> set:
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_ngrams(text1.lower())
    ngrams2 = get_ngrams(text2.lower())
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union)

def detect_duplicates(file_path: str, base_metadata: Dict, full_text: str, file_hash_store: Dict[str, str]) -> List[DuplicateDetectionResult]:
    """Run all 4 levels of duplicate detection"""
    logger.info("🔍 [duplicate_detection.py] Running multi-level duplicate detection...")
    
    results = []
    
    # Level 1: File hash
    result1 = level1_file_hash_check(file_path, file_hash_store)
    results.append(result1)
    logger.info(f"    {result1}")
    
    # If Level 1 found duplicate, stop here
    if result1.is_duplicate:
        logger.warning("🚨 Stopping at Level 1 - exact file duplicate found")
        return results
    
    # Level 2: ISBN
    result2 = level2_isbn_check(base_metadata)
    results.append(result2)
    logger.info(f"    {result2}")
    
    # If Level 2 found duplicate with high confidence, stop
    if result2.is_duplicate and result2.confidence >= 0.9:
        logger.warning("🚨 Stopping at Level 2 - ISBN duplicate found")
        return results
    
    # Level 3: Metadata
    result3 = level3_metadata_matching(base_metadata)
    results.append(result3)
    logger.info(f"    {result3}")
    
    # If Level 3 found duplicate with high confidence, stop
    if result3.is_duplicate and result3.confidence >= 0.9:
        logger.warning("🚨 Stopping at Level 3 - metadata duplicate found")
        return results
    
    # Level 4: Content similarity
    result4 = level4_content_similarity(full_text)
    results.append(result4)
    logger.info(f"    {result4}")
    
    # Summary
    duplicates = [r for r in results if r.is_duplicate]
    if duplicates:
        highest_confidence = max(duplicates, key=lambda x: x.confidence)
        logger.warning(f"🚨 FINAL RESULT: DUPLICATE DETECTED at {highest_confidence.level} level (confidence: {highest_confidence.confidence:.3f})")
        logger.warning(f"    Reason: {highest_confidence.reason}")
    else:
        logger.info("✅ FINAL RESULT: Document is UNIQUE - passed all duplicate checks")
    
    return results

def should_skip_ingestion(results: List[DuplicateDetectionResult], confidence_threshold: float = 0.8) -> Tuple[bool, str]:
    """Decide whether to skip ingestion based on duplicate detection results"""
    duplicates = [r for r in results if r.is_duplicate and r.confidence >= confidence_threshold]
    
    if not duplicates:
        return False, "No high-confidence duplicates found"
    
    # Find the highest confidence duplicate
    highest = max(duplicates, key=lambda x: x.confidence)
    
    return True, f"Duplicate found at {highest.level} level: {highest.reason}"