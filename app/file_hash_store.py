import os
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class FileHashStore:
    """Simple file-based storage for tracking file hashes to prevent duplicate uploads"""
    
    def __init__(self, store_path: str = None):
        if store_path is None:
            from .settings import FILE_HASH_STORE_PATH
            store_path = FILE_HASH_STORE_PATH
        self.store_path = store_path
        self._hashes = {}
        self.load()
    
    def load(self):
        """Load hash store from file"""
        try:
            if os.path.exists(self.store_path):
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    self._hashes = json.load(f)
                logger.info(f"📁 Loaded {len(self._hashes)} file hashes from {self.store_path}")
            else:
                logger.info(f"📁 Creating new file hash store at {self.store_path}")
                self._hashes = {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load file hash store: {e}")
            self._hashes = {}
    
    def save(self):
        """Save hash store to file"""
        try:
            with open(self.store_path, 'w', encoding='utf-8') as f:
                json.dump(self._hashes, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved {len(self._hashes)} file hashes to {self.store_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save file hash store: {e}")
    
    def add_hash(self, doc_id: str, file_hash: str, file_path: str = ""):
        """Add a file hash for a document"""
        self._hashes[doc_id] = {
            "hash": file_hash,
            "file_path": file_path,
            "timestamp": import_timestamp()
        }
        self.save()
        logger.info(f"📝 Added hash for doc {doc_id}: {file_hash[:16]}...")
    
    def get_hash(self, doc_id: str) -> str:
        """Get file hash for a document"""
        entry = self._hashes.get(doc_id, {})
        return entry.get("hash", "")
    
    def get_all_hashes(self) -> Dict[str, str]:
        """Get all document_id -> file_hash mappings"""
        return {doc_id: entry["hash"] for doc_id, entry in self._hashes.items() if "hash" in entry}
    
    def find_by_hash(self, file_hash: str) -> str:
        """Find document ID by file hash"""
        for doc_id, entry in self._hashes.items():
            if entry.get("hash") == file_hash:
                return doc_id
        return ""
    
    def remove_hash(self, doc_id: str):
        """Remove a file hash"""
        if doc_id in self._hashes:
            del self._hashes[doc_id]
            self.save()
            logger.info(f"🗑️ Removed hash for doc {doc_id}")
    
    def cleanup_orphaned(self, valid_doc_ids: set):
        """Remove hashes for documents that no longer exist"""
        orphaned = []
        for doc_id in self._hashes:
            if doc_id not in valid_doc_ids:
                orphaned.append(doc_id)
        
        for doc_id in orphaned:
            del self._hashes[doc_id]
        
        if orphaned:
            self.save()
            logger.info(f"🧹 Cleaned up {len(orphaned)} orphaned file hashes")
    
    def get_stats(self) -> Dict:
        """Get statistics about stored hashes"""
        return {
            "total_hashes": len(self._hashes),
            "store_file": self.store_path,
            "file_exists": os.path.exists(self.store_path)
        }

def import_timestamp():
    """Get current timestamp for import tracking"""
    import datetime
    return datetime.datetime.now().isoformat()

# Global instance
_hash_store = None

def get_file_hash_store() -> FileHashStore:
    """Get global file hash store instance"""
    global _hash_store
    if _hash_store is None:
        _hash_store = FileHashStore()
    return _hash_store