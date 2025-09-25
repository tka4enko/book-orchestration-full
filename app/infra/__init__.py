"""
Infrastructure package - Configuration, security, and core utilities
Contains: settings, security, error handling, debugging, hash storage

Note: This package uses explicit imports only. Import specific items from submodules:
  from app.infra.settings import OPENAI_API_KEY, CHROMA_DIR
  from app.infra.debug import start_debug, finalize_debug
  from app.infra.hash_store import get_file_hash_store
"""