from typing import Tuple
import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

def load_text_from_file(path: str) -> Tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt",".md",".json"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), ext.lstrip(".")
    if ext == ".pdf":
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "pdf"
    if ext == ".docx":
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text, "docx"
    raise ValueError(f"Unsupported file type: {ext}")
