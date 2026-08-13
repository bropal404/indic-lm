import mimetypes
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pdfplumber
from bs4 import BeautifulSoup
from ebooklib import epub

from .extractors.generic import Document


def identify_file_type(url: str, content_type: str) -> str:
    lower = content_type.lower()
    if "pdf" in lower or url.lower().endswith(".pdf"):
        return "pdf"
    if "epub" in lower or url.lower().endswith(".epub"):
        return "epub"
    if "text/plain" in lower or url.lower().endswith(".txt"):
        return "txt"
    if "html" in lower:
        return "html"
    guess, _ = mimetypes.guess_type(url)
    return guess or "unknown"


def extract_pdf_text(path: Path) -> Dict[str, object]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = "\n\n".join(pages).strip()
    return {"text": text, "ocr_used": False, "needs_ocr": len(text) < 100}


def extract_epub_text(path: Path) -> str:
    book = epub.read_epub(str(path))
    parts = []
    for item in book.get_items():
        if item.media_type == "application/xhtml+xml":
            soup = BeautifulSoup(item.get_content(), "lxml")
            parts.append(soup.get_text("\n", strip=True))
    return "\n\n".join(parts)


def archive_document_to_row(doc: Document, file_type: str, license_text: Optional[str] = None, ocr_used: bool = False, needs_ocr: bool = False) -> Dict[str, object]:
    row = asdict(doc)
    row.update(
        {
            "pipeline": "archive",
            "file_type": file_type,
            "license": license_text,
            "ocr_used": ocr_used,
            "needs_ocr": needs_ocr,
            "rights_note": "Publicly visible does not imply redistributable; verify source/license before redistribution.",
        }
    )
    return row
