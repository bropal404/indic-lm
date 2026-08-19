#!/usr/bin/env python3
"""
PDF to Text Dataset Extractor for Devanagari script.
Supports native text extraction with automatic fallback to EasyOCR when
confronted with scanned pages, blank pages, or legacy/non-Unicode fonts.
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import pdfplumber
import numpy as np
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover - optional OCR dependency
    convert_from_path = None

try:
    import easyocr
except ImportError:  # pragma: no cover - optional OCR dependency
    easyocr = None

class DevanagariOCRExtractor:
    def __init__(self, languages: List[str] = ['hi'], use_gpu: bool = False):
        """
        Initializes the Devanagari OCR Extractor.
        
        Args:
            languages: List of languages for EasyOCR. Defaults to ['hi'] (Hindi).
            use_gpu: Whether to use GPU acceleration for OCR.
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self._reader = None

    @property
    def reader(self):
        """Lazy-loaded EasyOCR reader instance."""
        if easyocr is None:
            raise RuntimeError("EasyOCR is not installed. Install optional OCR dependencies before using --force-ocr.")
        if self._reader is None:
            print("Initializing EasyOCR reader (this might download language models on the first run)...")
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    @staticmethod
    def is_devanagari(char: str) -> bool:
        """
        Checks if a character belongs to the Devanagari Unicode block (U+0900 to U+097F).
        """
        return 0x0900 <= ord(char) <= 0x097F

    def devanagari_ratio(self, text: str) -> float:
        """
        Calculates the ratio of Devanagari characters to total non-whitespace characters.
        
        Args:
            text: Input string to evaluate.
            
        Returns:
            Float representation of the ratio (0.0 to 1.0).
        """
        clean_text = "".join(text.split())
        if not clean_text:
            return 0.0
        dev_count = sum(1 for char in clean_text if self.is_devanagari(char))
        return dev_count / len(clean_text)

    def extract_native_text(self, page) -> str:
        """
        Extracts native text from a pdfplumber page.
        """
        text = page.extract_text()
        return text if text else ""

    def extract_ocr_text(self, page_image: Image.Image) -> str:
        """
        Performs OCR on a PIL Image containing a page and returns the recognized text.
        """
        # Convert PIL Image to numpy array (RGB) for EasyOCR
        image_np = np.array(page_image)
        # Perform readtext
        results = self.reader.readtext(image_np, detail=0)
        return "\n".join(results)

    def process_pdf(self, pdf_path: Path, force_ocr: bool = False, min_dev_ratio: float = 0.4) -> str:
        """
        Processes a single PDF file page-by-page.
        
        Args:
            pdf_path: Path to the PDF file.
            force_ocr: If True, forces OCR for all pages.
            min_dev_ratio: Minimum ratio of Devanagari chars required to accept native text.
            
        Returns:
            Extracted text string.
        """
        extracted_pages = []
        pdf_path_str = str(pdf_path)

        with pdfplumber.open(pdf_path_str) as pdf:
            total_pages = len(pdf.pages)
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                native_text = ""
                dev_ratio = 0.0
                use_ocr = force_ocr

                if not force_ocr:
                    native_text = self.extract_native_text(page)
                    dev_ratio = self.devanagari_ratio(native_text)
                    
                    # Fallback condition: empty/low text content or poor Devanagari ratio (suggesting legacy font/scanned page)
                    if not native_text.strip() or dev_ratio < min_dev_ratio:
                        use_ocr = True

                if use_ocr:
                    if convert_from_path is None:
                        raise RuntimeError("pdf2image is not installed. Install optional OCR dependencies before using OCR fallback.")
                    print(f"[{pdf_path.name}] Page {page_num}/{total_pages}: Running OCR (EasyOCR)...")
                    # Lazily convert only the required page to image to save CPU/Memory
                    images = convert_from_path(pdf_path_str, first_page=page_num, last_page=page_num)
                    if images:
                        page_text = self.extract_ocr_text(images[0])
                    else:
                        page_text = ""
                else:
                    print(f"[{pdf_path.name}] Page {page_num}/{total_pages}: Using Native Text (Devanagari ratio: {dev_ratio:.2f})")
                    page_text = native_text

                extracted_pages.append(f"--- PAGE {page_num} ---\n{page_text}")

        return "\n\n".join(extracted_pages)


def clean_devanagari_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace and handling common artifacts.
    """
    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of Devanagari PDFs to a text dataset."
    )
    parser.add_argument("--input-dir", "-i", type=str, required=True,
                        help="Path to the directory containing PDFs.")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                        help="Path to save the extracted text files.")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Force run EasyOCR on all pages instead of native extraction.")
    parser.add_argument("--min-dev-ratio", type=float, default=0.4,
                        help="Minimum Devanagari character ratio to accept native text.")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU acceleration for OCR if available.")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF files in '{input_dir}'. Starting extraction...")
    extractor = DevanagariOCRExtractor(use_gpu=args.use_gpu)

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            raw_text = extractor.process_pdf(
                pdf_file, 
                force_ocr=args.force_ocr, 
                min_dev_ratio=args.min_dev_ratio
            )
            cleaned_text = clean_devanagari_text(raw_text)
            
            output_file = output_dir / f"{pdf_file.stem}.txt"
            output_file.write_text(cleaned_text, encoding="utf-8")
            print(f"Successfully saved extracted text to: {output_file}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
