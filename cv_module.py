"""
Computer Vision module (Section B Q1: capability i).

Reads a scanned book-cover image and extracts text via OCR, then
fuzzy-matches that text against the library catalogue to identify the
title/author. This demonstrates the CV -> NLP/RAG hand-off described in
part (b) of the question.
"""
from __future__ import annotations
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\Desktop\Tessaract\tesseract.exe"

import difflib
import json
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

CATALOGUE_PATH = Path(__file__).parent / "data" / "catalogue.json"


def load_catalogue() -> list[dict]:
    with open(CATALOGUE_PATH) as f:
        return json.load(f)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Basic CV preprocessing: grayscale + autocontrast to help OCR."""
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    return gray


def extract_text_from_cover(image: Image.Image) -> str:
    """Runs OCR on a (preprocessed) book-cover image."""
    if not TESSERACT_AVAILABLE:
        return ""
    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed)
    return text.strip()


def match_book_from_text(raw_text: str, catalogue: Optional[list[dict]] = None) -> dict:
    """
    Fuzzy-matches OCR'd text against catalogue titles/authors.
    Returns the best match plus a confidence score.
    """
    if catalogue is None:
        catalogue = load_catalogue()

    if not raw_text.strip():
        return {"match": None, "confidence": 0.0, "ocr_text": raw_text}

    candidates = [f"{b['title']} {b['author']}" for b in catalogue]
    best_ratio = 0.0
    best_idx = -1
    for i, cand in enumerate(candidates):
        ratio = difflib.SequenceMatcher(None, raw_text.lower(), cand.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    match = catalogue[best_idx] if best_idx >= 0 else None
    return {"match": match, "confidence": round(best_ratio, 3), "ocr_text": raw_text}


def identify_book_cover(image: Image.Image) -> dict:
    """End-to-end pipeline: image -> OCR -> catalogue match."""
    text = extract_text_from_cover(image)
    return match_book_from_text(text)
