"""Unified PDF text extraction: PyMuPDF (local) or Google Document AI (cloud OCR/layout).

Document AI uses a **Google Cloud service account JSON key** via the standard env
``GOOGLE_APPLICATION_CREDENTIALS`` — not the Gemini API key.

Set ``PDF_TEXT_ENGINE=document_ai`` plus project, location, and processor ID to enable it.
Rasterization (PNG for Gemini) and vector geometry still use PyMuPDF elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def _page_texts_pymupdf(pdf_path: Path, max_pages: int | None) -> list[str]:
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        n = total if max_pages is None else min(total, max_pages)
        return [(doc[i].get_text("text") or "") for i in range(n)]
    finally:
        doc.close()


def _text_from_layout(layout: Any, doc_text: str) -> str:
    if layout is None or not doc_text:
        return ""
    ta = getattr(layout, "text_anchor", None)
    if ta is None:
        return ""
    parts: list[str] = []
    for seg in ta.text_segments:
        s = int(seg.start_index) if seg.start_index is not None else 0
        e = int(seg.end_index) if seg.end_index is not None else len(doc_text)
        parts.append(doc_text[s:e])
    return "".join(parts)


def _page_texts_document_ai(pdf_path: Path, max_pages: int | None) -> list[str]:
    try:
        from google.cloud import documentai_v1 as documentai
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-documentai is not installed. Run: pip install google-cloud-documentai"
        ) from e

    project_id = (os.getenv("DOCUMENT_AI_PROJECT_ID", "") or "").strip()
    location = (os.getenv("DOCUMENT_AI_LOCATION", "us") or "us").strip()
    processor_id = (os.getenv("DOCUMENT_AI_PROCESSOR_ID", "") or "").strip()
    if not project_id or not processor_id:
        raise RuntimeError(
            "DOCUMENT_AI_PROJECT_ID and DOCUMENT_AI_PROCESSOR_ID must be set for Document AI."
        )

    creds = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if not creds or not Path(creds).expanduser().is_file():
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS must point to your service account JSON key file."
        )

    pdf_path = pdf_path.expanduser().resolve()
    raw = pdf_path.read_bytes()

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    raw_doc = documentai.RawDocument(content=raw, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)

    timeout = float(os.getenv("DOCUMENT_AI_TIMEOUT_SEC", "300") or "300")
    result = client.process_document(request=request, timeout=timeout)
    document = result.document
    full = document.text or ""

    if not document.pages:
        return [full] if full else [""]

    texts: list[str] = []
    for page in document.pages:
        chunks: list[str] = []
        for block in page.blocks:
            t = _text_from_layout(block.layout, full)
            if t:
                chunks.append(t)
        # Fallback: page-level layout
        if not chunks and page.layout:
            t = _text_from_layout(page.layout, full)
            if t:
                chunks.append(t)
        texts.append("\n".join(chunks))

    if not any(t.strip() for t in texts) and full.strip():
        # Some processors attach most text at document level only.
        texts = [full] if len(document.pages) <= 1 else [p for p in full.split("\f")]

    if max_pages is not None:
        texts = texts[: max_pages]

    return texts


def get_pdf_page_texts(
    pdf_path: Path,
    *,
    max_pages: int | None = None,
    engine: str | None = None,
) -> tuple[list[str], str]:
    """Return (list of page text strings, engine name used).

    * ``engine`` — ``pymupdf``, ``document_ai``, or ``auto`` (default: env ``PDF_TEXT_ENGINE``).
    * ``auto`` tries Document AI when configured, else PyMuPDF.
    """
    pdf_path = pdf_path.expanduser().resolve()
    eng = (engine or os.getenv("PDF_TEXT_ENGINE", "auto") or "auto").strip().lower()

    if eng == "pymupdf":
        return _page_texts_pymupdf(pdf_path, max_pages), "pymupdf"

    if eng == "document_ai":
        return _page_texts_document_ai(pdf_path, max_pages), "document_ai"

    # auto
    has_doc_ai = bool(
        (os.getenv("DOCUMENT_AI_PROJECT_ID", "") or "").strip()
        and (os.getenv("DOCUMENT_AI_PROCESSOR_ID", "") or "").strip()
        and (os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
        and Path((os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()).expanduser().is_file()
    )
    if has_doc_ai:
        try:
            return _page_texts_document_ai(pdf_path, max_pages), "document_ai"
        except Exception:
            pass
    return _page_texts_pymupdf(pdf_path, max_pages), "pymupdf"
