"""Render PDF pages to PNG bytes for vision models (high DPI, capped max side)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF


def render_pdf_to_png_pages(
    pdf_path: Path,
    *,
    dpi: float,
    max_side_px: int,
    max_pages: int | None = None,
    on_page: Callable[[int, int], None] | None = None,
) -> list[bytes]:
    """Rasterize each PDF page to PNG.

    Uses PyMuPDF with a zoom derived from ``dpi`` (72 pt = 1 inch). If the
    raster would exceed ``max_side_px`` on the longest edge, zoom is reduced
    uniformly so the image fits (keeps aspect ratio; stays within API limits).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Target resolution in dots per inch (e.g. 400–600 for legible small text).
        max_side_px: Maximum width or height in pixels (Gemini has per-image limits).
        max_pages: Optional cap on number of pages (first N pages only).
        on_page: Optional callback ``(index_1based, total_pages)`` for logging.

    Returns:
        One PNG byte string per page, in order.
    """
    pdf_path = pdf_path.expanduser().resolve()
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        if total == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")
        n = total if max_pages is None else min(total, max_pages)
        out: list[bytes] = []
        for i in range(n):
            if on_page:
                on_page(i + 1, n)
            page = doc[i]
            rect = page.rect
            w_pt, h_pt = rect.width, rect.height
            zoom = dpi / 72.0
            pw, ph = w_pt * zoom, h_pt * zoom
            if max(pw, ph) > max_side_px:
                zoom *= max_side_px / max(pw, ph)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
            out.append(pix.tobytes("png"))
        return out
    finally:
        doc.close()
