"""Extract line segments from PDF pages (vector graphics) and optional raster Hough lines.

Outputs a JSON report for validation / future fusion with takeoff — not yet wired into Gemini.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def _iter_vector_lines(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    """Line segments as (x0, y0, x1, y1) in PDF page coordinates (points)."""
    out: list[tuple[float, float, float, float]] = []
    for path in page.get_drawings():
        if path.get("type") in ("clip", "group"):
            continue
        for item in path.get("items", []):
            cmd = item[0]
            if cmd == "l" and len(item) == 3:
                p1, p2 = item[1], item[2]
                out.append((float(p1.x), float(p1.y), float(p2.x), float(p2.y)))
    return out


def _snap_point(x: float, y: float, tol: float) -> tuple[float, float]:
    return (round(x / tol) * tol, round(y / tol) * tol)


def graph_stats_from_lines(
    lines: list[tuple[float, float, float, float]],
    *,
    snap_tol: float = 2.0,
) -> dict[str, Any]:
    """Merge endpoints within snap_tol; count degree-1 (ends), degree-3+ (joint-like)."""
    from collections import defaultdict

    adj: dict[tuple[float, float], set[tuple[float, float]]] = defaultdict(set)

    def add_edge(a: tuple[float, float], b: tuple[float, float]) -> None:
        if a == b:
            return
        adj[a].add(b)
        adj[b].add(a)

    for x0, y0, x1, y1 in lines:
        p0 = _snap_point(x0, y0, snap_tol)
        p1 = _snap_point(x1, y1, snap_tol)
        add_edge(p0, p1)

    deg = {n: len(neigh) for n, neigh in adj.items()}
    ends = sum(1 for d in deg.values() if d == 1)
    joints = sum(1 for d in deg.values() if d >= 3)
    corners = sum(1 for d in deg.values() if d == 2)

    return {
        "snap_tolerance_pt": snap_tol,
        "unique_nodes": len(adj),
        "edges_approx": sum(len(neigh) for neigh in adj.values()) // 2,
        "nodes_degree_1_ends": ends,
        "nodes_degree_2_corners": corners,
        "nodes_degree_3plus_joints": joints,
    }


def raster_hough_lines(
    page: fitz.Page,
    *,
    dpi: float = 150.0,
    max_side_px: int = 2000,
) -> list[tuple[float, float, float, float]]:
    """Detect line segments on a rasterized page (for scanned/flattened content). Requires OpenCV."""
    try:
        import cv2  # type: ignore[import-untyped]
        import numpy as np
    except ImportError:
        return []

    rect = page.rect
    zoom = dpi / 72.0
    pw, ph = rect.width * zoom, rect.height * zoom
    if max(pw, ph) > max_side_px:
        zoom *= max_side_px / max(pw, ph)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    seg = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=80,
        minLineLength=20,
        maxLineGap=10,
    )
    if seg is None:
        return []
    inv_z = 1.0 / zoom
    out: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in seg[:, 0]:
        out.append(
            (
                float(x0) * inv_z,
                float(y0) * inv_z,
                float(x1) * inv_z,
                float(y1) * inv_z,
            )
        )
    return out


def build_geometry_report(
    pdf_path: Path,
    *,
    page_index: int = 0,
    include_raster: bool = True,
) -> dict[str, Any]:
    pdf_path = pdf_path.expanduser().resolve()
    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"page_index {page_index} out of range (pages={len(doc)})")
        page = doc[page_index]
        vlines = _iter_vector_lines(page)
        raster: list[tuple[float, float, float, float]] = []
        if include_raster:
            raster = raster_hough_lines(page)
        vstats = graph_stats_from_lines(vlines)
        rstats = graph_stats_from_lines(raster) if raster else {}
        return {
            "pdf": str(pdf_path),
            "page_index": page_index,
            "page_size_pt": {"width": float(page.rect.width), "height": float(page.rect.height)},
            "vector_line_segments": len(vlines),
            "vector_graph": vstats,
            "raster_hough_segments": len(raster),
            "raster_graph": rstats if raster else None,
        }
    finally:
        doc.close()


def main() -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    raw = (os.getenv("INPUT_PDF", "") or "").strip()
    if not raw:
        print("ERROR: Set INPUT_PDF in .env", file=sys.stderr)
        return 1
    pdf = Path(raw).expanduser()
    if not pdf.is_file():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        return 1
    out = Path(
        os.getenv("GEOMETRY_REPORT_JSON", "geometry_report.json") or "geometry_report.json",
    ).expanduser()
    page_index = int(os.getenv("GEOMETRY_PAGE_INDEX", "0") or "0")
    rep = build_geometry_report(pdf, page_index=page_index, include_raster=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"Vector segments: {rep['vector_line_segments']}  "
        f"joint-like nodes (deg≥3): {rep['vector_graph']['nodes_degree_3plus_joints']}"
    )
    if rep.get("raster_graph"):
        print(
            f"Raster Hough segments: {rep['raster_hough_segments']}  "
            f"joint-like: {rep['raster_graph']['nodes_degree_3plus_joints']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
