"""Convert PDF point / pixel geometry to real feet using drawing scale factors."""

from __future__ import annotations

import math
from typing import Any

from saddleback_pipeline.drawing_scales import real_feet_from_drawing_inches


def pdf_pt_to_drawing_inches(pt: float) -> float:
    """PDF uses 72 points per inch of paper."""
    return float(pt) / 72.0


def pdf_pt_to_real_feet(pt_length: float, feet_per_drawing_inch: float) -> float:
    """Real-world feet along a distance measured on the paper (PDF points)."""
    return real_feet_from_drawing_inches(
        pdf_pt_to_drawing_inches(pt_length),
        feet_per_drawing_inch,
    )


def primary_architectural_scale_for_page(
    drawing_scales_payload: dict[str, Any],
    *,
    page_1based: int,
) -> dict[str, Any] | None:
    """First numeric architectural scale on a page (1-based page index)."""
    for pg in drawing_scales_payload.get("pages") or []:
        if not isinstance(pg, dict) or int(pg.get("page") or -1) != page_1based:
            continue
        for s in pg.get("scales") or []:
            if not isinstance(s, dict):
                continue
            if s.get("kind") == "metric_ratio":
                continue
            fpd = s.get("feet_per_drawing_inch")
            if isinstance(fpd, (int, float)) and fpd > 0 and not math.isnan(fpd):
                return s
    return None


def _seg_len(line: tuple[float, float, float, float]) -> float:
    return math.hypot(line[2] - line[0], line[3] - line[1])


def _is_h(
    line: tuple[float, float, float, float],
    *,
    min_len: float,
    max_vert_ratio: float = 0.22,
) -> bool:
    L = _seg_len(line)
    if L < min_len:
        return False
    dy = abs(line[3] - line[1])
    return dy <= max_vert_ratio * L


def _is_v(
    line: tuple[float, float, float, float],
    *,
    min_len: float,
    max_horiz_ratio: float = 0.22,
) -> bool:
    L = _seg_len(line)
    if L < min_len:
        return False
    dx = abs(line[2] - line[0])
    return dx <= max_horiz_ratio * L


def max_orthogonal_segment_lengths_pt(
    lines: list[tuple[float, float, float, float]],
    *,
    min_len_pt: float,
) -> tuple[float, float]:
    """Return (max horizontal length, max vertical length) in PDF points."""
    mh = mv = 0.0
    for ln in lines:
        if _is_h(ln, min_len=min_len_pt):
            mh = max(mh, _seg_len(ln))
        if _is_v(ln, min_len=min_len_pt):
            mv = max(mv, _seg_len(ln))
    return mh, mv
