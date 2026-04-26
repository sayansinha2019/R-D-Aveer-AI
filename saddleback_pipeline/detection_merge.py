"""Merge spatial-detection payloads (Gemini JSON, ONNX/YOLO JSON, external tools).

All payloads follow ``gemini_spatial_detection`` shape: ``{"version": 1, "pages": [...]}``.
"""

from __future__ import annotations

import math
from typing import Any


def _iou(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0, ax1, ay1 = a["x_min"], a["y_min"], a["x_max"], a["y_max"]
    bx0, by0, bx1, by1 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ba = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = aa + ba - inter
    return inter / denom if denom > 0 else 0.0


def _dedupe_instances(
    instances: list[dict[str, Any]],
    *,
    iou_thresh: float = 0.85,
) -> list[dict[str, Any]]:
    """Keep higher-confidence box when two normalized boxes overlap strongly."""
    items = sorted(
        instances,
        key=lambda x: float(x.get("confidence") or 0.0),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for inst in items:
        bb = inst.get("bbox") or {}
        if not isinstance(bb, dict):
            continue
        try:
            box = {
                "x_min": float(bb["x_min"]),
                "y_min": float(bb["y_min"]),
                "x_max": float(bb["x_max"]),
                "y_max": float(bb["y_max"]),
            }
        except (KeyError, TypeError, ValueError):
            kept.append(inst)
            continue
        dup = False
        for prev in kept:
            pb = prev.get("bbox") or {}
            if not isinstance(pb, dict):
                continue
            try:
                pbox = {
                    "x_min": float(pb["x_min"]),
                    "y_min": float(pb["y_min"]),
                    "x_max": float(pb["x_max"]),
                    "y_max": float(pb["y_max"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if _iou(box, pbox) >= iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(inst)
    return kept


def merge_detection_payloads(
    *payloads: dict[str, Any] | None,
    dedupe_iou: float = 0.85,
) -> dict[str, Any]:
    """Merge ``pages[*].structural_instances`` across payloads with same ``page_index``."""
    pages_by_idx: dict[int, dict[str, Any]] = {}

    for payload in payloads:
        if not payload:
            continue
        for pg in payload.get("pages") or []:
            if not isinstance(pg, dict):
                continue
            idx = int(pg.get("page_index") or 0)
            if idx < 1:
                continue
            if idx not in pages_by_idx:
                pages_by_idx[idx] = {
                    "page_index": idx,
                    "width_px": pg.get("width_px"),
                    "height_px": pg.get("height_px"),
                    "view_regions": list(pg.get("view_regions") or []),
                    "structural_instances": [],
                    "notes": list(pg.get("notes") or []),
                }
            base = pages_by_idx[idx]
            # Prefer pixel dimensions when missing
            if base.get("width_px") is None and pg.get("width_px"):
                base["width_px"] = pg.get("width_px")
            if base.get("height_px") is None and pg.get("height_px"):
                base["height_px"] = pg.get("height_px")
            for vr in pg.get("view_regions") or []:
                if isinstance(vr, dict):
                    base["view_regions"].append(vr)
            for n in pg.get("notes") or []:
                if n not in base["notes"]:
                    base["notes"].append(n)
            for si in pg.get("structural_instances") or []:
                if isinstance(si, dict):
                    base["structural_instances"].append(si)

    out_pages: list[dict[str, Any]] = []
    for idx in sorted(pages_by_idx.keys()):
        pg = pages_by_idx[idx]
        pg["structural_instances"] = _dedupe_instances(
            pg["structural_instances"],
            iou_thresh=dedupe_iou,
        )
        out_pages.append(pg)

    detectors = []
    for payload in payloads:
        if payload and payload.get("detector"):
            detectors.append(str(payload["detector"]))
    det_label = "+".join(detectors) if detectors else "composite"

    return {
        "version": 1,
        "detector": det_label,
        "pages": out_pages,
    }


def has_structural_instances(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    for pg in payload.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        if pg.get("structural_instances"):
            return True
    return False
