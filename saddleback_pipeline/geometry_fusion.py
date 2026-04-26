"""Fuse vector line geometry with takeoff entities (per view) and optional detection JSON.

Reference BOM accuracy (vs Excel) does not change unless you regenerate takeoff JSON.
This module adds a **geometry–takeoff consistency score** for QA and future model feedback.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import fitz

from saddleback_pipeline.drawing_scales import extract_drawing_scales
from saddleback_pipeline.geometry_scale import (
    max_orthogonal_segment_lengths_pt,
    pdf_pt_to_real_feet,
    primary_architectural_scale_for_page,
)
from saddleback_pipeline.pdf_geometry import _iter_vector_lines, graph_stats_from_lines


def _norm_bbox_to_rect(page: fitz.Page, bbox: dict[str, float]) -> fitz.Rect:
    w, h = page.rect.width, page.rect.height
    x0 = float(bbox["x_min"]) * w
    y0 = float(bbox["y_min"]) * h
    x1 = float(bbox["x_max"]) * w
    y1 = float(bbox["y_max"]) * h
    return fitz.Rect(x0, y0, x1, y1)


def _line_mid_in_rect(
    line: tuple[float, float, float, float],
    rect: fitz.Rect,
) -> bool:
    mx = (line[0] + line[2]) * 0.5
    my = (line[1] + line[3]) * 0.5
    return rect.contains(fitz.Point(mx, my))


def _segment_length(line: tuple[float, float, float, float]) -> float:
    return math.hypot(line[2] - line[0], line[3] - line[1])


def _is_vertical(
    line: tuple[float, float, float, float],
    *,
    min_len: float,
    max_horiz_ratio: float = 0.22,
) -> bool:
    """Nearly vertical segment (for column-like strokes in elevations)."""
    L = _segment_length(line)
    if L < min_len:
        return False
    dx = abs(line[2] - line[0])
    return dx <= max_horiz_ratio * L


def _is_horizontal(
    line: tuple[float, float, float, float],
    *,
    min_len: float,
    max_vert_ratio: float = 0.22,
) -> bool:
    L = _segment_length(line)
    if L < min_len:
        return False
    dy = abs(line[3] - line[1])
    return dy <= max_vert_ratio * L


def _lines_in_rect(
    lines: list[tuple[float, float, float, float]],
    rect: fitz.Rect,
) -> list[tuple[float, float, float, float]]:
    return [ln for ln in lines if _line_mid_in_rect(ln, rect)]


def _count_takeoff_in_view(
    entities: list[dict[str, Any]],
    substrings: tuple[str, ...],
    *,
    element_types: set[str] | None = None,
) -> int:
    """Sum quantity for rows whose source_reference matches all substrings (case-insensitive)."""
    total = 0
    for e in entities:
        if not isinstance(e, dict):
            continue
        src = (e.get("source_reference") or "").lower()
        if not all(s.lower() in src for s in substrings):
            continue
        et = (e.get("element_type") or "").strip()
        if element_types is not None and et not in element_types:
            continue
        try:
            total += int(e.get("quantity", 1))
        except (TypeError, ValueError):
            total += 1
    return total


def _detection_counts_per_view(
    det: dict[str, Any],
    page_index: int,
) -> dict[str, dict[str, int]]:
    """Count structural_instances by view_id and class."""
    out: dict[str, dict[str, int]] = {}
    for pg in det.get("pages", []):
        if int(pg.get("page_index", -1)) != page_index:
            continue
        for inst in pg.get("structural_instances", []):
            vid = str(inst.get("view_id") or "")
            cls = str(inst.get("class") or "").lower()
            out.setdefault(vid, {})
            out[vid][cls] = out[vid].get(cls, 0) + 1
    return out


# Elevation views: substring to find in takeoff source_reference
_VIEW_TAKEOFF_MAP: dict[str, tuple[str, ...]] = {
    "front_elevation": ("front elevation",),
    "side_elevation": ("side elevation",),
    "top_elevation": ("top elevation",),
}


def build_fusion_report(
    *,
    pdf_path: Path,
    takeoff_json_path: Path,
    detections_json_path: Path | None,
    page_index: int = 0,
    min_line_pt: float = 12.0,
) -> dict[str, Any]:
    pdf_path = pdf_path.expanduser().resolve()
    takeoff_json_path = takeoff_json_path.expanduser().resolve()

    payload = json.loads(takeoff_json_path.read_text(encoding="utf-8"))
    entities = payload.get("data")
    if not isinstance(entities, list):
        entities = []

    # Drawing scales: prefer takeoff meta (same as Gemini run); else extract from PDF text.
    scales_payload: dict[str, Any] | None = None
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    ds = meta.get("drawing_scales") if isinstance(meta.get("drawing_scales"), dict) else None
    scales_from_meta = bool(ds and ds.get("pages"))
    if scales_from_meta:
        scales_payload = ds
    else:
        try:
            scales_payload = extract_drawing_scales(pdf_path)
        except Exception:
            scales_payload = None

    page_1based = page_index + 1
    scale_row: dict[str, Any] | None = None
    fpd: float | None = None
    if scales_payload:
        scale_row = primary_architectural_scale_for_page(
            scales_payload,
            page_1based=page_1based,
        )
        if scale_row:
            v = scale_row.get("feet_per_drawing_inch")
            if isinstance(v, (int, float)) and v > 0:
                fpd = float(v)

    det: dict[str, Any] = {}
    if detections_json_path and detections_json_path.is_file():
        det = json.loads(detections_json_path.read_text(encoding="utf-8"))

    doc = fitz.open(pdf_path)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"page_index {page_index} invalid (pages={len(doc)})")
        page = doc[page_index]
        all_lines = _iter_vector_lines(page)
        page_w, page_h = float(page.rect.width), float(page.rect.height)

        views_out: list[dict[str, Any]] = []
        alignments: list[float] = []

        page_det: dict[str, Any] = {}
        for pg in det.get("pages", []) or []:
            if int(pg.get("page_index", -999)) == page_index:
                page_det = pg
                break
        if not page_det and det.get("pages"):
            page_det = det["pages"][0]
        view_regions = page_det.get("view_regions") or [] if isinstance(page_det, dict) else []

        det_by_view = _detection_counts_per_view(det, page_index)

        for vr in view_regions:
            vid = str(vr.get("view_id") or "")
            bbox = vr.get("bbox")
            if not isinstance(bbox, dict) or vid not in _VIEW_TAKEOFF_MAP:
                continue
            rect = _norm_bbox_to_rect(page, bbox)
            clipped = _lines_in_rect(all_lines, rect)
            gs = graph_stats_from_lines(clipped, snap_tol=2.0)
            v_long = sum(
                1
                for ln in clipped
                if _is_vertical(ln, min_len=min_line_pt)
            )
            h_long = sum(
                1
                for ln in clipped
                if _is_horizontal(ln, min_len=min_line_pt)
            )

            max_h_pt, max_v_pt = max_orthogonal_segment_lengths_pt(
                clipped,
                min_len_pt=min_line_pt,
            )
            bbox_w_pt = float(rect.width)
            bbox_h_pt = float(rect.height)
            scale_block: dict[str, Any] | None = None
            if fpd is not None:
                scale_block = {
                    "feet_per_drawing_inch": fpd,
                    "scale_raw": scale_row.get("raw") if scale_row else None,
                    "view_bbox_pdf_pt": {
                        "width": bbox_w_pt,
                        "height": bbox_h_pt,
                    },
                    "view_bbox_real_ft_est": {
                        "width": round(pdf_pt_to_real_feet(bbox_w_pt, fpd), 4),
                        "height": round(pdf_pt_to_real_feet(bbox_h_pt, fpd), 4),
                    },
                    "longest_segment_pdf_pt": {
                        "horizontal": round(max_h_pt, 3),
                        "vertical": round(max_v_pt, 3),
                    },
                    "longest_segment_real_ft_est": {
                        "horizontal": round(pdf_pt_to_real_feet(max_h_pt, fpd), 4)
                        if max_h_pt > 0
                        else None,
                        "vertical": round(pdf_pt_to_real_feet(max_v_pt, fpd), 4)
                        if max_v_pt > 0
                        else None,
                    },
                    "note": (
                        "Estimates assume the view uses the primary architectural scale parsed for "
                        "this page; detail views may use a different scale."
                    ),
                }

            subs = _VIEW_TAKEOFF_MAP[vid]
            takeoff_struct = _count_takeoff_in_view(
                entities,
                subs,
                element_types={
                    "Column",
                    "Beam",
                    "Brace",
                    "Rafter",
                    "Plate",
                    "Rod",
                    "Misc",
                },
            )
            takeoff_cols = _count_takeoff_in_view(
                entities,
                subs,
                element_types={"Column"},
            )

            # Geometric proxy: joint-like nodes + weak weight on long ortho segments
            geo_signal = float(gs["nodes_degree_3plus_joints"]) + 0.05 * float(
                v_long + h_long
            )
            takeoff_signal = float(max(takeoff_struct, 1))
            if geo_signal <= 0:
                align = 0.0
            else:
                align = min(geo_signal, takeoff_signal) / max(geo_signal, takeoff_signal)

            dcols = det_by_view.get(vid, {}).get("column", 0)
            dbeams = det_by_view.get(vid, {}).get("beam", 0)

            # Vision-detection vs takeoff (columns only): comparable scales
            det_col_align: float | None = None
            if max(dcols, takeoff_cols) > 0:
                det_col_align = min(dcols, takeoff_cols) / max(dcols, takeoff_cols)

            alignments.append(align)

            views_out.append(
                {
                    "view_id": vid,
                    "label": vr.get("label"),
                    "bbox_pdf_pt": {
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                    },
                    "vector_segments_in_view": len(clipped),
                    "long_vertical_segments": v_long,
                    "long_horizontal_segments": h_long,
                    "graph": gs,
                    "takeoff_structural_qty_sum": takeoff_struct,
                    "takeoff_column_qty_sum": takeoff_cols,
                    "detection_column_boxes": dcols,
                    "detection_beam_boxes": dbeams,
                    "geometry_takeoff_alignment_01": round(align, 4),
                    "detection_vs_takeoff_column_alignment_01": (
                        round(det_col_align, 4) if det_col_align is not None else None
                    ),
                    "scale_calibrated_geometry": scale_block,
                }
            )

        overall = sum(alignments) / len(alignments) if alignments else None

        det_col_aligns = [
            v["detection_vs_takeoff_column_alignment_01"]
            for v in views_out
            if v.get("detection_vs_takeoff_column_alignment_01") is not None
            and int(v.get("takeoff_column_qty_sum") or 0) > 0
            and int(v.get("detection_column_boxes") or 0) > 0
        ]
        mean_det_col = (
            sum(det_col_aligns) / len(det_col_aligns) if det_col_aligns else None
        )

        return {
            "pdf": str(pdf_path),
            "takeoff_json": str(takeoff_json_path),
            "detections_json": str(detections_json_path) if detections_json_path else None,
            "page_index": page_index,
            "page_size_pt": {"width": page_w, "height": page_h},
            "drawing_scale_used": (
                {
                    "raw": scale_row.get("raw"),
                    "feet_per_drawing_inch": fpd,
                    "source": (
                        "takeoff.meta.drawing_scales"
                        if scales_from_meta
                        else "extract_drawing_scales(pdf)"
                    ),
                }
                if fpd is not None and scale_row
                else {
                    "feet_per_drawing_inch": None,
                    "note": "No architectural scale for this page — scale_calibrated_geometry omitted.",
                }
            ),
            "elevation_views": views_out,
            "overall_geometry_takeoff_alignment_mean": (
                round(overall, 4) if overall is not None else None
            ),
            "mean_detection_vs_takeoff_column_alignment": (
                round(mean_det_col, 4) if mean_det_col is not None else None
            ),
            "note": (
                "This score compares coarse vector signals (joints + long segments) to "
                "takeoff quantities per view. It is NOT reference BOM F1 vs Excel."
            ),
        }
    finally:
        doc.close()


def main() -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    pdf = Path((os.getenv("INPUT_PDF", "") or "").strip()).expanduser()
    takeoff = Path(
        (os.getenv("INPUT_JSON", "") or "").strip() or "takeoff_output.json",
    ).expanduser()
    det = Path(
        (os.getenv("DETECTIONS_JSON", "") or "").strip() or "detections_output.json",
    ).expanduser()
    out = Path(
        (os.getenv("FUSION_REPORT_JSON", "") or "").strip() or "fusion_report.json",
    ).expanduser()

    if not pdf.is_file():
        print(f"ERROR: INPUT_PDF not found: {pdf}", file=sys.stderr)
        return 1
    if not takeoff.is_file():
        print(f"ERROR: takeoff JSON not found: {takeoff}", file=sys.stderr)
        return 1
    det_path = det if det.is_file() else None
    if det_path is None:
        print("WARNING: detections JSON missing; fusion needs view bboxes.", file=sys.stderr)
        return 2

    rep = build_fusion_report(
        pdf_path=pdf,
        takeoff_json_path=takeoff,
        detections_json_path=det_path,
        page_index=int(os.getenv("GEOMETRY_PAGE_INDEX", "0") or "0"),
    )
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(
        "Overall geometry↔takeoff alignment (mean):",
        rep.get("overall_geometry_takeoff_alignment_mean"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
