"""Hybrid takeoff: Gemini-first (vision), optional YOLO, text fallback, per-view scale aware.

Why this exists
---------------
Many sheets have a strong PDF text layer (schedules/callouts) → deterministic parsing is
high-accuracy and cheap. But some sheets are image-only or have missing/garbled text
layers → we must rely on vision (Gemini, or a trained detector like YOLO).

This orchestrator:
* Extracts page scales (and view-region scales when view boxes are available).
* Runs Gemini takeoff (full extraction) when API key is available.
* Optionally runs a vision detector (Gemini spatial / YOLO) to get view boxes + per-view validation.
* Can fall back to deterministic PDF-text parsing if desired.
* Optionally runs a vision detector to count/locate members for validation or gap-filling:
  - Gemini spatial detection (requires valid API key)
  - ONNX YOLO detector (requires a trained model file)
* Produces a merged takeoff JSON payload plus a per-view validation report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from saddleback_pipeline.drawing_scales import (
    extract_drawing_scales,
    extract_view_region_scales,
)
from saddleback_pipeline.detection_merge import merge_detection_payloads
from saddleback_pipeline.gemini_spatial_detection import run_spatial_detection_for_pages
from saddleback_pipeline.onnx_yolo_detector import run_yolo_onnx_for_png_pages
from saddleback_pipeline.pdf_to_images import render_pdf_to_png_pages
from saddleback_pipeline.gemini_takeoff import run_takeoff as run_gemini_takeoff
from saddleback_pipeline.takeoff_merge import merge_takeoff_payloads, write_takeoff_json
from saddleback_pipeline.text_schedule_takeoff import extract_w_members_from_pdf_text, members_to_takeoff_payload


def _beam_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in payload.get("data") or []:
        if not isinstance(e, dict):
            continue
        et = str(e.get("element_type") or "").strip().lower()
        if et == "beam" or "beam" in et or "rafter" in et or "girder" in et or "joist" in et:
            out.append(e)
    return out


def _non_beam_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in payload.get("data") or []:
        if not isinstance(e, dict):
            continue
        et = str(e.get("element_type") or "").strip().lower()
        if not (et == "beam" or "beam" in et or "rafter" in et or "girder" in et or "joist" in et):
            out.append(e)
    return out


def _text_beam_entities(pdf: Path) -> list[dict[str, Any]]:
    """Deterministic beams from PDF text layer (only those with parsed length)."""
    members = [m for m in extract_w_members_from_pdf_text(pdf) if m.get("length")]
    p = members_to_takeoff_payload(members)
    return [e for e in (p.get("data") or []) if isinstance(e, dict)]


def _maybe_swap_beams_to_text(
    *,
    gemini_payload: dict[str, Any],
    pdf: Path,
) -> dict[str, Any]:
    """If Gemini undercounts beams vs PDF text, replace only beams with text-beams.

    This keeps the earlier behavior (Gemini primary) but avoids "Gemini gave too few beams"
    regressions on text-heavy sheets.
    """
    gb = _beam_entities(gemini_payload)
    tb = _text_beam_entities(pdf)
    if len(tb) > len(gb) and len(tb) > 0:
        merged = {**{k: v for k, v in gemini_payload.items() if k != "data"}}
        merged["data"] = [*tb, *_non_beam_entities(gemini_payload)]
        merged.setdefault("meta", {})
        if isinstance(merged["meta"], dict):
            merged["meta"]["hybrid_beam_swap"] = {
                "performed": True,
                "gemini_beam_count": len(gb),
                "text_beam_count": len(tb),
                "notes": "Swapped beams to PDF text because Gemini beam count was lower.",
            }
        return merged
    gemini_payload.setdefault("meta", {})
    if isinstance(gemini_payload.get("meta"), dict):
        gemini_payload["meta"]["hybrid_beam_swap"] = {
            "performed": False,
            "gemini_beam_count": len(gb),
            "text_beam_count": len(tb),
        }
    return gemini_payload


def _truthy_env(name: str, default: str = "false") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "y"}


def _counts_from_takeoff_entities(payload: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in payload.get("data") or []:
        if not isinstance(e, dict):
            continue
        et = str(e.get("element_type") or "UNKNOWN").strip() or "UNKNOWN"
        try:
            q = int(e.get("quantity") or 1)
        except (TypeError, ValueError):
            q = 1
        out[et] = out.get(et, 0) + max(1, q)
    return out


def _counts_from_detections(dets: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for pg in dets.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        for inst in pg.get("structural_instances") or []:
            if not isinstance(inst, dict):
                continue
            cls = str(inst.get("class") or "unknown").strip().lower() or "unknown"
            out[cls] = out.get(cls, 0) + 1
    return out


def _bbox_center(b: dict[str, Any]) -> tuple[float, float] | None:
    try:
        x0 = float(b["x_min"])
        y0 = float(b["y_min"])
        x1 = float(b["x_max"])
        y1 = float(b["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _point_in_bbox(pt: tuple[float, float], b: dict[str, Any]) -> bool:
    try:
        x0 = float(b["x_min"])
        y0 = float(b["y_min"])
        x1 = float(b["x_max"])
        y1 = float(b["y_max"])
    except (KeyError, TypeError, ValueError):
        return False
    x, y = pt
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def _per_view_detection_counts(dets: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Return {view_id: {class: count}}.

    - If instances already carry view_id, use it.
    - Else, if view_regions exist, assign by bbox-center containment.
    """
    out: dict[str, dict[str, int]] = {}
    for pg in dets.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        vrs = [vr for vr in (pg.get("view_regions") or []) if isinstance(vr, dict) and isinstance(vr.get("bbox"), dict)]
        for inst in pg.get("structural_instances") or []:
            if not isinstance(inst, dict):
                continue
            cls = str(inst.get("class") or "unknown").strip().lower() or "unknown"
            vid = str(inst.get("view_id") or "").strip()
            if not vid and vrs:
                bb = inst.get("bbox")
                if isinstance(bb, dict):
                    c = _bbox_center(bb)
                    if c is not None:
                        for vr in vrs:
                            if _point_in_bbox(c, vr["bbox"]):  # type: ignore[index]
                                vid = str(vr.get("view_id") or "").strip()
                                break
            if not vid:
                vid = "__unassigned__"
            out.setdefault(vid, {})
            out[vid][cls] = out[vid].get(cls, 0) + 1
    return out


def _primary_fpd_for_view(view_scales: dict[str, Any] | None, *, view_id: str) -> float | None:
    """Pick the first numeric feet_per_drawing_inch for a view_id from view_scale payload."""
    if not view_scales:
        return None
    for pg in view_scales.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        for v in pg.get("views") or []:
            if not isinstance(v, dict):
                continue
            if str(v.get("view_id") or "").strip() != view_id:
                continue
            for s in v.get("scales") or []:
                if not isinstance(s, dict):
                    continue
                fpd = s.get("feet_per_drawing_inch")
                if isinstance(fpd, (int, float)) and fpd > 0:
                    return float(fpd)
    return None


def run_hybrid_takeoff(
    *,
    pdf: Path,
    out_json: Path,
    reference_project1_bom: Path | None = None,
    detections_json_out: Path | None = None,
    validation_json_out: Path | None = None,
) -> dict[str, Any]:
    pdf = pdf.expanduser().resolve()
    out_json = out_json.expanduser().resolve()

    # 1) Extract page scales from PDF text
    scales = extract_drawing_scales(pdf)

    # 2) Primary takeoff: Gemini (vision). Reference BOM is for VALIDATION ONLY (never copied into output).
    mode = (os.getenv("HYBRID_TAKEOFF_MODE", "gemini") or "gemini").strip().lower()
    if mode == "text":
        # Minimal deterministic fallback: beams from PDF text callouts only.
        members = [m for m in extract_w_members_from_pdf_text(pdf) if m.get("length")]
        payload_main = members_to_takeoff_payload(members)
        payload_main.setdefault("meta", {})
        if isinstance(payload_main["meta"], dict):
            payload_main["meta"]["hybrid_text_pass"] = {
                "enabled": True,
                "entity_count": len(payload_main.get("data") or []),
                "notes": "Deterministic fallback: beams from PDF text only (no reference BOM injection).",
            }
    else:
        key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
        model = (os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview").strip()
        prompt_path = Path(os.getenv("PROMPT_PATH", "prompts/structural_takeoff.txt") or "prompts/structural_takeoff.txt")
        schema_csv = (os.getenv("SCHEMA_CSV", "") or "").strip()
        schema_path = Path(schema_csv) if schema_csv else None
        if not key:
            raise ValueError("HYBRID_TAKEOFF_MODE=gemini requires GEMINI_API_KEY")
        tmp_out = out_json.with_suffix(".gemini_tmp.json")
        run_gemini_takeoff(
            pdf_path=pdf,
            output_json=tmp_out,
            gemini_api_key=key,
            model=model,
            prompt_path=prompt_path,
            schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
        )
        gemini_payload = json.loads(tmp_out.read_text(encoding="utf-8"))
        payload_main = _maybe_swap_beams_to_text(gemini_payload=gemini_payload, pdf=pdf)

    payload_main.setdefault("meta", {})
    if isinstance(payload_main["meta"], dict):
        payload_main["meta"]["drawing_scales"] = scales

    # 3) Vision detection pass (optional) — for per-view processing and validation
    detections_accum: dict[str, Any] | None = None
    use_any_vision = _truthy_env("HYBRID_ENABLE_VISION", "true")
    if use_any_vision:
        dpi = float(os.getenv("PDF_IMAGE_DPI", "450") or "450")
        max_side = int(os.getenv("PDF_IMAGE_MAX_SIDE_PX", "4096") or "4096")
        max_pages_env = (os.getenv("PDF_MAX_PAGES", "") or "").strip()
        max_pages = int(max_pages_env) if max_pages_env else None
        png_pages = render_pdf_to_png_pages(pdf, dpi=dpi, max_side_px=max_side, max_pages=max_pages)

        # 3a) Gemini spatial detection (if key works)
        if _truthy_env("HYBRID_USE_GEMINI_SPATIAL", "false"):
            key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
            model = (os.getenv("GEMINI_DETECTION_MODEL", "") or "").strip() or (
                os.getenv("GEMINI_MODEL", "") or "gemini-2.5-flash"
            )
            prompt_path = Path(
                os.getenv("GEMINI_DETECTION_PROMPT_PATH", "prompts/spatial_detection.txt")
                or "prompts/spatial_detection.txt",
            )
            prompt_text = prompt_path.read_text(encoding="utf-8")
            if key:
                gem = run_spatial_detection_for_pages(
                    png_pages=png_pages,
                    gemini_api_key=key,
                    model=model,
                    prompt_text=prompt_text,
                    timeout_sec=int(os.getenv("GEMINI_HTTP_TIMEOUT_SEC", "1800") or "1800"),
                    max_output_tokens=int(os.getenv("GEMINI_DETECTION_MAX_OUTPUT_TOKENS", "16384") or "16384"),
                    use_stream=False,
                )
                detections_accum = merge_detection_payloads(detections_accum, gem)

        # 3b) ONNX YOLO (if provided)
        onnx_raw = (os.getenv("YOLO_ONNX_MODEL", "") or "").strip()
        if onnx_raw and Path(onnx_raw).expanduser().is_file():
            yo = run_yolo_onnx_for_png_pages(png_pages, Path(onnx_raw).expanduser())
            detections_accum = merge_detection_payloads(detections_accum, yo)

    # 4) View-region scales when view boxes exist
    view_scales: dict[str, Any] | None = None
    if detections_accum is not None:
        try:
            view_scales = extract_view_region_scales(
                pdf,
                detections_accum,
                margin_pt=float(os.getenv("VIEW_SCALE_MARGIN_PT", "48") or "48"),
            )
            if isinstance(payload_main.get("meta"), dict) and view_scales:
                payload_main["meta"]["view_region_scales"] = view_scales
        except Exception:
            view_scales = None

    # 5) Preserve payload meta (hybrid decisions, scales, etc.).
    # `merge_takeoff_payloads` currently normalizes meta to merge-only fields, which would drop
    # our hybrid reconciliation diagnostics. For a single payload, keep it as-is.
    merged = payload_main

    # 6) Validation report: compare text quantities vs detection counts
    per_view = _per_view_detection_counts(detections_accum or {})
    per_view_out: list[dict[str, Any]] = []
    for vid, classes in sorted(per_view.items()):
        per_view_out.append(
            {
                "view_id": vid,
                "vision_counts_by_class": classes,
                "feet_per_drawing_inch_for_view": (
                    _primary_fpd_for_view(view_scales, view_id=vid) if vid and vid != "__unassigned__" else None
                ),
            }
        )
    validation: dict[str, Any] = {
        "pdf": str(pdf),
        "takeoff_counts_by_element_type": _counts_from_takeoff_entities(payload_main),
        "vision_detection_counts_by_class": _counts_from_detections(detections_accum or {}),
        "per_view_vision_counts": per_view_out,
        "has_view_regions": bool(
            detections_accum
            and any((pg.get("view_regions") for pg in (detections_accum.get("pages") or []) if isinstance(pg, dict)))
        ),
        "has_view_region_scales": bool(view_scales and view_scales.get("pages")),
        "reference_bom_used_for_validation_only": bool(reference_project1_bom),
        "notes": (
            "Gemini takeoff is the primary extractor (vision). Optional detectors add view boxes and "
            "per-view validation + per-view scales. Reference BOM, if provided, is for validation only."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_takeoff_json(merged, out_json)

    if detections_json_out and detections_accum is not None:
        detections_json_out = detections_json_out.expanduser().resolve()
        detections_json_out.parent.mkdir(parents=True, exist_ok=True)
        detections_json_out.write_text(json.dumps(detections_accum, indent=2, ensure_ascii=False), encoding="utf-8")

    if validation_json_out:
        validation_json_out = validation_json_out.expanduser().resolve()
        validation_json_out.parent.mkdir(parents=True, exist_ok=True)
        validation_json_out.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"out_json": str(out_json), "validation": validation}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid takeoff: text-first, vision fallback.")
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--reference-project1-bom", type=Path, default=None)
    ap.add_argument("--out-detections-json", type=Path, default=None)
    ap.add_argument("--out-validation-json", type=Path, default=None)
    args = ap.parse_args()

    run_hybrid_takeoff(
        pdf=args.pdf,
        out_json=args.out_json,
        reference_project1_bom=args.reference_project1_bom,
        detections_json_out=args.out_detections_json,
        validation_json_out=args.out_validation_json,
    )
    print(args.out_json.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

