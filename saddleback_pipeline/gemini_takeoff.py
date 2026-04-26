from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

from saddleback_pipeline.drawing_scales import (
    extract_drawing_scales,
    format_scales_for_prompt,
    merge_scales_into_takeoff,
)
from saddleback_pipeline.detection_merge import (
    has_structural_instances,
    merge_detection_payloads,
)
from saddleback_pipeline.gemini_spatial_detection import (
    format_detection_hints_block,
    run_spatial_detection_for_pages,
)
from saddleback_pipeline.pdf_to_images import render_pdf_to_png_pages


def _detection_hints_meaningful(d: dict | None) -> bool:
    if not d:
        return False
    if has_structural_instances(d):
        return True
    for pg in d.get("pages") or []:
        if isinstance(pg, dict) and pg.get("view_regions"):
            return True
    return False


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def run_takeoff(
    *,
    pdf_path: Path,
    output_json: Path,
    gemini_api_key: str,
    model: str,
    prompt_path: Path,
    schema_csv_path: Path | None,
) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(
            f"INPUT_PDF must be a PDF file path, not a directory. Got: {pdf_path}"
        )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    schema_block = ""
    if schema_csv_path and schema_csv_path.exists():
        schema_block = (
            "\n\n--- REQUIRED OUTPUT SCHEMA (from CSV) ---\n"
            + schema_csv_path.read_text(encoding="utf-8")
        )
    else:
        schema_block = (
            "\n\n--- REQUIRED OUTPUT SCHEMA (no CSV file; use JSON root key \"data\" as specified) ---\n"
            "If a separate column schema is needed, map each entity into the structure in section 12.\n"
        )

    full_prompt = system_prompt + schema_block

    import os

    scales_payload: dict | None = None
    extract_scales = (os.getenv("EXTRACT_DRAWING_SCALES", "true") or "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if extract_scales:
        max_pages_env = (os.getenv("PDF_MAX_PAGES", "") or "").strip()
        max_pages_sc = int(max_pages_env) if max_pages_env else None
        try:
            scales_payload = extract_drawing_scales(pdf_path, max_pages=max_pages_sc)
            scales_block = format_scales_for_prompt(scales_payload)
            full_prompt = full_prompt + "\n\n" + scales_block
            out_scales = (os.getenv("OUTPUT_DRAWING_SCALES_JSON", "") or "").strip()
            if out_scales:
                sp = Path(out_scales).expanduser()
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(
                    json.dumps(scales_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Wrote drawing scales: {sp}", file=sys.stderr)
        except Exception as ex:
            print(
                f"WARNING: Drawing scale extraction failed ({ex}); continuing without scale block.",
                file=sys.stderr,
            )
            scales_payload = None

    use_images = (os.getenv("PDF_AS_IMAGES", "false") or "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    spatial_detection = (os.getenv("GEMINI_SPATIAL_DETECTION", "false") or "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    onnx_raw = (os.getenv("YOLO_ONNX_MODEL", "") or "").strip()
    onnx_ok = bool(onnx_raw and Path(onnx_raw).expanduser().is_file())
    learned_det_path = (os.getenv("LEARNED_DETECTION_JSON", "") or "").strip()
    image_dpi = float(os.getenv("PDF_IMAGE_DPI", "400") or "400")
    max_side = int(os.getenv("PDF_IMAGE_MAX_SIDE_PX", "4096") or "4096")
    max_pages_env = (os.getenv("PDF_MAX_PAGES", "") or "").strip()
    max_pages = int(max_pages_env) if max_pages_env else None
    timeout_sec = int(os.getenv("GEMINI_HTTP_TIMEOUT_SEC", "1800") or "1800")

    def _log_page(idx: int, total: int) -> None:
        print(
            f"  Rasterizing PDF page {idx}/{total} @ up to {image_dpi} DPI (max side {max_side}px)…",
            file=sys.stderr,
        )

    png_pages: list[bytes] | None = None
    # Rasterize when images are used for takeoff, when Gemini spatial runs, or when ONNX/YOLO runs.
    if use_images or spatial_detection or onnx_ok:
        png_pages = render_pdf_to_png_pages(
            pdf_path,
            dpi=image_dpi,
            max_side_px=max_side,
            max_pages=max_pages,
            on_page=_log_page,
        )

    detections_accum: dict | None = None
    out_det = Path(
        os.getenv("OUTPUT_DETECTIONS_JSON", "detections_output.json") or "detections_output.json",
    ).expanduser()

    if spatial_detection:
        if not png_pages:
            raise RuntimeError(
                "GEMINI_SPATIAL_DETECTION requires rasterized pages; enable PDF_AS_IMAGES or "
                "YOLO_ONNX_MODEL (implicit rasterize) so pages exist.",
            )
        det_model = (os.getenv("GEMINI_DETECTION_MODEL", "") or "").strip() or (
            os.getenv("GEMINI_MODEL", "") or "gemini-2.5-flash"
        )
        det_prompt_path = Path(
            os.getenv("GEMINI_DETECTION_PROMPT_PATH", "prompts/spatial_detection.txt")
            or "prompts/spatial_detection.txt",
        )
        if not det_prompt_path.exists():
            raise FileNotFoundError(f"Detection prompt not found: {det_prompt_path}")
        det_prompt_text = det_prompt_path.read_text(encoding="utf-8")
        det_max_out = int(os.getenv("GEMINI_DETECTION_MAX_OUTPUT_TOKENS", "16384") or "16384")
        det_stream = (os.getenv("GEMINI_DETECTION_STREAM", "false") or "false").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        print(
            "Running Gemini spatial-detection pass (bounding boxes as JSON; not a CNN).",
            file=sys.stderr,
        )
        gemini_det = run_spatial_detection_for_pages(
            png_pages=png_pages,
            gemini_api_key=gemini_api_key,
            model=det_model,
            prompt_text=det_prompt_text,
            timeout_sec=timeout_sec,
            max_output_tokens=det_max_out,
            use_stream=det_stream,
        )
        detections_accum = merge_detection_payloads(detections_accum, gemini_det)

    if learned_det_path and Path(learned_det_path).expanduser().is_file():
        learned_obj = json.loads(
            Path(learned_det_path).expanduser().read_text(encoding="utf-8"),
        )
        if isinstance(learned_obj, dict):
            print(
                f"Merging LEARNED_DETECTION_JSON: {learned_det_path}",
                file=sys.stderr,
            )
            detections_accum = merge_detection_payloads(detections_accum, learned_obj)

    if onnx_ok:
        if not png_pages:
            raise RuntimeError(
                "YOLO_ONNX_MODEL requires rasterized pages; enable PDF_AS_IMAGES=true or "
                "GEMINI_SPATIAL_DETECTION=true so the PDF is rasterized once.",
            )
        try:
            from saddleback_pipeline.onnx_yolo_detector import run_yolo_onnx_for_png_pages

            yo_det = run_yolo_onnx_for_png_pages(
                png_pages,
                Path(onnx_raw).expanduser(),
            )
            detections_accum = merge_detection_payloads(detections_accum, yo_det)
        except ImportError as ex:
            print(f"WARNING: ONNX YOLO skipped ({ex}). pip install onnxruntime", file=sys.stderr)

    if detections_accum is not None:
        out_det.parent.mkdir(parents=True, exist_ok=True)
        out_det.write_text(json.dumps(detections_accum, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote merged detections: {out_det}", file=sys.stderr)
        if _detection_hints_meaningful(detections_accum):
            full_prompt = full_prompt + format_detection_hints_block(detections_accum)

    if use_images and scales_payload is not None:
        dpi_note = (
            f"\n\nRaster pages were rendered at PDF_IMAGE_DPI={image_dpi:g} "
            "(use this DPI in the pixel→real formulas above)."
        )
        full_prompt = full_prompt + dpi_note

    if use_images:
        if not png_pages:
            raise RuntimeError("PDF_AS_IMAGES set but no PNG pages (internal error).")
        intro = (
            "The following images are rasterized pages of the same construction drawing PDF, "
            "in order from page 1 to the last page. Treat them as the full document; read every sheet "
            "before extracting quantities. Preserve fine print, callouts, and grid/dimension text.\n"
        )
        image_parts: list[types.Part] = [types.Part.from_text(text=intro)]
        for png in png_pages:
            image_parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
        image_parts.append(types.Part.from_text(text=full_prompt))
        parts_for_model: list[types.Part] = image_parts
        print(
            f"Using {len(png_pages)} high-resolution PNG page(s) (PDF_AS_IMAGES=true).",
            file=sys.stderr,
        )
    else:
        pdf_bytes = pdf_path.read_bytes()
        parts_for_model = [
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=full_prompt),
        ]

    # Long-running: multimodal + your prompt can request a very large JSON (entity expansion).
    # google-genai HttpOptions.timeout is in MILLISECONDS. Passing 1800 meant 1.8s — use seconds in env and convert.

    timeout_ms = timeout_sec * 1000

    mode = "page images + prompt" if use_images else "native PDF + prompt"
    print(
        f"Calling Gemini ({mode}). This often takes several minutes: "
        "multimodal ingest, reasoning, and generating a large JSON. "
        f"HTTP read timeout={timeout_sec}s ({timeout_ms} ms for HttpOptions).",
        file=sys.stderr,
    )

    max_out = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "65536") or "65536")
    use_stream = (os.getenv("GEMINI_STREAM", "false") or "false").lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    contents = [types.Content(role="user", parts=parts_for_model)]
    gen_cfg = types.GenerateContentConfig(
        max_output_tokens=max_out,
        response_mime_type="application/json",
    )

    try:
        # Only ``timeout`` is portable across google-genai versions (some installs forbid
        # httpx_client / client_args on HttpOptions).
        with genai.Client(
            api_key=gemini_api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        ) as client:
            if use_stream:
                print(
                    "Streaming response (first tokens appear sooner; total time is often similar to non-streaming).",
                    file=sys.stderr,
                )
                chunks: list[str] = []
                last_log = 0
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                ):
                    piece = chunk.text or ""
                    if not piece:
                        continue
                    chunks.append(piece)
                    total = sum(len(c) for c in chunks)
                    if total - last_log >= 8000:
                        print(
                            f"  … received ~{total} characters so far",
                            file=sys.stderr,
                        )
                        last_log = total
                text = "".join(chunks).strip()
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                )
                text = (response.text or "").strip()
    except httpx.ReadTimeout:
        print(
            "ERROR: HTTP read timed out before Gemini finished. For long PDF runs, set "
            "GEMINI_STREAM=true (partial output keeps the connection active) and/or "
            "raise GEMINI_HTTP_TIMEOUT_SEC.",
            file=sys.stderr,
        )
        raise
    text = _strip_json_fences(text)
    # Validate JSON
    data = json.loads(text)
    if scales_payload is not None and isinstance(data, dict):
        merge_scales_into_takeoff(data, scales_payload)
        data.setdefault("meta", {})
        if isinstance(data["meta"], dict):
            data["meta"]["raster_dpi_used"] = float(
                os.getenv("PDF_IMAGE_DPI", "400") or "400",
            )
    if not isinstance(data, dict) or "material_summary" not in data:
        print(
            "WARNING: Response JSON should include top-level key material_summary "
            "(fabrication BOM). Re-export and BOM accuracy will be incomplete.",
            file=sys.stderr,
        )
    elif not data.get("material_summary"):
        print(
            "WARNING: material_summary is present but empty.",
            file=sys.stderr,
        )
    auto_w = (os.getenv("STEEL_WEIGHT_ENRICH_AUTO", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if auto_w and isinstance(data, dict):
        wo_raw = (os.getenv("WEIGHT_OVERRIDE_CSV", "") or "").strip()
        wo_path = Path(wo_raw).expanduser() if wo_raw else None
        if wo_path is not None and not wo_path.is_file():
            print(f"WARNING: WEIGHT_OVERRIDE_CSV not found: {wo_path}", file=sys.stderr)
            wo_path = None
        try:
            from saddleback_pipeline.steel_weight_enrichment import enrich_takeoff_payload

            enrich_takeoff_payload(data, weight_override_csv=wo_path)
            print("Applied steel_weight_enrichment (STEEL_WEIGHT_ENRICH_AUTO=true).", file=sys.stderr)
        except Exception as ex:
            print(f"WARNING: steel_weight_enrichment failed ({ex})", file=sys.stderr)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    load_dotenv(".env", override=False)
    import os

    # IMPORTANT: Path("") becomes Path(".") which is truthy — must check the string first.
    raw_pdf = (os.getenv("INPUT_PDF", "") or "").strip()
    if not raw_pdf:
        print(
            "ERROR: INPUT_PDF is not set or is empty. Set it in .env to your PDF filename, e.g.\n"
            '  INPUT_PDF="Pages from SADDLEBACK VILLAGE RAMADAS - Sealed Drawings 8-22-25.pdf"',
            file=sys.stderr,
        )
        return 1

    pdf = Path(raw_pdf).expanduser()
    out = Path(os.getenv("OUTPUT_JSON", "takeoff_output.json") or "takeoff_output.json").expanduser()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview"
    prompt_path = Path(os.getenv("PROMPT_PATH", "prompts/structural_takeoff.txt") or "prompts/structural_takeoff.txt")
    schema_csv = os.getenv("SCHEMA_CSV", "").strip()
    schema_path = Path(schema_csv) if schema_csv else None

    if not key:
        print("ERROR: Set GEMINI_API_KEY", file=sys.stderr)
        return 1

    if not pdf.exists():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        return 1
    if not pdf.is_file():
        print(
            f"ERROR: INPUT_PDF must be a file, not a directory: {pdf}",
            file=sys.stderr,
        )
        return 1

    run_takeoff(
        pdf_path=pdf,
        output_json=out,
        gemini_api_key=key,
        model=model,
        prompt_path=prompt_path,
        schema_csv_path=schema_path,
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
