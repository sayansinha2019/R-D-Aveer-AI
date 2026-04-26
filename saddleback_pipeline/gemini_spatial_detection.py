"""Optional spatial localization pass using the Gemini API (multimodal JSON).

This is NOT a classical CNN detector (YOLO/Mask R-CNN). The model proposes
normalized bounding boxes for views and visible structural instances — useful
as hints and for consistency checks with the main takeoff prompt.

Merged with ONNX/YOLO outputs in ``gemini_takeoff`` (see ``YOLO_ONNX_MODEL``, ``detection_merge``).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

# Enforced via Gemini structured output (response_json_schema) so the model cannot emit invalid JSON.
SPATIAL_DETECTION_RESPONSE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "page_index": {"type": "integer"},
        "width_px": {"type": "integer"},
        "height_px": {"type": "integer"},
        "view_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "view_id": {"type": "string"},
                    "label": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x_min": {"type": "number"},
                            "y_min": {"type": "number"},
                            "x_max": {"type": "number"},
                            "y_max": {"type": "number"},
                        },
                        "required": ["x_min", "y_min", "x_max", "y_max"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["view_id", "label", "bbox", "confidence"],
            },
        },
        "structural_instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "view_id": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x_min": {"type": "number"},
                            "y_min": {"type": "number"},
                            "x_max": {"type": "number"},
                            "y_max": {"type": "number"},
                        },
                        "required": ["x_min", "y_min", "x_max", "y_max"],
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["class", "view_id", "bbox", "confidence"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "page_index",
        "width_px",
        "height_px",
        "view_regions",
        "structural_instances",
    ],
}


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _strip_trailing_commas(s: str) -> str:
    """Remove JSON trailing commas before } or ] (common model slip)."""
    prev = None
    out = s
    while prev != out:
        prev = out
        out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def parse_model_json_object(text: str) -> dict:
    """Parse Gemini JSON; repair minor formatting issues before failing."""
    t = _strip_json_fences(text)
    t = _strip_trailing_commas(t)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        # Sometimes extra prose after the object — take first top-level { ... }
        start = t.find("{")
        if start < 0:
            raise
        depth = 0
        end = -1
        for i, ch in enumerate(t[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            raise
        frag = _strip_trailing_commas(t[start:end])
        obj = json.loads(frag)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object at top level")
    return obj


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    """Read width/height from PNG IHDR without Pillow."""
    if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG byte string")
    w = int.from_bytes(png_bytes[16:20], "big")
    h = int.from_bytes(png_bytes[20:24], "big")
    return w, h


def run_spatial_detection_for_pages(
    *,
    png_pages: list[bytes],
    gemini_api_key: str,
    model: str,
    prompt_text: str,
    timeout_sec: int,
    max_output_tokens: int = 16384,
    use_stream: bool = False,
) -> dict:
    """One Gemini request per page; returns ``{"version": 1, "pages": [...]}``."""
    if not png_pages:
        raise ValueError("png_pages is empty")

    timeout_ms = timeout_sec * 1000

    pages_out: list[dict] = []
    total = len(png_pages)

    with genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    ) as client:
        for i, png in enumerate(png_pages):
            page_1based = i + 1
            try:
                w_px, h_px = _png_dimensions(png)
            except ValueError:
                w_px, h_px = 0, 0

            header = (
                f"This is page {page_1based} of {total} in the document. "
                f"The image pixel size is {w_px} x {h_px}.\n\n"
            )
            parts: list[types.Part] = [
                types.Part.from_bytes(data=png, mime_type="image/png"),
                types.Part.from_text(text=header + prompt_text),
            ]
            contents = [types.Content(role="user", parts=parts)]
            gen_cfg = types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=SPATIAL_DETECTION_RESPONSE_JSON_SCHEMA,
            )
            print(
                f"  Spatial detection: page {page_1based}/{total} ({model})…",
                file=sys.stderr,
            )
            # Structured JSON mode is unreliable with streaming in some SDK paths; use one shot.
            if use_stream:
                print(
                    "  (GEMINI_DETECTION_STREAM ignored: using non-stream for structured JSON.)",
                    file=sys.stderr,
                )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
            text = (response.text or "").strip()

            try:
                page_obj = parse_model_json_object(text)
            except (json.JSONDecodeError, ValueError) as e:
                dbg = Path("detection_raw_response_failed.txt")
                dbg.write_text(text, encoding="utf-8")
                raise ValueError(
                    f"Page {page_1based}: invalid JSON from detection model "
                    f"(raw response saved to {dbg.resolve()}). "
                    f"Try GEMINI_DETECTION_MODEL=gemini-2.5-pro. Underlying: {e}"
                ) from e
            page_obj.setdefault("page_index", page_1based)
            if w_px and h_px:
                page_obj.setdefault("width_px", w_px)
                page_obj.setdefault("height_px", h_px)
            pages_out.append(page_obj)

    return {"version": 1, "detector": "gemini_multimodal", "pages": pages_out}


def format_detection_hints_block(detections: dict, *, max_chars: int = 120_000) -> str:
    """Appendix text injected into the main takeoff prompt."""
    try:
        s = json.dumps(detections, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(detections)
    if len(s) > max_chars:
        s = s[: max_chars - 80] + "\n… [truncated for prompt size]\n"
    return (
        "\n\n--- SPATIAL DETECTION HINTS (automated pass; VERIFY against images) ---\n"
        "These boxes are model-proposed regions/instances. Use them to reduce omissions "
        "and to cross-check counts; correct any mistakes using the drawing.\n"
        f"{s}\n"
        "--- END SPATIAL DETECTION HINTS ---\n"
    )
