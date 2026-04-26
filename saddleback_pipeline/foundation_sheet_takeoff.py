"""Gemini takeoff with a **foundation/civil sheet** supplement (footing schedules, CIP walls).

Standard ``structural_takeoff.txt`` often yields empty ``data`` on pure foundation plans;
this pass appends explicit instructions so schedules (F4.0, …) still serialize to JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

from saddleback_pipeline.drawing_scales import (
    extract_drawing_scales,
    format_scales_for_prompt,
    merge_scales_into_takeoff,
)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


FOUNDATION_PROMPT_APPEND = """
--- PROJECT — FOUNDATION / CIVIL SHEET ---
This PDF may be a FOUNDATION PLAN with a printed FOOTING SCHEDULE (marks like F4.0, F13.0,
F13.0B, F10X7, F13X15, MF-1, etc.) and wall/footing notes (T/FTG, CIP walls).

Standard steel extraction may find no W-shapes. You MUST still return non-empty ``data`` and
``material_summary`` whenever schedule tables or callouts exist.

Rules:
1) For EACH row in the footing schedule table (every MARK), emit at least ONE ``data`` object:
   element_type "Footing", parent_group "Foundations", section = mark as shown (e.g. "F13.0"),
   use plate_length / plate_width / plate_thickness for LENGTH / FTG WIDTH / THICKNESS when appropriate,
   quantity 1 per schedule line type (do not guess plan counts), piece_mark = mark when shown,
   material = rebar notation from schedule when present.

2) For "8\\" THK CIP CONC WALL" (or similar), emit ``data`` with element_type "Plate",
   parent_group "Walls", section from the note.

3) ``material_summary`` MUST list rolled rows for each distinct footing mark and wall note modeled.

Return ONLY valid JSON with keys "data" and "material_summary". No markdown fences.
"""


def run_foundation_sheet_takeoff(
    *,
    pdf_path: Path,
    output_json: Path,
    gemini_api_key: str,
    model: str,
    prompt_path: Path,
    schema_csv_path: Path | None,
) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    system_prompt = prompt_path.read_text(encoding="utf-8")
    schema_block = ""
    if schema_csv_path and schema_csv_path.exists():
        schema_block = (
            "\n\n--- REQUIRED OUTPUT SCHEMA (from CSV) ---\n"
            + schema_csv_path.read_text(encoding="utf-8")
        )
    full_prompt = system_prompt + schema_block + FOUNDATION_PROMPT_APPEND

    scales_payload = None
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
            full_prompt = full_prompt + "\n\n" + format_scales_for_prompt(scales_payload)
        except Exception as ex:
            print(f"WARNING: scales for foundation sheet ({ex})", file=sys.stderr)
            scales_payload = None

    pdf_bytes = pdf_path.read_bytes()
    parts_for_model = [
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        types.Part.from_text(text=full_prompt),
    ]
    timeout_sec = int(os.getenv("GEMINI_HTTP_TIMEOUT_SEC", "1800") or "1800")
    timeout_ms = timeout_sec * 1000
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
    print(
        f"Foundation-sheet Gemini pass (native PDF), timeout={timeout_sec}s, stream={use_stream}",
        file=sys.stderr,
    )
    with genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    ) as client:
        if use_stream:
            chunks: list[str] = []
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=gen_cfg,
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            text = "".join(chunks).strip()
        else:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
            text = (response.text or "").strip()

    text = _strip_json_fences(text)
    data = json.loads(text)
    if scales_payload is not None and isinstance(data, dict):
        merge_scales_into_takeoff(data, scales_payload)
        data.setdefault("meta", {})
        if isinstance(data["meta"], dict):
            data["meta"]["foundation_sheet_prompt"] = True
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
