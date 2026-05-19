"""Post-process a Gemini takeoff using PDF-text beams as a safety net.

Goal
----
Some PDFs have reliable beam callouts in the text layer (e.g. ``W14X22 (20)``) but vision
takeoff can undercount or drift run-to-run. This post-step:

* Counts beam-like entities in a Gemini takeoff JSON
* Extracts beam entities from the PDF text layer (deterministic)
* If Gemini beams < text beams, swaps ONLY the beam entities to the text-derived list
  (keeps columns/plates/bolts/anchors/etc. from Gemini).

Reference BOM is NOT used here (validation-only elsewhere).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from saddleback_pipeline.text_schedule_takeoff import extract_w_members_from_pdf_text, members_to_takeoff_payload


def _is_beam_like(et: str) -> bool:
    t = (et or "").strip().lower()
    return t == "beam" or "beam" in t or t in {"rafter", "girder", "joist"}


def _beam_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in payload.get("data") or []:
        if isinstance(e, dict) and _is_beam_like(str(e.get("element_type") or "")):
            out.append(e)
    return out


def _non_beam_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in payload.get("data") or []:
        if isinstance(e, dict) and not _is_beam_like(str(e.get("element_type") or "")):
            out.append(e)
    return out


def _text_beam_entities(pdf: Path) -> list[dict[str, Any]]:
    """Deterministic beams from PDF text layer, only when a length is parsed."""
    members = [m for m in extract_w_members_from_pdf_text(pdf) if m.get("length")]
    p = members_to_takeoff_payload(members)
    return [e for e in (p.get("data") or []) if isinstance(e, dict)]


def swap_beams_to_text_if_needed(
    payload: dict[str, Any],
    *,
    pdf_for_text: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (updated_payload, stats)."""
    gb = _beam_entities(payload)
    tb = _text_beam_entities(pdf_for_text)
    swapped = bool(tb and len(tb) > len(gb))
    out = payload
    if swapped:
        out = {**{k: v for k, v in payload.items() if k != "data"}}
        out["data"] = [*tb, *_non_beam_entities(payload)]
    out.setdefault("meta", {})
    if isinstance(out["meta"], dict):
        out["meta"]["hybrid_text_beam_swap"] = {
            "performed": swapped,
            "gemini_beam_count": len(gb),
            "text_beam_count": len(tb),
            "pdf_used_for_text": str(pdf_for_text),
        }
    return out, dict(out["meta"]["hybrid_text_beam_swap"])  # type: ignore[index]


def main() -> int:
    ap = argparse.ArgumentParser(description="Swap beam entities to PDF-text beams when Gemini undercounts.")
    ap.add_argument("--in-json", type=Path, required=True, help="Gemini takeoff JSON (merged)")
    ap.add_argument("--pdf-for-text", type=Path, required=True, help="PDF used to parse beam callouts (e.g. S121.pdf)")
    ap.add_argument("--out-json", type=Path, required=True, help="Output JSON path")
    args = ap.parse_args()

    payload = json.loads(args.in_json.expanduser().resolve().read_text(encoding="utf-8"))
    updated, stats = swap_beams_to_text_if_needed(payload, pdf_for_text=args.pdf_for_text.expanduser().resolve())
    outp = args.out_json.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_json": str(outp), "swap": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

