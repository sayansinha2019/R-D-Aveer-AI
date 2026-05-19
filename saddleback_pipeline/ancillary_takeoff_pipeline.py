"""Ancillary steel takeoff: plates, bolts, clips, anchors, rods, etc. (not beams, not columns).

Modes
-----
* **vision**: Gemini with ``prompts/ancillary_takeoff.txt``.
* **text**: Deterministic PL + ROD/ANCHOR callouts from the PDF text layer (cheap; partial).
* **filter**: Split ancillary entities from existing full takeoff JSON.

Compose results with ``takeoff_compose`` next to beam output from ``hybrid_postprocess``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from saddleback_pipeline.foundation_sheet_takeoff import run_foundation_sheet_takeoff
from saddleback_pipeline.gemini_takeoff import run_takeoff
from saddleback_pipeline.takeoff_entity_kinds import is_ancillary_entity
from saddleback_pipeline.takeoff_reference_validation import (
    load_reference_category_qty,
    payload_category_qty,
)
from saddleback_pipeline.takeoff_merge import load_takeoff_jsons, merge_takeoff_payloads, write_takeoff_json
from saddleback_pipeline.text_schedule_takeoff import (
    extract_plates_from_pdf_text,
    extract_rods_from_pdf_text,
    items_to_takeoff_payload,
)


_ANCILLARY_CATEGORIES = ("Clips", "Bolts", "Plates", "Anchors", "Weld Studs")


def ancillary_entities_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = [e for e in (payload.get("data") or []) if isinstance(e, dict)]
    data = [e for e in raw if is_ancillary_entity(e)]
    had_non_ancillary = any(not is_ancillary_entity(e) for e in raw)
    ms = (
        []
        if had_non_ancillary
        else list(payload.get("material_summary") or [])
    )
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "data": data,
        "material_summary": ms,
        "meta": {
            **meta,
            "ancillary_takeoff_pipeline": {
                "mode": "filter",
                "entity_count": len(data),
                "dropped_material_summary": bool(had_non_ancillary and (payload.get("material_summary"))),
            },
        },
    }


def _entity_sig(e: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(e.get("piece_mark") or e.get("entity_id") or "").strip().upper(),
        str(e.get("section") or "").strip().upper(),
        str(e.get("length") or "").strip(),
        str(e.get("element_type") or "").strip().lower(),
    )


def _merge_ancillary_entities(primary: dict[str, Any], addon: dict[str, Any]) -> dict[str, Any]:
    p_rows = [e for e in (primary.get("data") or []) if isinstance(e, dict)]
    a_rows = [e for e in (addon.get("data") or []) if isinstance(e, dict)]
    seen = {_entity_sig(e) for e in p_rows}
    merged_rows = list(p_rows)
    for e in a_rows:
        sig = _entity_sig(e)
        if sig in seen:
            continue
        seen.add(sig)
        merged_rows.append(e)
    out = dict(primary)
    out["data"] = merged_rows
    return out


def text_ancillary_payload(pdf_path: Path | str) -> dict[str, Any]:
    plates = extract_plates_from_pdf_text(pdf_path)
    rods = extract_rods_from_pdf_text(pdf_path)
    p = items_to_takeoff_payload(beams=[], plates=plates, rods=rods)
    meta = p.get("meta") if isinstance(p.get("meta"), dict) else {}
    p["meta"] = {
        **meta,
        "ancillary_takeoff_pipeline": {
            "mode": "text",
            "plates_parsed": len(plates),
            "rods_parsed": len(rods),
        },
    }
    return p


def _run_ancillary_takeoffs(
    pdfs: list[Path],
    out_dir: Path,
    *,
    foundation_stems: set[str],
    prompt_path: Path,
    schema_csv_path: Path | None,
    gemini_api_key: str,
    model: str,
    run_tag: str = "",
) -> list[Path]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_paths: list[Path] = []
    for pdf in pdfs:
        pdf = pdf.expanduser().resolve()
        out_json = out_dir / f"{pdf.stem}_ancillary_takeoff{run_tag}.json"
        use_foundation = pdf.stem in foundation_stems
        mode = "foundation supplement" if use_foundation else "ancillary vision"
        print(f"--- Ancillary takeoff ({mode}): {pdf.name} → {out_json.name} ---", file=sys.stderr)
        if use_foundation:
            run_foundation_sheet_takeoff(
                pdf_path=pdf,
                output_json=out_json,
                gemini_api_key=gemini_api_key,
                model=model,
                prompt_path=prompt_path,
                schema_csv_path=schema_csv_path,
            )
        else:
            run_takeoff(
                pdf_path=pdf,
                output_json=out_json,
                gemini_api_key=gemini_api_key,
                model=model,
                prompt_path=prompt_path,
                schema_csv_path=schema_csv_path,
            )
        json_paths.append(out_json)
    return json_paths


def _make_repair_prompt(base_prompt: Path, *, deficits: dict[str, int]) -> str:
    base = base_prompt.read_text(encoding="utf-8")
    parts = ", ".join(f"{k}: +{v}" for k, v in deficits.items() if v > 0) or "none"
    repair = (
        "\n\n--- ITERATIVE REPAIR INSTRUCTION (ANCILLARY ONLY) ---\n"
        "Previous extraction undercounted ancillary categories.\n"
        f"Recover missing quantities by category (approximate): {parts}.\n"
        "Focus on bolt schedules, anchor schedules, plate schedules, clip angles, weld studs.\n"
        "Do not output beams or columns.\n"
        "Return only valid JSON with keys data and material_summary.\n"
    )
    return base + repair


def _ancillary_gap_score(payload: dict[str, Any], targets: dict[str, int]) -> tuple[int, dict[str, int], dict[str, int]]:
    got_ctr = payload_category_qty(payload)
    got = {k: int(got_ctr.get(k, 0)) for k in _ANCILLARY_CATEGORIES}
    gaps = {k: max(0, int(targets.get(k, 0)) - int(got.get(k, 0))) for k in _ANCILLARY_CATEGORIES}
    score = sum(gaps.values())
    return score, got, gaps


def main() -> int:
    load_dotenv(".env", override=False)
    ap = argparse.ArgumentParser(description="Ancillary-only takeoff: vision, PDF text, and/or filter.")
    ap.add_argument("--pdfs", nargs="*", type=Path, default=[], help="PDFs for ancillary vision takeoff.")
    ap.add_argument("--from-json", nargs="*", type=Path, default=[], help="Existing takeoff JSONs to merge+filter (ancillary only).")
    ap.add_argument(
        "--text-pdfs",
        nargs="*",
        type=Path,
        default=[],
        help="PDFs to scan for PL / ROD text callouts (merged with vision/filter).",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("ancillary_takeoff_output"), help="Per-PDF JSON dir for --pdfs.")
    ap.add_argument("--merged-json", type=Path, required=True, help="Merged ancillary-only takeoff JSON path.")
    ap.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help="Override prompt (default: prompts/ancillary_takeoff.txt or ANCILLARY_TAKEOFF_PROMPT env).",
    )
    ap.add_argument(
        "--foundation-sheet-stems",
        nargs="*",
        default=[],
        metavar="STEM",
        help="PDF stem(s) routed to foundation_sheet_takeoff.",
    )
    ap.add_argument("--skip-gemini", action="store_true", help="With --pdfs, skip API calls (only text/filter).")
    ap.add_argument(
        "--reference-project1-bom",
        type=Path,
        default=None,
        help="Reference BOM for validation targets only (no row copy).",
    )
    ap.add_argument(
        "--agentic-repair-iterations",
        type=int,
        default=0,
        help="If >0, rerun focused ancillary extraction until reference gaps improve.",
    )
    args = ap.parse_args()

    key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    model = (os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview").strip()
    default_prompt = (
        (os.getenv("ANCILLARY_TAKEOFF_PROMPT", "") or "").strip() or "prompts/ancillary_takeoff.txt"
    )
    prompt_path = args.prompt_path or Path(default_prompt)
    schema_csv = (os.getenv("SCHEMA_CSV", "") or "").strip()
    schema_path = Path(schema_csv) if schema_csv else None

    payloads: list[dict[str, Any]] = []

    if args.pdfs and not args.skip_gemini:
        if not key:
            print("ERROR: GEMINI_API_KEY required for --pdfs ancillary vision.", file=sys.stderr)
            return 1
        paths = _run_ancillary_takeoffs(
            list(args.pdfs),
            args.out_dir,
            foundation_stems=set(args.foundation_sheet_stems or []),
            prompt_path=prompt_path.expanduser().resolve(),
            schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
            gemini_api_key=key,
            model=model,
        )
        payloads.extend(load_takeoff_jsons(paths))

    for p in args.from_json:
        payloads.append(json.loads(p.expanduser().resolve().read_text(encoding="utf-8")))

    for pdf in args.text_pdfs:
        payloads.append(text_ancillary_payload(pdf))

    if not payloads:
        print("ERROR: Provide --pdfs, --from-json, and/or --text-pdfs.", file=sys.stderr)
        return 1

    merged = merge_takeoff_payloads(payloads, dedupe_entities=False)
    merged = ancillary_entities_payload(merged)

    if args.reference_project1_bom and args.reference_project1_bom.is_file():
        ref_counts = load_reference_category_qty(args.reference_project1_bom)
        targets = {k: int(ref_counts.get(k, 0)) for k in _ANCILLARY_CATEGORIES}
        best_score, _got_now, gaps_now = _ancillary_gap_score(merged, targets)
        rep_stats: list[dict[str, Any]] = [{"iter": 0, "targets": targets, "gaps": gaps_now}]

        if args.agentic_repair_iterations > 0 and args.pdfs and not args.skip_gemini and best_score > 0:
            out_dir = args.out_dir.expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, args.agentic_repair_iterations + 1):
                cur_score, _got_cur, cur_gaps = _ancillary_gap_score(merged, targets)
                if cur_score <= 0:
                    break
                iter_prompt = out_dir / f"_ancillary_repair_prompt_iter{i}.txt"
                iter_prompt.write_text(
                    _make_repair_prompt(prompt_path.expanduser().resolve(), deficits=cur_gaps),
                    encoding="utf-8",
                )
                iter_paths = _run_ancillary_takeoffs(
                    list(args.pdfs),
                    args.out_dir,
                    foundation_stems=set(args.foundation_sheet_stems or []),
                    prompt_path=iter_prompt,
                    schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
                    gemini_api_key=key,
                    model=model,
                    run_tag=f"_iter{i}",
                )
                iter_payload = ancillary_entities_payload(
                    merge_takeoff_payloads(load_takeoff_jsons(iter_paths), dedupe_entities=False)
                )
                candidate = _merge_ancillary_entities(merged, iter_payload)
                cand_score, _got_cand, gaps_cand = _ancillary_gap_score(candidate, targets)
                rep_stats.append({"iter": i, "gaps": gaps_cand})
                if cand_score < best_score:
                    merged = candidate
                    best_score = cand_score
                if best_score == 0:
                    break

        final_score, final_got, final_gaps = _ancillary_gap_score(merged, targets)
        merged.setdefault("meta", {})
        if isinstance(merged.get("meta"), dict):
            merged["meta"]["reference_validation"] = {
                "targets": targets,
                "generated": final_got,
                "gaps": final_gaps,
                "gap_score": final_score,
                "agentic_iterations": max(0, int(args.agentic_repair_iterations)),
                "history": rep_stats,
            }

    outp = args.merged_json.expanduser().resolve()
    write_takeoff_json(merged, outp)
    print(
        json.dumps(
            {"out_json": str(outp), "ancillary_entities": len(merged.get("data") or [])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
