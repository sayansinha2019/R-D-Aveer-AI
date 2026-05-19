"""Column-only takeoff pipeline (independent from beams and ancillary steel).

Modes
-----
* **vision**: Run Gemini with ``prompts/column_takeoff.txt`` per PDF (same env as ``gemini_takeoff``).
* **filter**: Split column entities out of existing full takeoff JSON (no API calls).

Outputs a normal takeoff JSON (``data`` + ``material_summary`` + ``meta``) containing only columns.
Compose with ``takeoff_compose`` alongside beam output from ``hybrid_postprocess``.
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
from saddleback_pipeline.takeoff_entity_kinds import is_column_entity
from saddleback_pipeline.takeoff_reference_validation import (
    load_reference_category_qty,
    payload_category_qty,
)
from saddleback_pipeline.takeoff_merge import load_takeoff_jsons, merge_takeoff_payloads, write_takeoff_json


def column_entities_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of payload with only column rows in ``data``."""
    raw = [e for e in (payload.get("data") or []) if isinstance(e, dict)]
    data = [e for e in raw if is_column_entity(e)]
    had_non_column = any(not is_column_entity(e) for e in raw)
    # Avoid pulling beam/plate summary lines when slicing a mixed full takeoff.
    ms = (
        []
        if had_non_column
        else list(payload.get("material_summary") or [])
    )
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        "data": data,
        "material_summary": ms,
        "meta": {
            **meta,
            "column_takeoff_pipeline": {
                "mode": "filter",
                "entity_count": len(data),
                "dropped_material_summary": bool(had_non_column and (payload.get("material_summary"))),
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


def _merge_column_entities(primary: dict[str, Any], addon: dict[str, Any]) -> dict[str, Any]:
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


def _run_column_takeoffs(
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
        out_json = out_dir / f"{pdf.stem}_columns_takeoff{run_tag}.json"
        use_foundation = pdf.stem in foundation_stems
        mode = "foundation supplement" if use_foundation else "column vision"
        print(f"--- Column takeoff ({mode}): {pdf.name} → {out_json.name} ---", file=sys.stderr)
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


def _make_repair_prompt(base_prompt: Path, *, missing_columns: int) -> str:
    base = base_prompt.read_text(encoding="utf-8")
    repair = (
        "\n\n--- ITERATIVE REPAIR INSTRUCTION (COLUMNS ONLY) ---\n"
        "Previous extraction undercounted columns.\n"
        f"Recover approximately {missing_columns} additional columns from drawings/schedules.\n"
        "Focus on column schedules, piece marks, and member callouts; do not output beams or ancillary items.\n"
        "Return only valid JSON with keys data and material_summary.\n"
    )
    return base + repair


def main() -> int:
    load_dotenv(".env", override=False)
    ap = argparse.ArgumentParser(description="Column-only takeoff: vision (Gemini) and/or filter from existing JSON.")
    ap.add_argument("--pdfs", nargs="*", type=Path, default=[], help="PDFs for column vision takeoff.")
    ap.add_argument("--from-json", nargs="*", type=Path, default=[], help="Existing takeoff JSONs to merge+filter (columns only).")
    ap.add_argument("--out-dir", type=Path, default=Path("column_takeoff_output"), help="Per-PDF JSON directory when using --pdfs.")
    ap.add_argument("--merged-json", type=Path, required=True, help="Merged column-only takeoff JSON path.")
    ap.add_argument(
        "--prompt-path",
        type=Path,
        default=None,
        help="Override prompt (default: prompts/column_takeoff.txt or COLUMN_TAKEOFF_PROMPT env).",
    )
    ap.add_argument(
        "--foundation-sheet-stems",
        nargs="*",
        default=[],
        metavar="STEM",
        help="PDF stem(s) routed to foundation_sheet_takeoff (same as integrated_pipeline).",
    )
    ap.add_argument("--skip-gemini", action="store_true", help="With --pdfs, skip API calls (only use --from-json).")
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
        help="If >0, rerun focused column extraction until reference gap improves.",
    )
    args = ap.parse_args()

    key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    model = (os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview").strip()
    default_prompt = (
        (os.getenv("COLUMN_TAKEOFF_PROMPT", "") or "").strip() or "prompts/column_takeoff.txt"
    )
    prompt_path = args.prompt_path or Path(default_prompt)
    schema_csv = (os.getenv("SCHEMA_CSV", "") or "").strip()
    schema_path = Path(schema_csv) if schema_csv else None

    json_paths: list[Path] = []
    if args.pdfs and not args.skip_gemini:
        if not key:
            print("ERROR: GEMINI_API_KEY required for --pdfs column vision.", file=sys.stderr)
            return 1
        json_paths.extend(
            _run_column_takeoffs(
                list(args.pdfs),
                args.out_dir,
                foundation_stems=set(args.foundation_sheet_stems or []),
                prompt_path=prompt_path.expanduser().resolve(),
                schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
                gemini_api_key=key,
                model=model,
            )
        )

    for p in args.from_json:
        json_paths.append(p.expanduser().resolve())

    if not json_paths:
        print("ERROR: Provide --pdfs (with Gemini) and/or --from-json.", file=sys.stderr)
        return 1

    payloads = load_takeoff_jsons(json_paths)
    merged = merge_takeoff_payloads(payloads, dedupe_entities=False)
    merged = column_entities_payload(merged)

    if args.reference_project1_bom and args.reference_project1_bom.is_file():
        ref_counts = load_reference_category_qty(args.reference_project1_bom)
        target = int(ref_counts.get("Columns", 0))
        current = int(payload_category_qty(merged).get("Columns", 0))
        best_gap = abs(target - current)
        rep_stats: list[dict[str, int]] = [{"iter": 0, "columns": current, "target": target}]

        if args.agentic_repair_iterations > 0 and args.pdfs and not args.skip_gemini and target > current:
            out_dir = args.out_dir.expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, args.agentic_repair_iterations + 1):
                miss = max(0, target - int(payload_category_qty(merged).get("Columns", 0)))
                if miss <= 0:
                    break
                iter_prompt = out_dir / f"_column_repair_prompt_iter{i}.txt"
                iter_prompt.write_text(
                    _make_repair_prompt(prompt_path.expanduser().resolve(), missing_columns=miss),
                    encoding="utf-8",
                )
                iter_paths = _run_column_takeoffs(
                    list(args.pdfs),
                    args.out_dir,
                    foundation_stems=set(args.foundation_sheet_stems or []),
                    prompt_path=iter_prompt,
                    schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
                    gemini_api_key=key,
                    model=model,
                    run_tag=f"_iter{i}",
                )
                iter_payload = column_entities_payload(
                    merge_takeoff_payloads(load_takeoff_jsons(iter_paths), dedupe_entities=False)
                )
                candidate = _merge_column_entities(merged, iter_payload)
                cand_cols = int(payload_category_qty(candidate).get("Columns", 0))
                cand_gap = abs(target - cand_cols)
                rep_stats.append({"iter": i, "columns": cand_cols, "target": target})
                if cand_gap < best_gap:
                    merged = candidate
                    best_gap = cand_gap
                if best_gap == 0:
                    break

        merged.setdefault("meta", {})
        if isinstance(merged.get("meta"), dict):
            merged["meta"]["reference_validation"] = {
                "target_columns": target,
                "generated_columns": int(payload_category_qty(merged).get("Columns", 0)),
                "agentic_iterations": max(0, int(args.agentic_repair_iterations)),
                "history": rep_stats,
            }

    outp = args.merged_json.expanduser().resolve()
    write_takeoff_json(merged, outp)
    print(
        json.dumps(
            {"out_json": str(outp), "column_entities": len(merged.get("data") or [])},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
