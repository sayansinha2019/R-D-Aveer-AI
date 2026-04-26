"""One-shot pipeline: Gemini takeoff → Material Summary reconcile → optional Project-1 BOM align
→ optional steel weight enrichment → Excel export.

Recommended env (or rely on defaults below for missing keys)::

    PDF_AS_IMAGES=true
    GEMINI_SPATIAL_DETECTION=true
    GEMINI_DETECTION_MAX_OUTPUT_TOKENS=32768
    RECONCILE_TOLERANCE_INCHES=12
    REFERENCE_BOM_XLSX=.../Material Summary.xlsx
    ALIGN_REFERENCE_PROJECT1_BOM=.../Project-1 - BOM.xlsx   # optional shop BOM for weight/grade
    ALIGN_REFERENCE_TOLERANCE_INCHES=6
    OUTPUT_JSON=takeoff_output.json
    OUTPUT_JSON_RECONCILED=takeoff_output_reconciled.json
    OUTPUT_XLSX=takeoff_output.xlsx

Usage::

    python -m saddleback_pipeline.full_pipeline
    python -m saddleback_pipeline.full_pipeline --no-spatial   # skip detection pass (faster)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from saddleback_pipeline.gemini_takeoff import run_takeoff
from saddleback_pipeline.json_to_xlsx import export_takeoff_json_to_xlsx
from saddleback_pipeline.material_summary_reconcile import run_reconcile_file
from saddleback_pipeline.project1_reference_align import align_takeoff_json_file as align_project1_reference_json


def _apply_best_accuracy_defaults(*, enable_spatial: bool) -> None:
    """Fill unset env vars for strongest pipeline (does not override .env)."""
    os.environ.setdefault("PDF_AS_IMAGES", "true")
    if enable_spatial:
        os.environ.setdefault("GEMINI_SPATIAL_DETECTION", "true")
        os.environ.setdefault("GEMINI_DETECTION_MAX_OUTPUT_TOKENS", "32768")
    os.environ.setdefault("RECONCILE_TOLERANCE_INCHES", "12")
    os.environ.setdefault("OUTPUT_JSON", "takeoff_output.json")
    os.environ.setdefault("OUTPUT_JSON_RECONCILED", "takeoff_output_reconciled.json")
    os.environ.setdefault("OUTPUT_XLSX", "takeoff_output.xlsx")
    os.environ.setdefault("OUTPUT_DRAWING_SCALES_JSON", "drawing_scales.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full takeoff: Gemini → reconcile → optional Project-1 BOM align → steel enrich → XLSX.",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help="Disable Gemini spatial-detection pass (saves time/API calls).",
    )
    parser.add_argument(
        "--align-reference-project1-bom",
        type=Path,
        default=None,
        help="Optional Project-1 Bill of Materials xlsx: fill weight/grade from matching rows "
        "(overrides env ALIGN_REFERENCE_PROJECT1_BOM).",
    )
    parser.add_argument(
        "--align-reference-tolerance-inches",
        type=float,
        default=None,
        help="Length tolerance for align (default: env ALIGN_REFERENCE_TOLERANCE_INCHES or 6).",
    )
    parser.add_argument(
        "--align-fill-piecemarks",
        action="store_true",
        help="Copy reference piecemarks into empty piece_mark fields when aligning.",
    )
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    _apply_best_accuracy_defaults(enable_spatial=not args.no_spatial)
    if args.no_spatial:
        os.environ["GEMINI_SPATIAL_DETECTION"] = "false"

    raw_pdf = (os.getenv("INPUT_PDF", "") or "").strip()
    if not raw_pdf:
        print(
            "ERROR: INPUT_PDF is not set. Add it to .env.",
            file=sys.stderr,
        )
        return 1

    pdf = Path(raw_pdf).expanduser()
    out_json = Path(os.getenv("OUTPUT_JSON", "takeoff_output.json") or "takeoff_output.json").expanduser()
    reconciled = Path(
        os.getenv("OUTPUT_JSON_RECONCILED", "takeoff_output_reconciled.json")
        or "takeoff_output_reconciled.json",
    ).expanduser()
    xlsx_path = Path(os.getenv("OUTPUT_XLSX", "takeoff_output.xlsx") or "takeoff_output.xlsx").expanduser()

    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview"
    prompt_path = Path(os.getenv("PROMPT_PATH", "prompts/structural_takeoff.txt") or "prompts/structural_takeoff.txt")
    schema_csv = os.getenv("SCHEMA_CSV", "").strip()
    schema_path = Path(schema_csv) if schema_csv else None
    ref_xlsx = (os.getenv("REFERENCE_BOM_XLSX", "") or "").strip()
    ref_path = Path(ref_xlsx).expanduser() if ref_xlsx else None

    tol = float(os.getenv("RECONCILE_TOLERANCE_INCHES", "12") or "12")

    if not key:
        print("ERROR: Set GEMINI_API_KEY in .env", file=sys.stderr)
        return 1
    if not pdf.is_file():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        return 1

    print(
        "--- Step 1: Gemini takeoff (images=%s, spatial=%s) ---"
        % (
            os.getenv("PDF_AS_IMAGES", ""),
            os.getenv("GEMINI_SPATIAL_DETECTION", ""),
        ),
        file=sys.stderr,
    )
    run_takeoff(
        pdf_path=pdf,
        output_json=out_json,
        gemini_api_key=key,
        model=model,
        prompt_path=prompt_path,
        schema_csv_path=schema_path,
    )
    print(f"Wrote: {out_json}", file=sys.stderr)

    final_json = out_json
    if ref_path and ref_path.is_file():
        print(
            f"--- Step 2: Reconcile material_summary to reference ({tol:g}\") ---",
            file=sys.stderr,
        )
        stats = run_reconcile_file(
            input_json=out_json,
            output_json=reconciled,
            reference_xlsx=ref_path,
            tol_inches=tol,
        )
        print(f"Reconcile stats: {stats}", file=sys.stderr)
        final_json = reconciled
    else:
        print(
            "WARNING: REFERENCE_BOM_XLSX missing — skipping reconcile; "
            "XLSX will use raw takeoff JSON.",
            file=sys.stderr,
        )

    align_path = args.align_reference_project1_bom
    if align_path is None:
        ar = (os.getenv("ALIGN_REFERENCE_PROJECT1_BOM", "") or "").strip()
        align_path = Path(ar).expanduser() if ar else None
    if align_path is not None:
        align_path = align_path.expanduser().resolve()
        if align_path.is_file():
            tol_align = args.align_reference_tolerance_inches
            if tol_align is None:
                tol_align = float(os.getenv("ALIGN_REFERENCE_TOLERANCE_INCHES", "6") or "6")
            print(
                f"--- Align takeoff JSON to Project-1 reference BOM ({tol_align:g}\") ---",
                file=sys.stderr,
            )
            try:
                ast = align_project1_reference_json(
                    final_json,
                    reference_xlsx=align_path,
                    tol_inches=tol_align,
                    fill_piecemarks=args.align_fill_piecemarks,
                )
                print(f"Project-1 reference align: {ast}", file=sys.stderr)
            except Exception as ex:
                print(f"WARNING: Project-1 reference align failed ({ex})", file=sys.stderr)
        else:
            print(f"WARNING: ALIGN_REFERENCE_PROJECT1_BOM not found: {align_path}", file=sys.stderr)

    auto_w = (os.getenv("STEEL_WEIGHT_ENRICH_AUTO", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if auto_w:
        wo_raw = (os.getenv("WEIGHT_OVERRIDE_CSV", "") or "").strip()
        wo_path = Path(wo_raw).expanduser() if wo_raw else None
        if wo_path is not None and not wo_path.is_file():
            print(f"WARNING: WEIGHT_OVERRIDE_CSV not found: {wo_path}", file=sys.stderr)
            wo_path = None
        try:
            from saddleback_pipeline.steel_weight_enrichment import enrich_takeoff_json_file

            ow = (os.getenv("OVERWRITE_STEEL_WEIGHT", "") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            st = enrich_takeoff_json_file(
                final_json,
                weight_override_csv=wo_path,
                overwrite_existing=ow,
            )
            print(f"Steel weight enrichment on final JSON: {st}", file=sys.stderr)
        except Exception as ex:
            print(f"WARNING: steel_weight_enrichment failed ({ex})", file=sys.stderr)

    print("--- Export Excel ---", file=sys.stderr)
    n_ent, n_bom, n_sc = export_takeoff_json_to_xlsx(final_json, xlsx_path)
    print(
        f"Wrote: {xlsx_path} (Takeoff sheet: {n_ent} rows; Material Summary: {n_bom} rows; "
        f"Drawing scales sheet: {n_sc} rows)",
        file=sys.stderr,
    )
    print(final_json)
    print(xlsx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
