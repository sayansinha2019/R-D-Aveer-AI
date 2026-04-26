"""Integrated structural takeoff orchestration (multi-PDF, merge, reconcile, export, QA).

This **does not** remove or replace the geometric stack (``pdf_geometry``, ``geometry_fusion``,
``drawing_scales``): those modules add **measurable QA** (vector stats, fusion vs view boxes)
on top of Gemini extraction. They are complementary, not obsolete.

Flow (typical)::

    PDFs → Gemini takeoff (per file) → merge JSON → optional reconcile vs Material Summary
    → optional align vs reference Project-1 BOM (shop truth for weight/grade when keys match)
    → optional steel weight enrichment (CSV overrides, then AISC nominal lb/ft × length)
    → optional Project-1 BOM xlsx → optional validation_suite (BOM + geometry + fusion)

Commercial vs open source (accuracy)
-------------------------------------
* **Gemini / other VLMs** — best general “read the sheet” reasoning; keep as core.
* **Google Document AI** — already supported in ``pdf_text.py`` for OCR-quality text paths.
* **IfcOpenShell** — open source IFC read; use when the **model** is the contract
  (see ``model_bridge.py``). Install optionally; not pinned in default requirements.
* **Autodesk APS, Tekla/Trimble APIs, Speckle** — commercial / hosted options when the
  customer authorizes model access; integrate behind ``model_bridge``-style adapters.

Usage::

    python -m saddleback_pipeline.integrated_pipeline --pdfs a.pdf b.pdf --out-dir out/ \\
        --merged-json out/merged.json --project1-bom-xlsx out/merged_bom.xlsx \\
        --reference-material-xlsx \"path/to/Material Summary.xlsx\" --run-validation

Environment: same as ``gemini_takeoff`` (``GEMINI_API_KEY``, ``GEMINI_MODEL``, ``PDF_AS_IMAGES``, …).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from saddleback_pipeline.foundation_sheet_takeoff import run_foundation_sheet_takeoff
from saddleback_pipeline.gemini_takeoff import run_takeoff
from saddleback_pipeline.material_summary_reconcile import run_reconcile_file
from saddleback_pipeline.model_bridge import (
    ifcopenshell_available,
    load_steel_elements_from_ifc,
    summarize_model_vs_takeoff_stub,
)
from saddleback_pipeline.piecemark_resolve import apply_piecemark_map, load_label_to_piecemark_csv
from saddleback_pipeline.project1_bom_export import export_merged_takeoff_jsons_to_project1_bom_xlsx
from saddleback_pipeline.project1_reference_align import align_takeoff_json_file as align_project1_reference_json
from saddleback_pipeline.steel_weight_enrichment import enrich_takeoff_json_file
from saddleback_pipeline.takeoff_merge import load_takeoff_jsons, merge_takeoff_payloads, write_takeoff_json


def _run_takeoffs(
    pdfs: list[Path],
    out_dir: Path,
    *,
    foundation_stems: set[str],
    prompt_path: Path,
    schema_csv_path: Path | None,
    gemini_api_key: str,
    model: str,
) -> list[Path]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_paths: list[Path] = []
    for pdf in pdfs:
        pdf = pdf.expanduser().resolve()
        out_json = out_dir / f"{pdf.stem}_takeoff.json"
        use_foundation = pdf.stem in foundation_stems
        mode = "foundation supplement" if use_foundation else "standard"
        print(f"--- Takeoff ({mode}): {pdf.name} → {out_json.name} ---", file=sys.stderr)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-PDF takeoff → merged JSON → optional reconcile + Project-1 BOM + validation.",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        type=Path,
        default=[],
        help="Structural PDFs to run through Gemini (in order).",
    )
    parser.add_argument(
        "--from-json",
        nargs="*",
        type=Path,
        default=[],
        help="Existing takeoff JSON files to merge (skip Gemini when used alone).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("integrated_output"),
        help="Directory for per-PDF takeoff JSON when using --pdfs.",
    )
    parser.add_argument(
        "--merged-json",
        type=Path,
        required=True,
        help="Output path for merged takeoff JSON.",
    )
    parser.add_argument(
        "--project1-bom-xlsx",
        type=Path,
        default=None,
        help="If set, write merged entities to Project-1 style Bill of Materials xlsx.",
    )
    parser.add_argument(
        "--reference-material-xlsx",
        type=Path,
        default=None,
        help="Optional 6-column Material Summary xlsx for material_summary reconcile.",
    )
    parser.add_argument(
        "--reconcile-tolerance-inches",
        type=float,
        default=float(os.getenv("RECONCILE_TOLERANCE_INCHES", "12") or "12"),
    )
    parser.add_argument(
        "--reconciled-json",
        type=Path,
        default=None,
        help="If set with --reference-material-xlsx, write reconciled JSON to this path "
        "(default: merged path with _reconciled suffix).",
    )
    parser.add_argument(
        "--dedupe-entities",
        action="store_true",
        help="Dedupe merged data rows by (piece_mark|entity_id, section, length, element_type).",
    )
    parser.add_argument(
        "--piecemark-csv",
        type=Path,
        default=None,
        help="Optional CSV: drawing_label → piece_mark (see piecemark_resolve.py).",
    )
    parser.add_argument(
        "--ifc",
        type=Path,
        default=None,
        help="Optional IFC path; emits model_vs_takeoff summary JSON next to merged-json.",
    )
    parser.add_argument(
        "--run-validation",
        action="store_true",
        help="Run validation_suite on final JSON (sets INPUT_JSON; uses first PDF as INPUT_PDF).",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="With --pdfs, skip Gemini (only merge --from-json).",
    )
    parser.add_argument(
        "--foundation-sheet-stems",
        nargs="*",
        default=[],
        metavar="STEM",
        help="PDF stem(s) that need foundation/civil supplement (e.g. S111). "
        "Uses foundation_sheet_takeoff instead of standard steel prompt.",
    )
    parser.add_argument(
        "--skip-steel-weight-enrich",
        action="store_true",
        help="Skip AISC nominal lb/ft × length weight fill (see steel_weight_enrichment.py).",
    )
    parser.add_argument(
        "--weight-override-csv",
        type=Path,
        default=None,
        help="Optional CSV: section_key,length_inches,weight_lb (per piece). "
        "Also reads env WEIGHT_OVERRIDE_CSV if unset.",
    )
    parser.add_argument(
        "--align-reference-project1-bom",
        type=Path,
        default=None,
        help="Optional Project-1 style BOM xlsx: fill weight/grade from matching rows "
        "(see project1_reference_align). Env ALIGN_REFERENCE_PROJECT1_BOM if unset.",
    )
    parser.add_argument(
        "--align-reference-tolerance-inches",
        type=float,
        default=float(os.getenv("ALIGN_REFERENCE_TOLERANCE_INCHES", "6") or "6"),
        help="Length tolerance when matching takeoff entities to reference BOM rows.",
    )
    parser.add_argument(
        "--align-fill-piecemarks",
        action="store_true",
        help="When aligning to reference BOM, copy piecemarks into empty piece_mark fields.",
    )
    parser.add_argument(
        "--overwrite-steel-weight",
        action="store_true",
        help="Pass through to steel_weight_enrichment: replace non-null weights too (CSV + nominal).",
    )
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    model = (os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview").strip()
    prompt_path = Path(
        os.getenv("PROMPT_PATH", "prompts/structural_takeoff.txt") or "prompts/structural_takeoff.txt",
    )
    schema_csv = (os.getenv("SCHEMA_CSV", "") or "").strip()
    schema_path = Path(schema_csv) if schema_csv else None

    json_paths: list[Path] = list(args.from_json)
    if args.pdfs and not args.skip_gemini:
        if not key:
            print("ERROR: GEMINI_API_KEY required for --pdfs takeoff.", file=sys.stderr)
            return 1
        new_paths = _run_takeoffs(
            list(args.pdfs),
            args.out_dir,
            foundation_stems=set(args.foundation_sheet_stems or []),
            prompt_path=prompt_path,
            schema_csv_path=schema_path if schema_path and schema_path.is_file() else None,
            gemini_api_key=key,
            model=model,
        )
        # Prepend so explicit --from-json can add extra hand-edited layers after PDFs
        json_paths = new_paths + json_paths

    if not json_paths:
        print("ERROR: Provide --pdfs (with Gemini) and/or --from-json with JSON paths.", file=sys.stderr)
        return 1

    payloads = load_takeoff_jsons(json_paths)
    merged = merge_takeoff_payloads(payloads, dedupe_entities=args.dedupe_entities)

    if args.piecemark_csv and args.piecemark_csv.is_file():
        pmap = load_label_to_piecemark_csv(args.piecemark_csv)
        merged["data"] = apply_piecemark_map(merged.get("data") or [], pmap)

    merged_path = args.merged_json.expanduser().resolve()
    write_takeoff_json(merged, merged_path)
    print(f"Wrote merged: {merged_path} ({len(merged.get('data') or [])} data rows)", file=sys.stderr)

    final_json_path = merged_path
    if args.reference_material_xlsx and args.reference_material_xlsx.is_file():
        rec_out = args.reconciled_json
        if rec_out is None:
            rec_out = merged_path.with_name(merged_path.stem + "_reconciled.json")
        stats = run_reconcile_file(
            input_json=merged_path,
            output_json=rec_out,
            reference_xlsx=args.reference_material_xlsx.expanduser().resolve(),
            tol_inches=args.reconcile_tolerance_inches,
        )
        print(f"Reconciled material_summary → {rec_out}", file=sys.stderr)
        print(json.dumps(stats, indent=2), file=sys.stderr)
        final_json_path = rec_out

    align_ref = args.align_reference_project1_bom
    if align_ref is None:
        ar = (os.getenv("ALIGN_REFERENCE_PROJECT1_BOM", "") or "").strip()
        align_ref = Path(ar).expanduser() if ar else None
    if align_ref is not None:
        align_ref = align_ref.expanduser().resolve()
        if align_ref.is_file():
            try:
                ast = align_project1_reference_json(
                    final_json_path,
                    reference_xlsx=align_ref,
                    tol_inches=args.align_reference_tolerance_inches,
                    fill_piecemarks=args.align_fill_piecemarks,
                )
                print(f"Project-1 reference align: {ast}", file=sys.stderr)
            except Exception as ex:
                print(f"WARNING: Project-1 reference align failed ({ex})", file=sys.stderr)
        else:
            print(f"WARNING: reference BOM not found for align: {align_ref}", file=sys.stderr)

    if not args.skip_steel_weight_enrich:
        wo = args.weight_override_csv
        if wo is None:
            wr = (os.getenv("WEIGHT_OVERRIDE_CSV", "") or "").strip()
            wo = Path(wr).expanduser() if wr else None
        if wo is not None and not wo.is_file():
            print(f"WARNING: weight override CSV not found: {wo}", file=sys.stderr)
            wo = None
        try:
            st = enrich_takeoff_json_file(
                final_json_path,
                weight_override_csv=wo,
                overwrite_existing=args.overwrite_steel_weight,
            )
            print(f"Steel weight enrichment: {st}", file=sys.stderr)
        except Exception as ex:
            print(f"WARNING: steel weight enrichment failed ({ex})", file=sys.stderr)

    if args.project1_bom_xlsx:
        n = export_merged_takeoff_jsons_to_project1_bom_xlsx(
            [final_json_path],
            args.project1_bom_xlsx,
            template_xlsx=None,
        )
        print(f"Wrote Project-1 BOM xlsx ({n} rows): {args.project1_bom_xlsx}", file=sys.stderr)

    if args.ifc:
        if not ifcopenshell_available():
            print(
                "WARNING: IfcOpenShell not installed; skip IFC load. pip install ifcopenshell",
                file=sys.stderr,
            )
        else:
            rows = load_steel_elements_from_ifc(args.ifc.expanduser().resolve())
            final_payload = json.loads(final_json_path.read_text(encoding="utf-8"))
            summary = summarize_model_vs_takeoff_stub(rows, final_payload)
            sidecar = final_json_path.with_name(final_json_path.stem + "_ifc_summary.json")
            sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Wrote IFC summary: {sidecar}", file=sys.stderr)

    if args.run_validation:
        env = os.environ.copy()
        env["INPUT_JSON"] = str(final_json_path)
        first_pdf = None
        if args.pdfs:
            first_pdf = str(args.pdfs[0].expanduser().resolve())
        if first_pdf:
            env["INPUT_PDF"] = first_pdf
        r = subprocess.run(
            [sys.executable, "-m", "saddleback_pipeline.validation_suite", "--json", str(final_json_path)],
            cwd=Path.cwd(),
            env=env,
        )
        if r.returncode != 0:
            return r.returncode

    print(str(final_json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
