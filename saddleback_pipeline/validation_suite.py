"""Run validation steps after each pipeline improvement: BOM metrics → geometry report.

Usage (from project root, with .env and reference xlsx)::

    python -m saddleback_pipeline.validation_suite
    python -m saddleback_pipeline.validation_suite --json takeoff_output_spatial.json

Does not call Gemini; uses existing takeoff JSON and INPUT_PDF.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from saddleback_pipeline.bom_relaxed import relaxed_key_match_metrics
from saddleback_pipeline.bom_accuracy import load_generated_bom_from_json, load_reference_bom
from saddleback_pipeline.geometry_fusion import build_fusion_report
from saddleback_pipeline.pdf_geometry import build_geometry_report


def _run_bom_subprocess(json_path: Path, relaxed: float | None) -> int:
    env = os.environ.copy()
    env["INPUT_JSON"] = str(json_path)
    cmd = [sys.executable, "-m", "saddleback_pipeline.bom_accuracy"]
    if relaxed is not None:
        cmd.extend(["--relaxed-inches", str(relaxed)])
    r = subprocess.run(cmd, cwd=Path.cwd(), env=env, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="", file=sys.stderr)
    return r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation suite: BOM + geometry metrics.")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Takeoff JSON (default: INPUT_JSON or takeoff_output.json).",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Only print BOM metrics.",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip geometry–takeoff fusion (needs detections_output.json).",
    )
    args = parser.parse_args()

    load_dotenv(".env", override=False)
    json_path = args.json or Path(os.getenv("INPUT_JSON", "takeoff_output.json")).expanduser()
    ref_path = Path(
        (os.getenv("REFERENCE_BOM_XLSX", "") or "").strip()
        or "26-LQ-094_SADDLEBACK VILLAGE_Material Summary.xlsx",
    ).expanduser()

    if not json_path.is_file():
        print(f"ERROR: JSON not found: {json_path}", file=sys.stderr)
        return 1
    if not ref_path.is_file():
        print(f"ERROR: Reference BOM not found: {ref_path}", file=sys.stderr)
        return 1

    report: dict = {"input_json": str(json_path), "reference_xlsx": str(ref_path), "steps": {}}

    print("\n######## STEP 1 — BOM vs reference (strict + relaxed) ########\n")
    for tol, label in [(None, "strict_keys_only"), (2.0, "relaxed_2in"), (6.0, "relaxed_6in")]:
        title = "strict (no relaxed block)" if tol is None else f"relaxed ±{tol:g}\""
        print(f"--- {title} ---\n")
        code = _run_bom_subprocess(json_path, tol)
        if code != 0:
            return code
        print()

    ref_agg = load_reference_bom(ref_path)
    gen_agg = load_generated_bom_from_json(json_path)
    if gen_agg:
        for tol in (2.0, 6.0, 12.0):
            rel = relaxed_key_match_metrics(
                set(ref_agg.keys()),
                set(gen_agg.keys()),
                length_tol_inches=tol,
            )
            report["steps"][f"relaxed_f1_{tol}in"] = rel["relaxed_key_f1"]

    if args.skip_geometry:
        out = Path(os.getenv("VALIDATION_REPORT_JSON", "validation_report.json")).expanduser()
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
        return 0

    raw_pdf = (os.getenv("INPUT_PDF", "") or "").strip()
    if not raw_pdf:
        print("SKIP geometry: INPUT_PDF not set.", file=sys.stderr)
        out = Path(os.getenv("VALIDATION_REPORT_JSON", "validation_report.json")).expanduser()
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
        return 0

    print("\n######## STEP 2 — Vector + raster geometry (per PDF page 0) ########\n")
    pdf = Path(raw_pdf).expanduser()
    if not pdf.is_file():
        print(f"SKIP geometry: PDF not found: {pdf}", file=sys.stderr)
    else:
        geom = build_geometry_report(pdf, page_index=0, include_raster=True)
        report["steps"]["geometry"] = {
            "vector_segments": geom["vector_line_segments"],
            "vector_joints_deg3plus": geom["vector_graph"]["nodes_degree_3plus_joints"],
            "raster_hough_segments": geom["raster_hough_segments"],
        }
        if geom.get("raster_graph"):
            report["steps"]["geometry"]["raster_joints_deg3plus"] = geom["raster_graph"][
                "nodes_degree_3plus_joints"
            ]
        print(json.dumps(report["steps"]["geometry"], indent=2))
        gout = Path(os.getenv("GEOMETRY_REPORT_JSON", "geometry_report.json")).expanduser()
        gout.write_text(json.dumps(geom, indent=2), encoding="utf-8")
        print(f"\nFull geometry report: {gout}")

    if not args.skip_geometry and not args.skip_fusion and raw_pdf:
        det_path = Path(
            (os.getenv("DETECTIONS_JSON", "") or "").strip() or "detections_output.json",
        ).expanduser()
        print(
            "\n######## STEP 2.5 — Geometry ↔ takeoff fusion (per elevation view) ########\n"
        )
        if not det_path.is_file():
            print(
                f"SKIP fusion: {det_path} not found (run takeoff with GEMINI_SPATIAL_DETECTION=true).",
                file=sys.stderr,
            )
        else:
            pdf = Path(raw_pdf).expanduser()
            fus = build_fusion_report(
                pdf_path=pdf,
                takeoff_json_path=json_path,
                detections_json_path=det_path,
                page_index=int(os.getenv("GEOMETRY_PAGE_INDEX", "0") or "0"),
            )
            report["steps"]["fusion"] = {
                "geometry_takeoff_alignment_mean": fus.get(
                    "overall_geometry_takeoff_alignment_mean"
                ),
                "detection_vs_takeoff_column_alignment_mean": fus.get(
                    "mean_detection_vs_takeoff_column_alignment"
                ),
                "drawing_scale_used": fus.get("drawing_scale_used"),
                "scale_calibrated_sample": (
                    fus.get("elevation_views", [{}])[0].get("scale_calibrated_geometry")
                    if fus.get("elevation_views")
                    else None
                ),
            }
            fout = Path(
                os.getenv("FUSION_REPORT_JSON", "fusion_report.json"),
            ).expanduser()
            fout.write_text(json.dumps(fus, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report["steps"]["fusion"], indent=2))
            print(f"\nFull fusion report: {fout}")
            r6 = report["steps"].get("relaxed_f1_6.0in")
            print(
                "\n--- Interpretation ---\n"
                f"Reference BOM relaxed key F1 @ 6\" (vs Excel): {r6!s}\n"
                "Geometry fusion does **not** change that score (takeoff JSON is unchanged). "
                "It adds QA signals: vector lines inside view boxes vs takeoff quantities, "
                "and detection column boxes vs takeoff column counts.\n"
            )

    print(
        "\n######## STEP 3 — Learned detector (YOLO / ONNX) ########\n"
        "Not run: train a detector on labeled crops, export ONNX, then add an inference "
        "hook next to gemini_spatial_detection. Placeholder for future accuracy gains.\n"
    )

    out = Path(os.getenv("VALIDATION_REPORT_JSON", "validation_report.json")).expanduser()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
