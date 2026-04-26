"""
Benchmark multiple Google Gemini multimodal models against the reference Material Summary.

Cursor does not expose a vision API for your PDF pipeline; this uses the same Gemini API
as ``gemini_takeoff`` (state-of-the-art multimodal models from Google AI).

Usage (from project root, with ``.env`` containing GEMINI_API_KEY, INPUT_PDF, etc.)::

    python -m saddleback_pipeline.benchmark_models

    python -m saddleback_pipeline.benchmark_models --models gemini-2.5-flash,gemini-3.1-pro-preview

Outputs under ``benchmark_runs/<model_slug>/takeoff_output.json`` and ``benchmark_runs/benchmark_results.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from saddleback_pipeline.bom_accuracy import (
    compare_boms,
    compare_columns_by_material,
    load_generated_bom_from_json,
    load_generated_bom_rows,
    load_reference_bom,
    load_reference_rows,
    rollup_by_material,
)
from saddleback_pipeline.gemini_takeoff import run_takeoff

# Multimodal / vision-capable models (PDF + images); Gemini 3 + 2.5 families per Google AI docs.
DEFAULT_SOTA_MODELS: list[str] = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model.strip())


def composite_bom_score(col: dict, strict: dict) -> float:
    """Single 0–1 score for ranking (length-ignored column view + strict key F1)."""
    q = float(col.get("qty_match_rate_on_both") or 0.0)
    mr = float(col.get("material_recall") or 0.0)
    mp = float(col.get("material_precision") or 0.0)
    kf = float(strict.get("key_f1") or 0.0)
    return 0.45 * q + 0.25 * mr + 0.20 * mp + 0.10 * kf


def evaluate_output(json_path: Path, ref_path: Path) -> tuple[dict, dict, float]:
    ref_rows = load_reference_rows(ref_path)
    gen_rows = load_generated_bom_rows(json_path)
    ref_by_mat = rollup_by_material(ref_rows)
    gen_by_mat = rollup_by_material(gen_rows)
    col = compare_columns_by_material(ref_by_mat, gen_by_mat)
    ref_agg = load_reference_bom(ref_path)
    gen_agg = load_generated_bom_from_json(json_path)
    strict = compare_boms(ref_agg, gen_agg) if gen_agg else {}
    if not gen_agg:
        strict = {
            "key_f1": 0.0,
            "reference_line_items": len(ref_agg),
            "generated_line_items": 0,
            "keys_in_both": 0,
        }
    score = composite_bom_score(col, strict)
    return col, strict, score


def main() -> int:
    load_dotenv(".env", override=False)
    import os

    p = argparse.ArgumentParser(description="Benchmark Gemini vision models for takeoff accuracy.")
    p.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_SOTA_MODELS),
        help="Comma-separated Gemini model IDs (default: SOTA multimodal set).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_runs"),
        help="Directory for per-model JSON and benchmark_results.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print models and paths only; do not call the API.",
    )
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Rescore existing benchmark_runs/<slug>/takeoff_output.json files (no API calls).",
    )
    args = p.parse_args()

    raw_pdf = (os.getenv("INPUT_PDF", "") or "").strip()
    key = (os.getenv("GEMINI_API_KEY", "") or "").strip()
    prompt_path = Path(os.getenv("PROMPT_PATH", "prompts/structural_takeoff.txt") or "prompts/structural_takeoff.txt")
    schema_csv = (os.getenv("SCHEMA_CSV", "") or "").strip()
    schema_path = Path(schema_csv) if schema_csv else None
    ref_path = Path(
        (os.getenv("REFERENCE_BOM_XLSX", "") or "").strip()
        or "26-LQ-094_SADDLEBACK VILLAGE_Material Summary.xlsx"
    ).expanduser()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERROR: No models in --models", file=sys.stderr)
        return 1

    pdf = Path(raw_pdf).expanduser()
    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_pdf or not pdf.is_file():
        print("ERROR: Set INPUT_PDF in .env to a valid PDF path.", file=sys.stderr)
        return 1
    if not key:
        print("ERROR: Set GEMINI_API_KEY in .env", file=sys.stderr)
        return 1
    if not ref_path.is_file():
        print(f"ERROR: Reference BOM not found: {ref_path}", file=sys.stderr)
        return 1

    print(
        "Note: This benchmarks Google Gemini multimodal models via the Gemini API "
        "(same as the takeoff pipeline). Cursor IDE models are not used here.\n",
        file=sys.stderr,
    )
    print(f"Reference BOM: {ref_path}", file=sys.stderr)
    print(f"PDF: {pdf}", file=sys.stderr)
    print(f"Output dir: {out_dir}\n", file=sys.stderr)

    if args.evaluate_only:
        results: list[dict] = []
        for sub in sorted(out_dir.iterdir()):
            if not sub.is_dir():
                continue
            jp = sub / "takeoff_output.json"
            if not jp.is_file():
                continue
            model = sub.name.replace("_", ".")  # imperfect; prefer results file
            try:
                col, strict, score = evaluate_output(jp, ref_path)
            except Exception as e:
                results.append(
                    {
                        "model": sub.name,
                        "slug": sub.name,
                        "error": str(e),
                        "output_json": str(jp),
                    }
                )
                continue
            row = {
                "model": sub.name,
                "slug": sub.name,
                "error": None,
                "seconds": None,
                "composite_score": round(score, 4),
                "material_recall": round(col["material_recall"], 4),
                "material_precision": round(col["material_precision"], 4),
                "qty_match_rate_on_both": round(col["qty_match_rate_on_both"], 4),
                "grade_match_rate": round(col["grade_match_rate_on_both"], 4),
                "pcmk_match_rate": round(col["pcmk_match_rate_on_both"], 4),
                "key_f1_strict_mat_length": round(strict.get("key_f1", 0.0), 4),
                "total_qty_ratio_gen_over_ref": strict.get("total_qty_ratio_gen_over_ref"),
                "output_json": str(jp),
            }
            results.append(row)
            print(
                f"{sub.name}: composite={score:.4f} qty@mat={col['qty_match_rate_on_both']:.2f}",
                file=sys.stderr,
            )
        summary_path = out_dir / "benchmark_results.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {summary_path}", file=sys.stderr)
        ok = [r for r in results if not r.get("error")]
        if ok:
            best = max(ok, key=lambda r: r["composite_score"])
            print(f"Best: {best.get('model')} score={best['composite_score']}", file=sys.stderr)
        print("\n--- Rankings ---")
        for r in sorted(ok, key=lambda x: -x["composite_score"]):
            print(f"{r['composite_score']:.4f}  {r['model']}")
        return 0 if ok else 1

    print("Models: " + ", ".join(models), file=sys.stderr)

    if args.dry_run:
        for m in models:
            print(f"  would write: {out_dir / model_slug(m) / 'takeoff_output.json'}")
        return 0

    results: list[dict] = []
    for model in models:
        slug = model_slug(model)
        run_dir = out_dir / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "takeoff_output.json"
        t0 = time.perf_counter()
        err: str | None = None
        print(f"\n--- Model: {model} ---", file=sys.stderr)
        try:
            run_takeoff(
                pdf_path=pdf,
                output_json=out_json,
                gemini_api_key=key,
                model=model,
                prompt_path=prompt_path,
                schema_csv_path=schema_path,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            results.append(
                {
                    "model": model,
                    "slug": slug,
                    "error": err,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "output_json": str(out_json),
                }
            )
            continue

        elapsed = round(time.perf_counter() - t0, 1)
        try:
            col, strict, score = evaluate_output(out_json, ref_path)
        except Exception as e:
            err = f"evaluate: {type(e).__name__}: {e}"
            results.append(
                {
                    "model": model,
                    "slug": slug,
                    "error": err,
                    "seconds": elapsed,
                    "output_json": str(out_json),
                }
            )
            continue

        row = {
            "model": model,
            "slug": slug,
            "error": None,
            "seconds": elapsed,
            "composite_score": round(score, 4),
            "material_recall": round(col["material_recall"], 4),
            "material_precision": round(col["material_precision"], 4),
            "qty_match_rate_on_both": round(col["qty_match_rate_on_both"], 4),
            "grade_match_rate": round(col["grade_match_rate_on_both"], 4),
            "pcmk_match_rate": round(col["pcmk_match_rate_on_both"], 4),
            "key_f1_strict_mat_length": round(strict.get("key_f1", 0.0), 4),
            "total_qty_ratio_gen_over_ref": strict.get("total_qty_ratio_gen_over_ref"),
            "output_json": str(out_json),
            "strict_key_metrics": {
                k: strict[k]
                for k in (
                    "reference_line_items",
                    "generated_line_items",
                    "keys_in_both",
                    "key_recall_vs_reference",
                    "key_precision_vs_generated",
                )
                if k in strict
            },
        }
        results.append(row)
        print(
            f"  composite={score:.3f}  mat_recall={col['material_recall']:.2f}  "
            f"qty@mat={col['qty_match_rate_on_both']:.2f}  key_f1={strict.get('key_f1', 0):.2f}  "
            f"({elapsed}s)",
            file=sys.stderr,
        )

    out_json_summary = out_dir / "benchmark_results.json"
    out_json_summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = [r for r in results if not r.get("error")]
    if ok:
        best = max(ok, key=lambda r: r["composite_score"])
        print("\n=== Best composite score (higher = better) ===", file=sys.stderr)
        print(f"  {best['model']}  score={best['composite_score']}", file=sys.stderr)
        print(f"\nWrote: {out_json_summary}", file=sys.stderr)

    print("\n--- Rankings (composite) ---")
    for r in sorted(ok, key=lambda x: -x["composite_score"]):
        print(
            f"{r['composite_score']:.4f}  {r['model']}  "
            f"(qty@mat={r['qty_match_rate_on_both']:.2f}, mat_recall={r['material_recall']:.2f})"
        )
    for r in results:
        if r.get("error"):
            print(f"FAILED  {r['model']}: {r['error']}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
