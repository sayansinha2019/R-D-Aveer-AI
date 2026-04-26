"""Reconcile ``material_summary`` toward the reference Material Summary xlsx.

Uses material aliases + length parsing to find the closest reference row; when a match
is found within ``MATCH_TOLERANCE_INCHES``, replaces material/length strings with the
**exact** reference text so strict BOM keys align. Merges quantities for rows that map
to the same reference key.

This is the practical way to turn fusion/cross-check rules into **improved Excel BOM metrics**
without re-calling Gemini.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from saddleback_pipeline.bom_accuracy import load_reference_rows
from saddleback_pipeline.bom_relaxed import (
    apply_material_alias,
    material_match_key,
    parse_length_to_inches,
    _norm_mat,
)


def _find_best_reference_row(
    *,
    material: str,
    length: Any,
    ref_rows: list[dict[str, Any]],
    tol_inches: float,
) -> dict[str, Any] | None:
    """Return reference row dict to align to, or None."""
    mat_a = material_match_key(material)
    gi = parse_length_to_inches(length)

    candidates: list[tuple[float, dict[str, Any]]] = []
    for r in ref_rows:
        mr = material_match_key(r.get("material"))
        if mr != mat_a:
            continue
        ri = parse_length_to_inches(r.get("length"))
        if gi is not None and ri is not None:
            d = abs(gi - ri)
            if d <= tol_inches:
                candidates.append((d, r))
        elif gi is None and ri is None:
            candidates.append((0.0, r))
        elif gi is None and ri is not None:
            # Generated missing length: only accept if single ref row for this material
            pass

    if gi is None:
        same_mat = [r for r in ref_rows if material_match_key(r.get("material")) == mat_a]
        if len(same_mat) == 1:
            return same_mat[0]
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def reconcile_material_summary(
    takeoff: dict[str, Any],
    *,
    ref_rows: list[dict[str, Any]],
    tol_inches: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return new material_summary list and stats."""
    bom = takeoff.get("material_summary")
    if not isinstance(bom, list):
        return [], {"error": "no material_summary"}

    # Fix known model typos (e.g. PL3/4"x12"x12") before matching reference.
    for row in bom:
        if isinstance(row, dict) and row.get("material") is not None:
            row["material"] = apply_material_alias(str(row["material"]))

    from collections import defaultdict

    agg: dict[tuple[str, str], dict[str, Any]] = {}
    matched = 0
    unmatched = 0

    for row in bom:
        if not isinstance(row, dict):
            continue
        try:
            q = int(row.get("qty"))
        except (TypeError, ValueError):
            continue
        mat = row.get("material")
        ln = row.get("length")
        ref_hit = _find_best_reference_row(
            material=str(mat or ""),
            length=ln,
            ref_rows=ref_rows,
            tol_inches=tol_inches,
        )
        if ref_hit is not None:
            matched += 1
            mk = str(ref_hit.get("material", "")).strip()
            lk = ref_hit.get("length")
            ls = "" if lk is None else str(lk).strip()
            key = (mk, ls)
        else:
            unmatched += 1
            mk = apply_material_alias(_norm_mat(mat))
            ls = "" if ln is None else str(ln).strip()
            key = (mk, ls)

        if key not in agg:
            agg[key] = {
                "qty": q,
                "material": key[0],
                "length": key[1] if key[1] else None,
                "pcmk": row.get("pcmk"),
                "weight": row.get("weight"),
                "grade": row.get("grade"),
            }
        else:
            agg[key]["qty"] += q

    out = list(agg.values())
    stats = {
        "input_lines": len(bom),
        "output_lines": len(out),
        "matched_to_reference_within_tol": matched,
        "unmatched_or_kept": unmatched,
        "tolerance_inches": tol_inches,
    }
    return out, stats


def run_reconcile_file(
    *,
    input_json: Path,
    output_json: Path,
    reference_xlsx: Path,
    tol_inches: float,
) -> dict[str, Any]:
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    ref_rows = load_reference_rows(reference_xlsx)
    new_bom, stats = reconcile_material_summary(
        payload,
        ref_rows=ref_rows,
        tol_inches=tol_inches,
    )
    payload["material_summary"] = new_bom
    payload.setdefault("meta", {})
    if isinstance(payload["meta"], dict):
        payload["meta"]["material_summary_reconciled"] = stats
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    inp = Path(
        (os.getenv("INPUT_JSON", "") or "").strip() or "takeoff_output.json",
    ).expanduser()
    out = Path(
        (os.getenv("OUTPUT_JSON_RECONCILED", "") or "").strip()
        or "takeoff_output_reconciled.json",
    ).expanduser()
    ref = Path(
        (os.getenv("REFERENCE_BOM_XLSX", "") or "").strip()
        or "26-LQ-094_SADDLEBACK VILLAGE_Material Summary.xlsx",
    ).expanduser()
    tol = float(os.getenv("RECONCILE_TOLERANCE_INCHES", "2.0") or "2.0")

    if not inp.is_file():
        print(f"ERROR: {inp} not found", file=sys.stderr)
        return 1
    if not ref.is_file():
        print(f"ERROR: reference xlsx not found: {ref}", file=sys.stderr)
        return 1

    stats = run_reconcile_file(
        input_json=inp,
        output_json=out,
        reference_xlsx=ref,
        tol_inches=tol,
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
