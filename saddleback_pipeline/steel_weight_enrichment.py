"""Post-process takeoff JSON to fill **missing** steel member weights.

Why this exists
---------------
Shop BOM accuracy needs **lbs**. Drawings often omit weights; Gemini may leave ``weight`` null.

Industry-standard shortcut (AISC naming): for **W / M / S / HP / MC** shapes the designation is
``Prefix<NominalDepth>x<WeightPerFoot>`` — the **second number is lb/ft** (see AISC “Sizes &
Grades”). Piece weight ≈ ``lb_per_ft × cut_length_ft``. This matches calculator/BOM CSV lines
keyed by section + length when those tools use the same convention.

Limits
------
* Nominal lb/ft × length is **not** camber/cope/fireproofing adjusted — same limitation as manual
  quick estimates.
* **HSS / angles / plates** need tables or CSV overrides (not inferred from simple designation alone).

CSV overrides (optional)
------------------------
UTF-8 CSV with header::

    section_key,length_inches,weight_lb

``section_key`` = uppercased normalized section like ``W24X62`` (quotes stripped).
``weight_lb`` = total weight for **one piece** at that cut length (matches expanded ``data`` rows).
For rolled-up ``material_summary`` lines, weight is scaled by ``qty`` when filling from nominal.

Also supported: ``length_ft`` column instead of ``length_inches`` (one or the other required).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from saddleback_pipeline.bom_relaxed import material_match_key, parse_length_to_inches


# AISC-style rolled shapes where the trailing number is weight per foot (lb/ft).
_RE_WM_SHP = re.compile(r"^[WMS]\s*(\d+)\s*[Xx]\s*([\d.]+)\s*$")
_RE_HP = re.compile(r"^HP\s*(\d+)\s*[Xx]\s*([\d.]+)\s*$", re.I)
_RE_MC = re.compile(r"^MC\s*(\d+)\s*[Xx]\s*([\d.]+)\s*$", re.I)


def normalize_steel_section_callout(section: Any) -> str:
    if section is None:
        return ""
    s = str(section).strip()
    # Plan labels like ``W21X48 (28)`` — trailing parenthetical is quantity hint, not lb/ft.
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = s.upper().replace(" ", "").replace("×", "X")
    return re.sub(r'["\']', "", s)


def nominal_lb_per_foot(section: Any) -> float | None:
    """Return AISC nominal lb/ft encoded in designation, or None."""
    s = normalize_steel_section_callout(section)
    if not s:
        return None
    for rx in (_RE_WM_SHP, _RE_HP, _RE_MC):
        m = rx.match(s)
        if m:
            try:
                return float(m.group(2))
            except ValueError:
                return None
    return None


def length_ft_from_field(length_val: Any) -> float | None:
    inches = parse_length_to_inches(length_val)
    if inches is None:
        return None
    return inches / 12.0


def estimate_piece_weight_lb(
    *,
    section: Any,
    length_val: Any,
    quantity: int = 1,
) -> float | None:
    """Estimate total weight for ``quantity`` identical pieces (usually qty=1 in expanded data)."""
    lbft = nominal_lb_per_foot(section)
    if lbft is None:
        return None
    lf = length_ft_from_field(length_val)
    if lf is None or lf <= 0:
        return None
    return round(lbft * lf * max(1, quantity), 2)


def load_weight_overrides_csv(path: Path) -> dict[tuple[str, float], float]:
    """Map (material_match_key(section), length_inches) → weight_lb per piece."""
    path = path.expanduser().resolve()
    out: dict[tuple[str, float], float] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sec = (row.get("section_key") or row.get("section") or "").strip()
            if not sec:
                continue
            mk = material_match_key(sec)
            li = row.get("length_inches") or row.get("length_in")
            lf = row.get("length_ft")
            inches: float | None = None
            if li not in (None, ""):
                try:
                    inches = float(li)
                except ValueError:
                    continue
            elif lf not in (None, ""):
                try:
                    inches = float(lf) * 12.0
                except ValueError:
                    continue
            else:
                continue
            wcol = row.get("weight_lb") or row.get("weight") or row.get("lbs")
            if wcol in (None, ""):
                continue
            try:
                wlb = float(str(wcol).replace(",", ""))
            except ValueError:
                continue
            key = (mk, round(inches, 3))
            out[key] = wlb
    return out


def _lookup_override(
    overrides: dict[tuple[str, float], float],
    section: Any,
    length_val: Any,
    *,
    tol_in: float = 1.0,
) -> float | None:
    mk = material_match_key(str(section or ""))
    li = parse_length_to_inches(length_val)
    if li is None:
        return None
    if (mk, round(li, 3)) in overrides:
        return overrides[(mk, round(li, 3))]
    best: tuple[float, float] | None = None
    for (km, kin), w in overrides.items():
        if km != mk:
            continue
        d = abs(kin - li)
        if d <= tol_in and (best is None or d < best[0]):
            best = (d, w)
    return best[1] if best else None


def enrich_takeoff_payload(
    payload: dict[str, Any],
    *,
    weight_override_csv: Path | None = None,
    overwrite_existing: bool = False,
    apply_csv_overrides: bool = True,
    apply_nominal_lbft: bool = True,
) -> dict[str, Any]:
    """Mutate ``payload`` in place; add ``meta.steel_weight_enrichment`` stats."""
    overrides = (
        load_weight_overrides_csv(weight_override_csv)
        if (weight_override_csv and apply_csv_overrides)
        else {}
    )
    stats = {
        "from_override_csv": 0,
        "from_nominal_lbft": 0,
        "skipped_existing_weight": 0,
        "entities_considered": 0,
        "material_summary_rows_filled": 0,
    }

    data = payload.get("data")
    if isinstance(data, list):
        for ent in data:
            if not isinstance(ent, dict):
                continue
            stats["entities_considered"] += 1
            w0 = ent.get("weight")
            if w0 is not None and not overwrite_existing:
                try:
                    float(w0)
                    stats["skipped_existing_weight"] += 1
                    continue
                except (TypeError, ValueError):
                    pass

            ov = None
            if overrides:
                ov = _lookup_override(overrides, ent.get("section"), ent.get("length"))
            if overrides and ov is not None:
                q = 1
                try:
                    q = int(ent.get("quantity") or 1)
                except (TypeError, ValueError):
                    q = 1
                ent["weight"] = round(ov * q, 2)
                ent["weight_source"] = "override_csv"
                stats["from_override_csv"] += 1
                continue

            if not apply_nominal_lbft:
                continue

            est = estimate_piece_weight_lb(
                section=ent.get("section"),
                length_val=ent.get("length"),
                quantity=int(ent.get("quantity") or 1),
            )
            if est is not None:
                ent["weight"] = est
                ent["weight_source"] = "aisc_nominal_lbft_x_length"
                stats["from_nominal_lbft"] += 1

    bom = payload.get("material_summary")
    if isinstance(bom, list):
        for row in bom:
            if not isinstance(row, dict):
                continue
            if row.get("weight") is not None and not overwrite_existing:
                continue
            qty = 1
            try:
                qty = int(row.get("qty") or 1)
            except (TypeError, ValueError):
                qty = 1
            ov = None
            if overrides:
                ov = _lookup_override(overrides, row.get("material"), row.get("length"))
            if overrides and ov is not None:
                row["weight"] = round(ov * qty, 2)
                row["weight_source"] = "override_csv"
                stats["material_summary_rows_filled"] += 1
                continue
            if not apply_nominal_lbft:
                continue
            piece = estimate_piece_weight_lb(
                section=row.get("material"),
                length_val=row.get("length"),
                quantity=1,
            )
            if piece is not None:
                row["weight"] = round(piece * qty, 2)
                row["weight_source"] = "aisc_nominal_lbft_x_length"
                stats["material_summary_rows_filled"] += 1

    payload.setdefault("meta", {})
    if isinstance(payload["meta"], dict):
        payload["meta"]["steel_weight_enrichment"] = {
            **stats,
            "notes": (
                "W/M/S/HP/MC: weight ≈ (lb/ft from designation) × (length in ft). "
                "Overrides CSV wins when matched. Does not replace explicit drawing weights unless "
                "overwrite_existing=True."
            ),
        }
    return payload


def enrich_takeoff_json_file(
    path: Path,
    *,
    weight_override_csv: Path | None = None,
    overwrite_existing: bool = False,
    apply_csv_overrides: bool = True,
    apply_nominal_lbft: bool = True,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    enrich_takeoff_payload(
        payload,
        weight_override_csv=weight_override_csv,
        overwrite_existing=overwrite_existing,
        apply_csv_overrides=apply_csv_overrides,
        apply_nominal_lbft=apply_nominal_lbft,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return meta.get("steel_weight_enrichment") or {}


def main() -> int:
    p = argparse.ArgumentParser(description="Fill steel weights from AISC nominal lb/ft × length.")
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--weight-override-csv", type=Path, default=None)
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace non-null weights (default: only fill null).",
    )
    args = p.parse_args()
    st = enrich_takeoff_json_file(
        args.json,
        weight_override_csv=args.weight_override_csv,
        overwrite_existing=args.overwrite_existing,
    )
    print(json.dumps(st, indent=2))
    print(f"Updated: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
