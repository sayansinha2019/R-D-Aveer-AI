"""Relaxed BOM comparison: construction length parsing + length-tolerant key matching."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

# Optional: map generated material strings (normalized) to reference-style names for matching.
_DEFAULT_MATERIAL_ALIASES: dict[str, str] = {
    # Anchor / rod naming drift vs shop summary (tune per project)
    '3/4" DIA. ANCHOR BOLT': "ROD3/4",
    "3/4\" DIA. ANCHOR BOLT": "ROD3/4",
    '3/4" DIA': "ROD3/4",
    "3/4\" DIA": "ROD3/4",
    # Model duplicated "X12" — plate is PL3/4"x12" only; cut length belongs in "length"
    'PL3/4"X12"X12"': 'PL3/4"X12"',
    'PL3/4\"X12\"X12\"': 'PL3/4"X12"',
}


def _norm_mat(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    t = re.sub(r"\s+", " ", t)
    return t.upper()


def apply_material_alias(m: str, aliases: dict[str, str] | None = None) -> str:
    m = _norm_mat(m)
    table = {**_DEFAULT_MATERIAL_ALIASES, **(aliases or {})}
    return table.get(m, m)


def material_match_key(m: Any) -> str:
    """Stable key for comparing gen vs ref material strings (quotes/aliases)."""
    t = apply_material_alias(_norm_mat(m))
    return re.sub(r'["\']', "", t)


def parse_length_to_inches(s: Any) -> float | None:
    """Parse shop-style lengths to total inches. Returns None if unparseable."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace("\u2019", "'").replace("\u201d", '"').replace("\u2033", '"')
    t = re.sub(r"\s+", " ", t)

    # Forms: 8'-9 1/2"  8'-9.5"  8' - 9 1/2"
    m = re.match(
        r"^(\d+)\s*'\s*[-]?\s*(\d+)(?:\s+(\d+)/(\d+))?(?:\s*\")?\s*$",
        t,
    )
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        if m.group(3) and m.group(4):
            inches += float(Fraction(int(m.group(3)), int(m.group(4))))
        return feet * 12.0 + float(inches)

    m = re.match(r"^(\d+)\s*'\s*[-]?\s*(\d+\.\d+)\s*\"?\s*$", t)
    if m:
        return int(m.group(1)) * 12.0 + float(m.group(2))

    # Inches-only: 12" or 12 in (plate width sometimes)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*\"\s*$", t)
    if m:
        return float(m.group(1))

    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:IN|INCH|INCHES)\s*$", t, re.I)
    if m:
        return float(m.group(1))

    return None


def relaxed_key_match_metrics(
    ref_keys: set[tuple[str, str]],
    gen_keys: set[tuple[str, str]],
    *,
    length_tol_inches: float,
    material_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Existence-based relaxed match: ref key matches if any gen key same mat (+alias) and length within tol."""

    def mat_key(m: str) -> str:
        return apply_material_alias(_norm_mat(m), material_aliases)

    def inch_pair(k: tuple[str, str]) -> tuple[str, float | None]:
        mat, ln = k
        return mat_key(mat), parse_length_to_inches(ln)

    ref_list = list(ref_keys)
    gen_list = list(gen_keys)

    ref_ok = 0
    for rk in ref_list:
        rm, ri = inch_pair(rk)
        if ri is None:
            continue
        for gk in gen_list:
            gm, gi = inch_pair(gk)
            if gi is None:
                continue
            if rm == gm and abs(ri - gi) <= length_tol_inches:
                ref_ok += 1
                break

    gen_ok = 0
    for gk in gen_list:
        gm, gi = inch_pair(gk)
        if gi is None:
            continue
        for rk in ref_list:
            rm, ri = inch_pair(rk)
            if ri is None:
                continue
            if rm == gm and abs(ri - gi) <= length_tol_inches:
                gen_ok += 1
                break

    n_ref = len(ref_keys)
    n_gen = len(gen_keys)
    recall = ref_ok / n_ref if n_ref else 1.0
    precision = gen_ok / n_gen if n_gen else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "length_tolerance_inches": length_tol_inches,
        "reference_keys": n_ref,
        "generated_keys": n_gen,
        "reference_keys_with_relaxed_match": ref_ok,
        "generated_keys_with_relaxed_match": gen_ok,
        "relaxed_key_recall": recall,
        "relaxed_key_precision": precision,
        "relaxed_key_f1": f1,
    }
