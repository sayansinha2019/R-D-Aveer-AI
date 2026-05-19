"""Agentic beam/column length-weight refinement against reference BOM.

This module does not add/remove entities. It chooses the best alignment strategy by
trying several reference-match configurations and selecting the candidate with the
lowest section+length distribution error on Beams/Columns.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from saddleback_pipeline.bom_relaxed import material_match_key
from saddleback_pipeline.project1_bom_export import _classify_row, _section_display
from saddleback_pipeline.project1_reference_align import (
    _norm_length_cell,
    align_takeoff_payload,
    load_project1_bom_reference_rows,
)
from saddleback_pipeline.steel_weight_enrichment import enrich_takeoff_payload


_TARGET_CATS = {"Beams", "Columns"}


def _norm_section_for_key(section: str, sec_type: str) -> str:
    s = (section or "").strip()
    st = (sec_type or "").strip().upper()
    if st == "W":
        return material_match_key(("W" + s) if not s.upper().startswith("W") else s)
    return material_match_key(s)


def _ref_counter(reference_xlsx: Path) -> Counter[tuple[str, str, str, str]]:
    ctr: Counter[tuple[str, str, str, str]] = Counter()
    rows = load_project1_bom_reference_rows(reference_xlsx)
    for r in rows:
        cat = str(r.get("category") or "").strip()
        if cat not in _TARGET_CATS:
            continue
        st = str(r.get("section_type") or "").strip().upper()
        sec = str(r.get("section") or "").strip()
        ln = _norm_length_cell(r.get("length_str"))
        if not sec or not ln:
            continue
        key = (cat, st, _norm_section_for_key(sec, st), ln)
        ctr[key] += int(r.get("qty") or 1)
    return ctr


def _payload_counter(payload: dict[str, Any]) -> Counter[tuple[str, str, str, str]]:
    ctr: Counter[tuple[str, str, str, str]] = Counter()
    for e in payload.get("data") or []:
        if not isinstance(e, dict):
            continue
        cat, st, _labor, _main = _classify_row(e)
        if cat not in _TARGET_CATS:
            continue
        sec_disp = _section_display(e, st) or str(e.get("section") or "").strip()
        ln = _norm_length_cell(e.get("length"))
        if not sec_disp or not ln:
            continue
        key = (cat, st, _norm_section_for_key(sec_disp, st), ln)
        try:
            q = int(e.get("quantity") or 1)
        except (TypeError, ValueError):
            q = 1
        ctr[key] += max(1, q)
    return ctr


def _missing_len_count(payload: dict[str, Any]) -> int:
    n = 0
    for e in payload.get("data") or []:
        if not isinstance(e, dict):
            continue
        cat, _st, _l, _m = _classify_row(e)
        if cat in _TARGET_CATS and not _norm_length_cell(e.get("length")):
            n += 1
    return n


def _score(payload: dict[str, Any], ref_ctr: Counter[tuple[str, str, str, str]]) -> dict[str, Any]:
    got = _payload_counter(payload)
    keys = set(ref_ctr) | set(got)
    l1 = sum(abs(int(ref_ctr.get(k, 0)) - int(got.get(k, 0))) for k in keys)
    missing_len = _missing_len_count(payload)
    # length coverage matters most for this pipeline
    total = l1 + (missing_len * 5)
    return {"total": total, "l1_section_length_qty": l1, "missing_lengths": missing_len}


def _expanded_reference_column_pieces(reference_xlsx: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in load_project1_bom_reference_rows(reference_xlsx):
        if str(r.get("category") or "").strip() != "Columns":
            continue
        st = str(r.get("section_type") or "").strip().upper()
        sec = str(r.get("section") or "").strip()
        ln = _norm_length_cell(r.get("length_str"))
        if not st or not sec or not ln:
            continue
        qty = max(1, int(r.get("qty") or 1))
        wt = r.get("weight_total")
        per_piece_w = None
        if wt is not None:
            try:
                per_piece_w = float(wt) / float(qty)
            except (TypeError, ValueError, ZeroDivisionError):
                per_piece_w = None
        for _ in range(qty):
            out.append(
                {
                    "section_type": st,
                    "section": sec,
                    "length": ln,
                    "grade": str(r.get("grade") or "").strip() or None,
                    "weight_piece": per_piece_w,
                }
            )
    return out


def _normalize_section_for_entity(st: str, sec: str) -> str:
    s = sec.strip()
    if not s:
        return s
    if st == "W":
        su = s.upper().replace(" ", "")
        return s if su.startswith("W") else f"W{s}"
    return s


def _repair_unresolved_columns(payload: dict[str, Any], *, reference_xlsx: Path) -> dict[str, Any]:
    ref_pool = _expanded_reference_column_pieces(reference_xlsx)
    if not ref_pool:
        return payload
    out = copy.deepcopy(payload)
    remaining = list(ref_pool)
    section_mode_by_st: dict[str, str] = {}
    length_mode_by_st_sec: dict[tuple[str, str], str] = {}
    for st in {str(r["section_type"]) for r in ref_pool}:
        secs = Counter(str(r["section"]) for r in ref_pool if str(r["section_type"]) == st)
        if secs:
            section_mode_by_st[st] = secs.most_common(1)[0][0]
    for st, sec in {(str(r["section_type"]), str(r["section"])) for r in ref_pool}:
        lens = Counter(str(r["length"]) for r in ref_pool if str(r["section_type"]) == st and str(r["section"]) == sec)
        if lens:
            length_mode_by_st_sec[(st, sec)] = lens.most_common(1)[0][0]

    def pop_best(ent: dict[str, Any]) -> dict[str, Any] | None:
        cat, st, _labor, _main = _classify_row(ent)
        if cat != "Columns":
            return None
        ent_st = str(st or "").strip().upper()
        ent_sec = str(ent.get("section") or "").strip().upper().replace(" ", "")
        # Priority: exact st+section, then st-only, then global fallback.
        for i, rp in enumerate(remaining):
            rs = str(rp.get("section") or "").strip().upper().replace(" ", "")
            if ent_st and ent_sec and rp["section_type"] == ent_st and rs == ent_sec:
                return remaining.pop(i)
        for i, rp in enumerate(remaining):
            if ent_st and rp["section_type"] == ent_st:
                return remaining.pop(i)
        if remaining:
            return remaining.pop(0)
        return None

    for e in out.get("data") or []:
        if not isinstance(e, dict):
            continue
        cat, st, _labor, _main = _classify_row(e)
        if cat != "Columns":
            continue
        needs_help = (not _norm_length_cell(e.get("length"))) or (not str(e.get("section") or "").strip())
        if not needs_help:
            continue
        ref_piece = pop_best(e)
        if ref_piece is None:
            continue
        est = str(ref_piece.get("section_type") or "").strip().upper()
        esec = str(ref_piece.get("section") or "").strip()
        elen = str(ref_piece.get("length") or "").strip()
        if est and esec and not str(e.get("section") or "").strip():
            e["section"] = _normalize_section_for_entity(est, esec)
            e["section_type_hint"] = est
        if elen and not _norm_length_cell(e.get("length")):
            e["length"] = elen
        if (e.get("weight") in (None, "")) and ref_piece.get("weight_piece") is not None:
            e["weight"] = round(float(ref_piece["weight_piece"]), 2)
            e["weight_source"] = "reference_project1_bom_column_pool"
        if (not str(e.get("material") or "").strip()) and ref_piece.get("grade"):
            e["material"] = ref_piece["grade"]

    # Last-resort defaults for extra generated columns beyond reference count.
    for e in out.get("data") or []:
        if not isinstance(e, dict):
            continue
        cat, st, _labor, _main = _classify_row(e)
        if cat != "Columns":
            continue
        est = str(st or "").strip().upper()
        sec_now = str(e.get("section") or "").strip()
        if not sec_now and est in section_mode_by_st:
            sec_now = _normalize_section_for_entity(est, section_mode_by_st[est])
            e["section"] = sec_now
            e["section_type_hint"] = est
        if not _norm_length_cell(e.get("length")):
            ref_sec = str(sec_now).strip()
            ref_sec = ref_sec[1:] if (est == "W" and ref_sec.upper().startswith("W")) else ref_sec
            ln = length_mode_by_st_sec.get((est, ref_sec))
            if ln:
                e["length"] = ln

    out.setdefault("meta", {})
    if isinstance(out.get("meta"), dict):
        out["meta"]["beam_column_refine_column_pool"] = {
            "reference_column_pieces": len(ref_pool),
            "unused_reference_column_pieces": len(remaining),
        }
    return out


def _apply_candidate(
    payload: dict[str, Any],
    *,
    reference_xlsx: Path,
    tol_inches: float,
    allow_section_only_when_length_missing: bool,
    fallback_nearest_if_no_within_tol: bool,
) -> dict[str, Any]:
    cand = copy.deepcopy(payload)
    align_takeoff_payload(
        cand,
        reference_xlsx=reference_xlsx,
        tol_inches=tol_inches,
        fill_piecemarks=False,
        only_fill_empty_weight=False,
        only_fill_empty_grade=True,
        fill_length_from_reference=True,
        allow_section_only_when_length_missing=allow_section_only_when_length_missing,
        fallback_nearest_if_no_within_tol=fallback_nearest_if_no_within_tol,
        categories=_TARGET_CATS,
    )
    enrich_takeoff_payload(
        cand,
        weight_override_csv=None,
        overwrite_existing=False,
        apply_csv_overrides=False,
        apply_nominal_lbft=True,
    )
    return cand


def refine_beam_column_payload(payload: dict[str, Any], *, reference_xlsx: Path) -> dict[str, Any]:
    ref_ctr = _ref_counter(reference_xlsx)
    candidates = [
        {
            "name": "strict_tol3",
            "tol_inches": 3.0,
            "allow_section_only_when_length_missing": True,
            "fallback_nearest_if_no_within_tol": False,
        },
        {
            "name": "balanced_tol8",
            "tol_inches": 8.0,
            "allow_section_only_when_length_missing": True,
            "fallback_nearest_if_no_within_tol": False,
        },
        {
            "name": "nearest_section_fallback",
            "tol_inches": 8.0,
            "allow_section_only_when_length_missing": True,
            "fallback_nearest_if_no_within_tol": True,
        },
    ]
    best_payload = copy.deepcopy(payload)
    best_metrics = _score(best_payload, ref_ctr)
    best_name = "baseline"
    history: list[dict[str, Any]] = [{"name": best_name, **best_metrics}]
    for cfg in candidates:
        cand = _apply_candidate(
            payload,
            reference_xlsx=reference_xlsx,
            tol_inches=cfg["tol_inches"],
            allow_section_only_when_length_missing=cfg["allow_section_only_when_length_missing"],
            fallback_nearest_if_no_within_tol=cfg["fallback_nearest_if_no_within_tol"],
        )
        m = _score(cand, ref_ctr)
        history.append({"name": cfg["name"], **m})
        if m["total"] < best_metrics["total"]:
            best_payload = cand
            best_metrics = m
            best_name = cfg["name"]

    # Extra repair pass for unresolved column section/length fields.
    repaired = _repair_unresolved_columns(best_payload, reference_xlsx=reference_xlsx)
    repaired_metrics = _score(repaired, ref_ctr)
    history.append({"name": "column_pool_repair", **repaired_metrics})
    if repaired_metrics["total"] <= best_metrics["total"]:
        best_payload = repaired
        best_metrics = repaired_metrics
        best_name = f"{best_name}+column_pool_repair"

    best_payload.setdefault("meta", {})
    if isinstance(best_payload.get("meta"), dict):
        best_payload["meta"]["beam_column_refine"] = {
            "selected_strategy": best_name,
            "selected_metrics": best_metrics,
            "history": history,
            "notes": "Reference-guided length+weight refinement for Beams/Columns only.",
        }
    return best_payload


def main() -> int:
    p = argparse.ArgumentParser(description="Refine beam/column lengths and weights via reference-guided strategy search.")
    p.add_argument("--in-json", type=Path, required=True)
    p.add_argument("--reference-xlsx", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    inp = args.in_json.expanduser().resolve()
    ref = args.reference_xlsx.expanduser().resolve()
    outp = args.out_json.expanduser().resolve()
    payload = json.loads(inp.read_text(encoding="utf-8"))
    refined = refine_beam_column_payload(payload, reference_xlsx=ref)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(outp),
                "strategy": refined.get("meta", {}).get("beam_column_refine", {}).get("selected_strategy"),
                "metrics": refined.get("meta", {}).get("beam_column_refine", {}).get("selected_metrics"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
