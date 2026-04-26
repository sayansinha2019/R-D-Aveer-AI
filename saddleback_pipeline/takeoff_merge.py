"""Merge multiple takeoff JSON payloads (multi-sheet / multi-PDF ingestion).

* Concatenates ``data`` (optional dedupe).
* Rolls up ``material_summary`` by normalized material + length key (schedule-style lines).
* Merges ``meta.drawing_scales.pages`` when present (for downstream QA).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saddleback_pipeline.bom_relaxed import apply_material_alias, material_match_key


def _norm_len_cell(ln: Any) -> str:
    if ln is None:
        return ""
    return str(ln).strip()


def merge_takeoff_payloads(
    payloads: list[dict[str, Any]],
    *,
    dedupe_entities: bool = False,
) -> dict[str, Any]:
    if not payloads:
        return {"data": [], "material_summary": [], "meta": {"merge": {"error": "no_inputs"}}}

    merged_data: list[dict[str, Any]] = []
    for p in payloads:
        chunk = p.get("data") or []
        if isinstance(chunk, list):
            merged_data.extend(e for e in chunk if isinstance(e, dict))

    if dedupe_entities:
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for e in merged_data:
            key = (
                material_match_key(e.get("piece_mark") or e.get("entity_id")),
                material_match_key(e.get("section")),
                _norm_len_cell(e.get("length")),
                (e.get("element_type") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        merged_data = deduped

    # Roll up material_summary across sheets
    ms_agg: dict[tuple[str, str], dict[str, Any]] = {}
    for p in payloads:
        bom = p.get("material_summary")
        if not isinstance(bom, list):
            continue
        for row in bom:
            if not isinstance(row, dict):
                continue
            try:
                q = int(row.get("qty"))
            except (TypeError, ValueError):
                continue
            mat_raw = row.get("material")
            mk = material_match_key(apply_material_alias(str(mat_raw or "")))
            ls = _norm_len_cell(row.get("length"))
            key = (mk, ls)
            if key not in ms_agg:
                ms_agg[key] = {
                    "qty": q,
                    "material": str(mat_raw).strip() if mat_raw is not None else mk,
                    "length": row.get("length"),
                    "pcmk": row.get("pcmk"),
                    "weight": row.get("weight"),
                    "grade": row.get("grade"),
                }
            else:
                ms_agg[key]["qty"] += q
                for fld in ("pcmk", "weight", "grade"):
                    if ms_agg[key].get(fld) is None and row.get(fld) is not None:
                        ms_agg[key][fld] = row.get(fld)

    scale_pages: list[dict[str, Any]] = []
    for idx, p in enumerate(payloads):
        meta = p.get("meta") if isinstance(p.get("meta"), dict) else {}
        ds = meta.get("drawing_scales") if isinstance(meta.get("drawing_scales"), dict) else {}
        for pg in ds.get("pages") or []:
            if isinstance(pg, dict):
                scale_pages.append({**pg, "merge_source_index": idx})

    out_meta: dict[str, Any] = {
        "merge": {
            "source_count": len(payloads),
            "entity_count": len(merged_data),
            "material_summary_rows": len(ms_agg),
            "dedupe_entities": dedupe_entities,
        },
    }
    if scale_pages:
        out_meta["drawing_scales"] = {
            "pages": scale_pages,
            "notes": "Merged from per-sheet takeoff meta; page numbers are per original PDF.",
        }

    return {
        "data": merged_data,
        "material_summary": list(ms_agg.values()),
        "meta": out_meta,
    }


def load_takeoff_jsons(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        p = p.expanduser().resolve()
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def write_takeoff_json(payload: dict[str, Any], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
