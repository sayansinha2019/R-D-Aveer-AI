"""Merge independent takeoff slices: beams (from hybrid post-process) + columns + ancillary.

Typical flow::

    integrated_pipeline → hybrid_postprocess (beam swap) → beam_source.json
    column_takeoff_pipeline → columns_only.json
    ancillary_takeoff_pipeline → ancillary_only.json
    takeoff_compose → final.json → project1_bom_export

If ``--columns-json`` / ``--ancillary-json`` are omitted, those entities are taken from
``--beam-source-json`` so existing one-file workflows keep working.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from saddleback_pipeline.takeoff_entity_kinds import (
    entities_matching,
    is_ancillary_entity,
    is_beam_entity,
    is_column_entity,
)
from saddleback_pipeline.takeoff_merge import merge_takeoff_payloads, write_takeoff_json


def compose_takeoff(
    beam_source: dict[str, Any],
    *,
    columns_override: dict[str, Any] | None = None,
    ancillary_override: dict[str, Any] | None = None,
    inherit_columns_when_no_override: bool = True,
    inherit_ancillary_when_no_override: bool = True,
) -> dict[str, Any]:
    beams_data = entities_matching(beam_source, is_beam_entity)
    beams_part: dict[str, Any] = {
        "data": beams_data,
        "material_summary": [],
        "meta": {"compose_slice": "beams_from_beam_source"},
    }

    if columns_override is not None:
        cols_data = entities_matching(columns_override, is_column_entity)
        cols_ms = list(columns_override.get("material_summary") or [])
        cols_part = {
            "data": cols_data,
            "material_summary": cols_ms,
            "meta": {"compose_slice": "columns_explicit"},
        }
    elif inherit_columns_when_no_override:
        cols_part = {
            "data": entities_matching(beam_source, is_column_entity),
            "material_summary": [],
            "meta": {"compose_slice": "columns_from_beam_source"},
        }
    else:
        cols_part = {"data": [], "material_summary": [], "meta": {"compose_slice": "columns_empty"}}

    if ancillary_override is not None:
        anc_data = entities_matching(ancillary_override, is_ancillary_entity)
        anc_ms = list(ancillary_override.get("material_summary") or [])
        anc_part = {
            "data": anc_data,
            "material_summary": anc_ms,
            "meta": {"compose_slice": "ancillary_explicit"},
        }
    elif inherit_ancillary_when_no_override:
        anc_part = {
            "data": entities_matching(beam_source, is_ancillary_entity),
            "material_summary": [],
            "meta": {"compose_slice": "ancillary_from_beam_source"},
        }
    else:
        anc_part = {"data": [], "material_summary": [], "meta": {"compose_slice": "ancillary_empty"}}

    merged = merge_takeoff_payloads([beams_part, cols_part, anc_part], dedupe_entities=False)

    base_meta = beam_source.get("meta") if isinstance(beam_source.get("meta"), dict) else {}
    compose_meta = {
        "takeoff_compose": {
            "beams": len(beams_part.get("data") or []),
            "columns": len(cols_part.get("data") or []),
            "ancillary": len(anc_part.get("data") or []),
            "columns_source": cols_part.get("meta", {}).get("compose_slice"),
            "ancillary_source": anc_part.get("meta", {}).get("compose_slice"),
        }
    }
    merged["meta"] = {**base_meta, **(merged.get("meta") or {}), **compose_meta}
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose beam + column + ancillary takeoff JSON slices.")
    ap.add_argument("--beam-source-json", type=Path, required=True, help="Usually hybrid_postprocess output (beams + optional fallbacks).")
    ap.add_argument("--columns-json", type=Path, default=None, help="Column pipeline output; omit to inherit columns from beam source.")
    ap.add_argument("--ancillary-json", type=Path, default=None, help="Ancillary pipeline output; omit to inherit from beam source.")
    ap.add_argument(
        "--no-inherit-columns",
        action="store_true",
        help="If set and --columns-json omitted, emit zero columns.",
    )
    ap.add_argument(
        "--no-inherit-ancillary",
        action="store_true",
        help="If set and --ancillary-json omitted, emit zero ancillary rows.",
    )
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    beam_src = json.loads(args.beam_source_json.expanduser().resolve().read_text(encoding="utf-8"))
    col_ov = None
    if args.columns_json is not None:
        col_ov = json.loads(args.columns_json.expanduser().resolve().read_text(encoding="utf-8"))
    anc_ov = None
    if args.ancillary_json is not None:
        anc_ov = json.loads(args.ancillary_json.expanduser().resolve().read_text(encoding="utf-8"))

    out = compose_takeoff(
        beam_src,
        columns_override=col_ov,
        ancillary_override=anc_ov,
        inherit_columns_when_no_override=not args.no_inherit_columns,
        inherit_ancillary_when_no_override=not args.no_inherit_ancillary,
    )
    outp = args.out_json.expanduser().resolve()
    write_takeoff_json(out, outp)
    print(json.dumps({"out_json": str(outp), "meta": out.get("meta", {}).get("takeoff_compose")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
