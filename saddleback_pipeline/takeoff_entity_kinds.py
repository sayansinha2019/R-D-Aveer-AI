"""Entity kind helpers for splitting takeoff payloads (beams vs columns vs ancillary).

Used by column/ancillary pipelines and ``takeoff_compose``. The beam post-process
(``hybrid_postprocess``) is unchanged and does not import this module.
"""

from __future__ import annotations

from typing import Any, Callable


def is_beam_entity(entity: dict[str, Any]) -> bool:
    et = str(entity.get("element_type") or "").strip().lower()
    pg = str(entity.get("parent_group") or "").strip().lower()
    if et == "beam" or "beam" in et or et in {"rafter", "girder", "joist"}:
        return True
    return pg == "beams"


def is_column_entity(entity: dict[str, Any]) -> bool:
    et = str(entity.get("element_type") or "").strip().lower()
    pg = str(entity.get("parent_group") or "").strip().lower()
    if et == "column" or "column" in et:
        return True
    return pg == "columns"


def is_ancillary_entity(entity: dict[str, Any]) -> bool:
    return not is_beam_entity(entity) and not is_column_entity(entity)


def entities_matching(payload: dict[str, Any], pred: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in payload.get("data") or []:
        if isinstance(e, dict) and pred(e):
            out.append(e)
    return out
