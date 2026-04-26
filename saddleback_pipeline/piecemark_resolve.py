"""Map drawing labels to shop / schedule piecemarks using **external** tables (CSV).

Expected CSV columns (header row)::

    drawing_label,piece_mark,notes

Example row: ``W21X48 (28),B_24,framing plan grid 2``

Use after takeoff so ``piece_mark`` on entities can be normalized before BOM export.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_label_to_piecemark_csv(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    m: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lbl = (row.get("drawing_label") or "").strip()
            pm = (row.get("piece_mark") or "").strip()
            if lbl and pm:
                m[lbl.upper()] = pm
    return m


def apply_piecemark_map(
    entities: list[dict[str, Any]],
    label_map: dict[str, str],
    *,
    section_field: str = "section",
) -> list[dict[str, Any]]:
    """Copy entities; set ``piece_mark`` when ``section`` (or label) matches a CSV key."""
    out: list[dict[str, Any]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        ne = dict(e)
        sec = (ne.get(section_field) or "").strip()
        key = sec.upper()
        if not (ne.get("piece_mark") or "").strip() and key in label_map:
            ne["piece_mark"] = label_map[key]
        out.append(ne)
    return out
