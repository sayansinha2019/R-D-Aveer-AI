"""Bridge to **authoring models** (IFC / BIM) as system-of-record — optional, dependency-heavy.

Open source (typical stack)
--------------------------
* **IfcOpenShell** (Python): read IFC, filter ``IfcBeam`` / ``IfcColumn`` / ``IfcPlate`` /
  ``IfcFastener``, map to fabrication-style dicts. Install: ``pip install ifcopenshell``
  (platform wheels vary; not enabled in default ``requirements.txt``).

Commercial / vendor APIs (higher structure fidelity when model is the contract)
---------------------------------------------------------------------------------
* **Autodesk APS** (Forge): translate RVT → SVF / extract properties (cloud, subscription).
* **Trimble Connect** + **Tekla** ecosystem APIs (project + model data where licensed).
* **Speckle**: open **server** + connectors (Revit/Tekla/Rhino) — good for diffing /
  streaming geometry; hosted Speckle is a service.

This repo keeps **PDF + Gemini** as the portable default; call into this module when you
have an IFC path and want **reconciliation** (compare takeoff JSON vs model-derived BOM),
not as a replacement for drawings on site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ifcopenshell_available() -> bool:
    try:
        import ifcopenshell  # noqa: F401

        return True
    except ImportError:
        return False


def load_steel_elements_from_ifc(ifc_path: Path) -> list[dict[str, Any]]:
    """Return lightweight dict rows suitable for diffing against ``material_summary``.

    Raises ``ImportError`` if IfcOpenShell is not installed.
    """
    ifc_path = ifc_path.expanduser().resolve()
    if not ifc_path.is_file():
        raise FileNotFoundError(ifc_path)

    import ifcopenshell  # type: ignore[import-untyped]

    f = ifcopenshell.open(str(ifc_path))
    rows: list[dict[str, Any]] = []

    def _tag(el: Any) -> str:
        return el.is_a()

    for cls in ("IfcBeam", "IfcColumn", "IfcMember", "IfcPlate"):
        try:
            for el in f.by_type(cls):
                name = getattr(el, "Name", None) or getattr(el, "GlobalId", None)
                # Tag + name as pseudo-material; real apps map Pset_* quantities.
                rows.append(
                    {
                        "ifc_class": cls,
                        "name": str(name) if name else None,
                        "global_id": getattr(el, "GlobalId", None),
                        "source": str(ifc_path),
                    },
                )
        except Exception:
            continue

    return rows


def summarize_model_vs_takeoff_stub(
    model_rows: list[dict[str, Any]],
    takeoff: dict[str, Any],
) -> dict[str, Any]:
    """Placeholder for quantitative reconciliation (counts, unmatched IDs).

    Extend with: normalize model profile strings to ``material_summary`` keys,
    then reuse ``bom_relaxed`` / ``bom_accuracy`` matchers.
    """
    n_model = len(model_rows)
    n_data = len(takeoff.get("data") or [])
    n_ms = len(takeoff.get("material_summary") or [])
    return {
        "model_element_count": n_model,
        "takeoff_data_count": n_data,
        "takeoff_material_summary_rows": n_ms,
        "note": "Implement key alignment (piece mark / profile) for real F1-style scores.",
    }
