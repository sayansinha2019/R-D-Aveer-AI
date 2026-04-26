from __future__ import annotations

"""
Structural takeoff CLI: PDF + prompt + Gemini → takeoff JSON only.

For **reconciled BOM + Excel in one command**, use::

    python -m saddleback_pipeline.full_pipeline

For **multi-PDF / merge / reconcile / Project-1 BOM / validation** in one command, use::

    python -m saddleback_pipeline.integrated_pipeline --help

Optional ``GEMINI_SPATIAL_DETECTION`` runs a separate Gemini pass for bounding-box
hints (see ``gemini_spatial_detection.py``); classical CNN detectors are not included.
"""

import sys

from saddleback_pipeline.gemini_takeoff import main as takeoff_main


def main() -> int:
    return takeoff_main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
