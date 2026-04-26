"""Extract architectural drawing scales from PDF text and compute real-world factors.

Parses strings like ``SCALE 1/4\\" = 1'`` (1/4 inch on paper = 1 foot real) and computes
``feet_per_drawing_inch`` so: ``real_feet = inches_measured_on_drawing * feet_per_drawing_inch``.

Also supports pixel pipelines: ``real_feet = (pixels / raster_dpi) * feet_per_drawing_inch``.

This does **not** replace written dimensions; it gives calibrated factors when the model
must infer from graphics. Multiple scales per sheet are all recorded (per detail/title).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saddleback_pipeline.pdf_text import get_pdf_page_texts

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _norm_text(s: str) -> str:
    t = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2033", '"')
    )
    # Some PDFs emit a backslash before ASCII quotes in extracted text (e.g. 1/4\" = 1').
    return t.replace('\\"', '"')


def _fix_missing_slash_scale_typos(t: str) -> str:
    """PDF text often drops the slash in fractions (e.g. ``1/4`` → ``14``).

    Rewrites common patterns next to SCALE so parsers see ``num/den" = …``.
    """
    out = t
    # Use callable replacers so we do not inject backslashes before inch marks.
    def _sub(pat: str, rhs: str) -> None:
        nonlocal out
        out = re.sub(pat, lambda m: m.group(1) + rhs, out)

    # Most common: 1/4 misread as 14 (slash missing in PDF text layer)
    _sub(r"(?i)(SCALE\s+)14\"\s*=\s*1'", '1/4" = 1\'')
    _sub(r"(?i)(SCALE\s+)18\"\s*=\s*1'", '1/8" = 1\'')
    _sub(r"(?i)(SCALE\s+)12\"\s*=\s*1'", '1/2" = 1\'')
    _sub(r"(?i)(SCALE\s+)38\"\s*=\s*1'", '3/8" = 1\'')
    _sub(r"(?i)(SCALE\s+)34\"\s*=\s*1'", '3/4" = 1\'')
    _sub(r"(?i)(SCALE\s+)316\"\s*=\s*1'", '3/16" = 1\'')
    return out


# Fraction on left, feet and optional inches on right: 1/4" = 1'  or  1/4" = 1'-6"
_RE_FRAC_ARCH = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*\"\s*=\s*(\d+)\s*'\s*(?:-\s*(\d+)\s*\")?",
    re.IGNORECASE,
)

# Integer inches on left: 1" = 10'  (engineering / civil).
# (?<![/]) avoids matching the ``4"`` inside ``1/4" = 1'``.
_RE_INT_ARCH = re.compile(
    r"(?<![/])(?<!\d)(\d+)\s*\"\s*=\s*(\d+)\s*'",
    re.IGNORECASE,
)

# Decimal inch on left: 0.25" = 1' (less common)
_RE_DEC_ARCH = re.compile(
    r"(\d+\.\d+)\s*\"\s*=\s*(\d+)\s*'\s*(?:-\s*(\d+)\s*\")?",
    re.IGNORECASE,
)

_RE_NOT_TO_SCALE = re.compile(
    r"NOT\s+TO\s+SCALE",
    re.IGNORECASE,
)

# Simple metric note 1 : 50  (ratio; drawing_unit : real_unit same system)
_RE_METRIC = re.compile(
    r"\b(\d+)\s*:\s*(\d+)\b",
)


def _real_feet_from_parts(feet: int, inches: int | None) -> float:
    if inches is None:
        return float(feet)
    return float(feet) + inches / 12.0


@dataclass(frozen=True)
class ScaleHit:
    """One parsed scale expression."""

    raw: str
    drawing_inches: float
    real_feet: float
    feet_per_drawing_inch: float
    kind: str  # architectural_fraction | architectural_integer | architectural_decimal | metric_ratio


def _hit_from_frac(m: re.Match[str]) -> ScaleHit | None:
    num_s, den_s, ft_s, inch_opt = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        num, den = int(num_s), int(den_s)
        if den == 0:
            return None
        d_in = num / float(den)
        ft = int(ft_s)
        inch = int(inch_opt) if inch_opt is not None else None
    except (TypeError, ValueError):
        return None
    r_ft = _real_feet_from_parts(ft, inch)
    if d_in <= 0:
        return None
    fpd = r_ft / d_in
    raw = m.group(0).strip()
    return ScaleHit(
        raw=raw,
        drawing_inches=d_in,
        real_feet=r_ft,
        feet_per_drawing_inch=fpd,
        kind="architectural_fraction",
    )


def _hit_from_int(m: re.Match[str]) -> ScaleHit | None:
    left, right_ft = m.group(1), m.group(2)
    try:
        d_in = float(left)
        r_ft = float(right_ft)
    except (TypeError, ValueError):
        return None
    if d_in <= 0:
        return None
    return ScaleHit(
        raw=m.group(0).strip(),
        drawing_inches=d_in,
        real_feet=r_ft,
        feet_per_drawing_inch=r_ft / d_in,
        kind="architectural_integer",
    )


def _hit_from_dec(m: re.Match[str]) -> ScaleHit | None:
    dec_s, ft_s, inch_opt = m.group(1), m.group(2), m.group(3)
    try:
        d_in = float(dec_s)
        ft = int(ft_s)
        inch = int(inch_opt) if inch_opt else None
    except (TypeError, ValueError):
        return None
    r_ft = _real_feet_from_parts(ft, inch)
    if d_in <= 0:
        return None
    return ScaleHit(
        raw=m.group(0).strip(),
        drawing_inches=d_in,
        real_feet=r_ft,
        feet_per_drawing_inch=r_ft / d_in,
        kind="architectural_decimal",
    )


def _metric_ratio_dicts(t: str, seen_raw: set[str]) -> list[dict[str, Any]]:
    """``1 : 50``-style ratios near the word SCALE (JSON-safe; no NaN)."""
    out: list[dict[str, Any]] = []
    for m in _RE_METRIC.finditer(t):
        if "SCALE" not in t[max(0, m.start() - 40) : m.end() + 10].upper():
            continue
        try:
            a, b = int(m.group(1)), int(m.group(2))
        except ValueError:
            continue
        if a <= 0 or b <= 0:
            continue
        raw = m.group(0).strip()
        if raw in seen_raw:
            continue
        seen_raw.add(raw)
        out.append(
            {
                "raw": raw,
                "kind": "metric_ratio",
                "ratio_drawing": a,
                "ratio_real": b,
                "real_per_drawing_unit": b / a,
                "interpretation": "1 drawing unit : N real units (same length unit).",
            }
        )
    return out


def _find_hits_in_page(text: str) -> tuple[list[ScaleHit], list[dict[str, Any]]]:
    t = _fix_missing_slash_scale_typos(_norm_text(text))
    hits: list[ScaleHit] = []
    seen_raw: set[str] = set()

    for m in _RE_FRAC_ARCH.finditer(t):
        h = _hit_from_frac(m)
        if h is None or h.raw in seen_raw:
            continue
        seen_raw.add(h.raw)
        hits.append(h)

    for m in _RE_DEC_ARCH.finditer(t):
        h = _hit_from_dec(m)
        if h is None or h.raw in seen_raw:
            continue
        seen_raw.add(h.raw)
        hits.append(h)

    for m in _RE_INT_ARCH.finditer(t):
        pre = t[max(0, m.start() - 100) : m.start()].upper()
        try:
            left_i = int(m.group(1))
        except ValueError:
            continue
        if "SCALE" not in pre and left_i > 12:
            continue
        if "SCALE" not in pre and left_i not in (1, 2, 3, 4, 6, 8, 10, 12):
            continue
        h = _hit_from_int(m)
        if h is None or h.raw in seen_raw:
            continue
        seen_raw.add(h.raw)
        hits.append(h)

    metric_extra = _metric_ratio_dicts(t, seen_raw)
    return hits, metric_extra


def extract_drawing_scales(
    pdf_path: Path,
    *,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Scan PDF page text for scale callouts.

    Returns a dict suitable for ``meta["drawing_scales"]`` and standalone JSON export.
    """
    pdf_path = pdf_path.expanduser().resolve()
    page_texts, text_engine = get_pdf_page_texts(pdf_path, max_pages=max_pages)
    n = len(page_texts)
    pages_out: list[dict[str, Any]] = []
    not_to_scale_pages: list[int] = []

    for i, raw_text in enumerate(page_texts):
        nt = _norm_text(raw_text)
        page_no = i + 1
        if _RE_NOT_TO_SCALE.search(nt):
            not_to_scale_pages.append(page_no)

        hits, metric_extra = _find_hits_in_page(raw_text)
        scales_serialized: list[dict[str, Any]] = []
        for h in hits:
            scales_serialized.append(
                {
                    "raw": h.raw,
                    "drawing_inches": h.drawing_inches,
                    "real_feet": h.real_feet,
                    "feet_per_drawing_inch": h.feet_per_drawing_inch,
                    "kind": h.kind,
                }
            )
        scales_serialized.extend(metric_extra)
        if scales_serialized or (page_no in not_to_scale_pages):
            pages_out.append(
                {
                    "page": page_no,
                    "scales": scales_serialized,
                    "not_to_scale_mentioned": page_no in not_to_scale_pages,
                }
            )

    return {
        "pdf": str(pdf_path),
        "pages_scanned": n,
        "text_engine": text_engine,
        "pages": pages_out,
        "not_to_scale_pages": not_to_scale_pages,
        "notes": (
            "feet_per_drawing_inch: multiply a distance measured on the drawing (in inches) "
            "by this factor to get real-world feet. For raster images: "
            "real_feet = (pixels / dpi) * feet_per_drawing_inch. "
            "Prefer written dimensions on the sheet when present."
        ),
    }


def format_scales_for_prompt(payload: dict[str, Any], *, max_chars: int = 12000) -> str:
    """Human-readable block appended to the Gemini prompt."""
    engine = payload.get("text_engine") or "pymupdf"
    lines: list[str] = [
        f"--- DETECTED DRAWING SCALES (PDF text via {engine}; apply when a dimension is missing) ---",
        "",
        "The following scales were found by parsing the PDF. Use the scale that applies to the "
        "view or detail you are reading (title block / detail bubble). When you must infer a "
        "length from the graphic only, convert using feet_per_drawing_inch:",
        "  real_feet = (distance_on_drawing_in_inches) * feet_per_drawing_inch",
        "For a measurement in pixels on a raster image at DPI:",
        "  real_feet = (pixels / DPI) * feet_per_drawing_inch",
        "",
    ]
    if payload.get("not_to_scale_pages"):
        lines.append(
            f"Pages mentioning NOT TO SCALE (do not scale graphics): {payload['not_to_scale_pages']}"
        )
        lines.append("")

    for pg in payload.get("pages", []):
        pno = pg.get("page")
        scales = pg.get("scales") or []
        nts = pg.get("not_to_scale_mentioned")
        if not scales and not nts:
            continue
        lines.append(f"Page {pno}:")
        if nts and not scales:
            lines.append("  (NOT TO SCALE noted; no numeric scale parsed on this page text.)")
        for s in scales:
            if str(s.get("kind", "")).startswith("metric_ratio"):
                lines.append(
                    f"  - {s.get('raw')!s}  [metric ratio — interpret per project units]"
                )
                continue
            fpd = s.get("feet_per_drawing_inch")
            di = s.get("drawing_inches")
            rf = s.get("real_feet")
            if isinstance(fpd, (int, float)) and fpd == fpd:  # not NaN
                lines.append(
                    f"  - {s.get('raw')!s}  →  "
                    f"{di!r} in on drawing = {rf!r} ft real; "
                    f"feet_per_drawing_inch = {fpd:.6g}"
                )
            else:
                lines.append(f"  - {s.get('raw')!s}")
        lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n... [truncated]"
    return text


def real_feet_from_drawing_inches(
    drawing_inches: float,
    feet_per_drawing_inch: float,
) -> float:
    """Convert a span measured on the drawing (inches on paper) to real feet."""
    return float(drawing_inches) * float(feet_per_drawing_inch)


def real_feet_from_raster_pixels(
    pixels: float,
    *,
    raster_dpi: float,
    feet_per_drawing_inch: float,
) -> float:
    """Convert pixel length along a scaled view to real feet (same axis as drawing)."""
    drawing_inches = float(pixels) / float(raster_dpi)
    return real_feet_from_drawing_inches(drawing_inches, feet_per_drawing_inch)


def merge_scales_into_takeoff(takeoff: dict[str, Any], scales_payload: dict[str, Any]) -> None:
    """Attach ``drawing_scales`` to ``meta`` without clobbering other meta keys."""
    takeoff.setdefault("meta", {})
    if not isinstance(takeoff["meta"], dict):
        takeoff["meta"] = {}
    takeoff["meta"]["drawing_scales"] = scales_payload


def main() -> int:
    import argparse
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    parser = argparse.ArgumentParser(description="Extract drawing scales from a PDF (text).")
    parser.add_argument("pdf", nargs="?", help="PDF path (default: INPUT_PDF from .env)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON (default: drawing_scales.json or OUTPUT_DRAWING_SCALES_JSON)",
    )
    args = parser.parse_args()
    import os

    raw = (args.pdf or os.getenv("INPUT_PDF", "") or "").strip()
    if not raw:
        print("ERROR: Pass a PDF path or set INPUT_PDF in .env", file=sys.stderr)
        return 1
    pdf = Path(raw).expanduser()
    if not pdf.is_file():
        print(f"ERROR: PDF not found: {pdf}", file=sys.stderr)
        return 1
    max_pages_env = (os.getenv("PDF_MAX_PAGES", "") or "").strip()
    max_pages = int(max_pages_env) if max_pages_env else None

    payload = extract_drawing_scales(pdf, max_pages=max_pages)
    out = args.output
    if out is None:
        out_s = (os.getenv("OUTPUT_DRAWING_SCALES_JSON", "") or "").strip()
        out = Path(out_s or "drawing_scales.json").expanduser()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(out),
                "text_engine": payload.get("text_engine"),
                "pages_with_scales": len(payload.get("pages", [])),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
