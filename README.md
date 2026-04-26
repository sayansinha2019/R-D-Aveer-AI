# Structural quantity takeoff (Gemini 3.1 preview)

Single pipeline: **PDF + prompt + optional schema CSV → JSON** via **Google Gemini** (`gemini-3.1-pro-preview`).

No Document AI, OpenCV, or legacy row extractors.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, INPUT_PDF, OUTPUT_JSON, SCHEMA_CSV (optional)
```

## Run

```bash
python -m saddleback_pipeline.cli
```

Output: `OUTPUT_JSON` (default `takeoff_output.json`) — engineering-grade JSON per `prompts/structural_takeoff.txt`.

### Integrated pipeline (multi-PDF, merge, reconcile, BOM, QA)

Single entry point that chains **Gemini takeoff per PDF**, **merges** `data` / `material_summary`, optional **reconcile** against a 6-column Material Summary xlsx, **Project-1 BOM** export, optional **validation_suite** (BOM metrics + vector geometry + fusion when `INPUT_PDF` / detections exist):

```bash
python -m saddleback_pipeline.integrated_pipeline \
  --pdfs "sheet_a.pdf" "sheet_b.pdf" \
  --out-dir integrated_output \
  --merged-json integrated_output/merged.json \
  --project1-bom-xlsx integrated_output/merged_bom.xlsx \
  --reference-material-xlsx "26-LQ-094_SADDLEBACK VILLAGE_Material Summary.xlsx" \
  --run-validation
```

Merge **only** existing JSON files (no Gemini):

```bash
python -m saddleback_pipeline.integrated_pipeline \
  --from-json takeoff_floor1.json takeoff_floor2.json \
  --merged-json merged.json \
  --project1-bom-xlsx merged_bom.xlsx
```

Foundation-only sheets (footing schedules, no W-shapes) need the supplement pass:

```bash
python -m saddleback_pipeline.integrated_pipeline \
  --pdfs foundation_plan.pdf framing_plan.pdf \
  --foundation-sheet-stems foundation_plan \
  --out-dir integrated_output --merged-json integrated_output/merged.json \
  --project1-bom-xlsx integrated_output/merged_bom.xlsx
```

Compare a generated Project-1 BOM xlsx to a reference BOM:

```bash
python -m saddleback_pipeline.project1_bom_compare \
  --reference "Project 1/Project-1 - BOM - April 19, 2026.xlsx" \
  --generated "Project 1/Project-1 - Generated Merged BOM.xlsx" \
  --out-json "Project 1/bom_validation_report.json"
```

Qty / weight / length / material keys (strict rollup + relaxed W-beam keys):

```bash
python -m saddleback_pipeline.project1_bom_material_accuracy \
  --reference "Project 1/Project-1 - BOM - April 19, 2026.xlsx" \
  --generated "Project 1/Project-1 - Generated Merged BOM.xlsx" \
  --out-json "Project 1/accuracy_material_qty_weight_length.json"
```

**Geometry (`pdf_geometry`, `geometry_fusion`, scales)** is still part of QA: it does not replace Gemini extraction; it scores consistency between vector lines / view boxes and takeoff rows. Optional **IfcOpenShell** model bridge: `pip install ifcopenshell`, then `--ifc model.ifc` on the integrated command (see `saddleback_pipeline/model_bridge.py`).

**Roadmap toward full shop BOM (connections, piecemarks, model alignment):** see [`docs/PIPELINE_ROADMAP.md`](docs/PIPELINE_ROADMAP.md).

**YOLO / ONNX:** set `YOLO_ONNX_MODEL` (and `pip install onnxruntime`) to merge trained beam/column boxes into the same detection JSON as Gemini spatial hints — see `.env.example` and `onnx_yolo_detector.py`.

**Steel weights:** `python -m saddleback_pipeline.steel_weight_enrichment --json takeoff.json` fills missing **lb** using AISC nominal lb/ft from **W/M/S/HP/MC** designations × cut length; optional `--weight-override-csv`. Integrated pipeline runs this before Project-1 BOM export unless `--skip-steel-weight-enrich`.

## Why it can take a long time (or produce no file until the end)

- **One blocking API call**: the script sends the **whole PDF** plus your prompt and waits for **one** completion. Nothing is written to disk until Gemini returns valid JSON.
- **Gemini 3.1 Pro Preview** on a PDF is a **heavy multimodal** job (often **several minutes**, sometimes **10–20+** on large prompts).
- Your prompt asks for **entity-expanded** output (one JSON object per physical piece). That can mean a **very large** response, long generation, and higher chance of **hitting token/time limits** if the drawing is complex.
- If the process is **stopped**, **times out**, or the model returns **non-JSON**, you will **not** see `takeoff_output.json`.

Tune with `GEMINI_HTTP_TIMEOUT_SEC` (seconds in `.env`; the code converts to **milliseconds** for the Gemini client) and `GEMINI_MAX_OUTPUT_TOKENS` in `.env` (see `.env.example`).

If you see `Read timed out. (read timeout=1.8)` while `GEMINI_HTTP_TIMEOUT_SEC` is `1800`, that was the old bug: `1800` was passed as **milliseconds** (1.8s). Update to the latest `gemini_takeoff.py` which multiplies by 1000.

## Why the Gemini **web chat** feels faster than this script

- **Streaming in the UI**: The chat shows tokens as they arrive, so it *feels* immediate. This script waited for one full response (unless you set `GEMINI_STREAM=true`).
- **Different workload**: Here you send the **entire PDF** plus a **very long** fabrication prompt and ask for a **potentially huge JSON** (entity expansion). That is much heavier than a short Q&A in the UI.
- **Model choice**: The UI may default to a **faster** model for some tasks; this pipeline uses **`GEMINI_MODEL`** (e.g. `gemini-3.1-pro-preview`), which is slower but stronger.
- **No “partial save”**: The script only writes `takeoff_output.json` after the full JSON is received and parsed.

**To see progress in the terminal**, set `GEMINI_STREAM=true` in `.env`. **To iterate faster**, temporarily use a smaller prompt or a faster model (e.g. `gemini-2.0-flash` or `gemini-2.5-flash`) for smoke tests, then switch back to Pro for final runs.

## Files

- `prompts/structural_takeoff.txt` — master prompt (entity expansion, no assumptions, plate rules, etc.)
- `schema_required.csv` — optional field list appended to the prompt as “REQUIRED OUTPUT SCHEMA”
