# Pipeline roadmap: toward full shop BOM (Tekla-style) output

Current state: **Gemini takeoff → merge → Project-1 BOM sheet** captures **members + foundations** from the PDFs you feed. The reference BOM (`Project-1 - BOM - …`) reflects a **fabrication model**: same piecemarks on main + clips + bolts + plates + anchors + columns — **not** reproducible from two sheets alone without extra inputs.

Below is an ordered path to close the gap.

---

## SDS2 / 3D and piecemarks

**SDS2** (like Tekla SDS2 / Advance Steel) is not “just a 3D viewer”: it is **steel detailing software** where every member and connection exists in a **database** before drawings. Piecemarks (**`B_24`**, …) are assigned **in that model**, and the same IDs flow to **shop drawings + BOM**.

So parity with **`B_*` piecemarks** does **not** come from rotating a 3D mesh — it comes from **that database path** (export BOM/IFC/KISS, or schedules from the detailer). PDF vision alone learns **symbols and text**, not fabrication IDs, unless you **map** drawing labels → marks (CSV).

---

## Phase 1 — Inputs (biggest leverage)

| Step | Goal | Repo hook / action |
|------|------|-------------------|
| **1a. Full drawing set** | Plans, elevations, sections, **member schedules**, **connection details**. | Batch folder: loop PDFs → `integrated_pipeline --pdfs …` or many `--from-json` after parallel takeoffs. |
| **1b. Schedule-first policy** | Trust schedule quantities; use geometry for lengths. | Tune `prompts/structural_takeoff.txt`; tag entities where `source_reference` contains `SCHEDULE` for QA. |
| **1c. BIM / IFC as anchor** | Names, profiles, approximate counts vs model. | Optional `pip install ifcopenshell`; `--ifc model.ifc` then extend `model_bridge.py` with profile→string mapping and diff reports (not guesses). |

Deliverable: merged JSON whose **`material_summary`** is closer to estimating sheets; **`data`** still expanded for audit.

---

## Phase 2 — Piecemarks (`B_24`-style parity)

| Step | Goal | Repo hook |
|------|------|-----------|
| **2a. Schedule CSV** | Map drawing strings (`W21X48 (28)`, grid labels) → shop piecemark. | `piecemark_resolve.py` + **`--piecemark-csv`** on `integrated_pipeline`. Maintain one CSV per job. |
| **2b. Export from authoring tool** | Revit / Tekla schedule export → CSV join key. | Same CSV format: `drawing_label,piece_mark,notes`. |

Deliverable: BOM **Piecemark** column aligns with fabrication naming for rows you could not infer from geometry alone.

---

## Phase 3 — Assembly rows (clips, bolts, plates under same mark)

Without a **model export** or an **approved rule library**, expanding connections is risky (invents hardware).

| Track | Approach |
|-------|----------|
| **A — Preferred** | Import BOM or IFC from Tekla / SDS2 / Advance Steel → **normalize** into Project-1 columns (mapper script). PDF pipeline validates / fills gaps only. |
| **B — Rules** | Engineer-approved YAML/DB: shear tab family → bolt rows, angles, plates. Implement in `assembly_expand.py` behind `--assembly-rules path.yaml` (future). |
| **C — Hybrid** | Gemini reads **detail sheets** only for marks that appear on the plan; strict prompt “no hardware without detail reference”. |

Deliverable: **Category** mix (Bolts, Clips, …) moves toward reference **only** when rules or model data exist.

---

## Phase 4 — QA and numeric gates

| Step | Goal | Repo hook |
|------|------|-----------|
| **4a. Material Summary reconcile** | Align LLM strings to estimator / reference **6-column** xlsx. | `integrated_pipeline --reference-material-xlsx …` + `material_summary_reconcile.py`. |
| **4b. Project-1 BOM compare** | Track W-shape rollups vs shop BOM. | `python -m saddleback_pipeline.project1_bom_compare …` |
| **4c. Geometry QA** | Catch impossible lengths / missing columns in views. | `validation_suite`, `geometry_fusion.py` (needs spatial detections for best value). |

Deliverable: **CI-style** thresholds on compare JSON (e.g. min W-shape qty match rate on shared sections before release).

---

## Weights (shop BOM parity)

1. **Prompt**: require **weight in lb** when schedules/tags show it; allow **traced** nominal ``lb/ft × ft`` for W/M/S/HP/MC (designation encodes lb/ft per AISC).
2. **Post-process**: ``steel_weight_enrichment.py`` fills null weights from nominal lb/ft × length; optional **CSV** overrides (fabricator / calculator export keyed by section + length). Enable via integrated pipeline (default **on**) or ``STEEL_WEIGHT_ENRICH_AUTO=true`` on single-PDF takeoff / ``full_pipeline``.

---

## Learned detectors (YOLO / ONNX) vs Gemini spatial

| Approach | Strength | Limit |
|----------|-----------|-------|
| **Gemini spatial pass** (`GEMINI_SPATIAL_DETECTION`) | No training; proposes views + boxes from pixels | Not a calibrated detector; can miss small repeats |
| **YOLO ONNX** (`YOLO_ONNX_MODEL`) | Fast, repeatable **localization** once you train on **your** drawing style | Needs **labeled crops** + export ONNX; class names via `YOLO_CLASS_NAMES` |
| **Merged** | Same `detections_output.json` schema; **deduped** IoU merge | Best used as **hints** to the main takeoff prompt, not sole truth |

Optional env:

* `YOLO_ONNX_MODEL` — path to `.onnx` (Ultralytics YOLOv8-style export recommended)
* `LEARNED_DETECTION_JSON` — precomputed boxes from **any** tool (same JSON shape as Gemini spatial output)
* `onnxruntime` — `pip install onnxruntime`

The main takeoff still uses **Gemini** for quantities and text; boxes **reduce omissions** and support **geometry_fusion** QA.

---

## Phase 5 — Commercial / hosted APIs (when licensed)

Use where the **contract** is the model, not the PDF:

- **Autodesk APS** (cloud RVT processing)
- **Trimble / Tekla** ecosystem APIs
- **Speckle** (streams from Revit/Tekla)

Keep **Gemini** for drawing ingestion; use vendor APIs to **merge authoritative** BOM rows or to **score** PDF takeoff vs model.

---

## One-line “definition of done” for shop-grade output

You can claim parity with the reference class of BOM when **all** are true:

1. **Inputs**: Full structural PDF set **or** IFC/Tekla export for steel scope.  
2. **Piecemarks**: Mapped via schedule CSV (or model).  
3. **Connections**: From **model BOM** or **validated rule pack**, not prompt-only LLM.  
4. **Metrics**: `project1_bom_compare` + optional `bom_accuracy` / reconciled material summary meet project thresholds.

---

## Suggested order of execution (practical)

1. Expand PDF coverage + merge (already automated).  
2. Add **piecemark CSV** for one pilot grid / level.  
3. Attach **reference Material Summary** xlsx for reconcile on `material_summary`.  
4. Pilot **IFC** diff (`model_bridge`) if IFC exists.  
5. Only then invest in **assembly rules** or Tekla BOM import for the remaining gap.
