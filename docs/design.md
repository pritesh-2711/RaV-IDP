# RaV-IDP Design Document

This document is the working design reference for the RaV-IDP system. I update it when the code direction changes so the design stays aligned with what the repository actually does.

Status: April 2026

---

## What I am building

RaV-IDP is an intelligent document extraction pipeline that does more than just read pixels. The goal is to:

- separate layout first
- process each detected region independently
- extract structured or semantic content by entity type
- validate extractions with reconstruction-based checks
- preserve enough intermediate evidence that I can inspect failures visually

The end goal is not only benchmark performance. The system should become a document extraction service that can later support a RAG workflow or an automation pipeline.

---

## Core design idea

The pipeline extracts an entity from a document region, reconstructs a representation of that entity, and compares the reconstruction against the original source region.

If the reconstruction agrees with the source crop, the extraction is treated as reliable.
If the reconstruction diverges, a fallback extractor can be triggered and compared again.

One hard rule governs the system:

> Comparison must always anchor against the original document crop, never against the extraction itself.

This is the bootstrap constraint. If it is violated, a bad extraction can reconstruct its own mistakes and still appear correct.

---

## Current architecture

The pipeline is layout-first by design.

```text
Document
  -> 1. Page Rendering
  -> 2. Layout Detection
  -> 3. Region Quality Classification
  -> 4. Region Pre-processing
  -> 5. Entity Routing
  -> 6. Primary Extraction
  -> 7. Reconstruction
  -> 8. Fidelity Comparison
       -> if low fidelity: fallback extraction + re-compare
  -> 9. Context Enrichment
  -> 10. Image Semantic Enrichment
  -> 11. Final Entity Record
```

The most important architectural correction during implementation was this:

- layout detection runs before region quality classification and preprocessing
- the full page is not classified or preprocessed before layout detection
- only individual detected regions are preprocessed after layout separation

That ordering matches the logic of mixed-content documents much better, especially when text, images, tables, labels, and formulas coexist on the same page.

---

## How each stage works

### 1. Page Rendering

Each page is rendered into a stable image representation so the downstream region-level pipeline behaves consistently across PDFs and image inputs.

### 2. Layout Detection

Docling detects text regions, table regions, and picture regions. This step defines the fundamental unit of work for the rest of the pipeline.

### 3. Region Quality Classification

Each detected region is classified independently rather than treating the whole page as a single quality unit. Skew and sharpness heuristics decide whether a crop is clean, scanned-clean, or scanned-degraded.

### 4. Region Pre-processing

Preprocessing happens only after the crop type is known. For non-image regions, transformations such as deskewing, binarization, or CLAHE are applied depending on quality.

### 5. Entity Routing

Each region is routed into a type-specific path:

- table
- image
- text
- formula
- url

### 6. Primary Extraction

The primary extractor depends on the region type:

- tables: structured reconstruction from Docling table records
- images: crop extraction from the document region
- text-like regions: text extraction from the region record

### 7. Reconstruction

An independent proxy of the extracted content is reconstructed:

- table -> rendered grid image + structural signature
- image -> pHash + sharpness + caption adjacency signal
- text -> re-OCR or PDF text stream

### 8. Fidelity Comparison

The reconstruction is compared against the source crop to compute a fidelity score.

If the score falls below the threshold, GPT-4o fallback extraction can be triggered and compared again.

### 9. Context Enrichment

Nearby text and captions are attached so each extracted entity carries useful document context.

### 10. Image Semantic Enrichment

Semantic enrichment runs for image entities after fidelity validation when an API key is available. This matters for the service goal because image regions often need semantic descriptions, OCR text, and structured interpretation, not just pixel crops.

### 11. Final Record

Each region is packaged into a final entity record with:

- entity type
- bounding box
- extracted content
- fidelity score
- low-confidence flag
- context
- provenance

---

## Fidelity formulas currently used

These formulas describe the current code path, not a final paper-locked definition.

### Table

```text
f = 0.5 * grid_match + 0.2 * (1 - CER_headers) + 0.3 * (1 - CER_cells)
```

`grid_match` is currently treated as agreement between extracted row/column structure and signals derived from the original crop.

The table comparator is still under active revision. It runs, but it is not yet considered solved.

### Image

```text
f = 0.6 * pHash_similarity + 0.3 * sharpness_ratio + 0.1 * caption_check
```

This fidelity score measures extraction stability, not semantic usefulness. Image semantic enrichment is therefore treated as a separate layer on top of pixel-level fidelity.

### Text

```text
f = max(0, 1 - CER)
```

For evaluation, overlap-aware metrics are also used because dataset annotations do not always include every visible text fragment on the page.

---

## Thresholds currently used

| Entity | Default threshold |
|--------|-------------------|
| Table  | 0.75 |
| Image  | 0.70 |
| Text   | 0.85 |

These thresholds will likely evolve after more inspection-driven review on real documents.

---

## Repository map

The system is organized like this:

```text
rav_idp/
  cli.py
  config.py
  io.py
  inspection.py
  models.py
  pipeline.py
  utils.py

  components/
    layout_detector.py
    region_quality_classifier.py
    region_preprocessor.py
    entity_router.py
    context_enricher.py
    image_enricher.py
    fallback_extractor.py

    extractors/
      table.py
      image.py
      text.py

    reconstructors/
      table.py
      image.py
      text.py

    comparators/
      table.py
      image.py
      text.py

  evaluation/
    stage1_quality.py
    stage2_layout.py
    stage3a_tables.py
    stage3b_images.py
    stage3c_text.py
    stage4_fidelity.py
    stage5_reextraction.py
    stage6_endtoend.py

  data/
    registry.py
    downloader.py
    cli.py
```

---

## What changed recently

### 1. Layout-first processing became explicit

Quality classification and preprocessing are no longer treated as page-level steps that happen before layout detection. Layout is detected first, and only then are the resulting regions processed.

### 2. Always-on image semantics were added

`ImageContent` was extended to store:

- `image_type`
- `description`
- `extracted_text`
- `structured_data`

When `OPENAI_API_KEY` is available, images are enriched semantically after fidelity validation.

### 3. Text evaluation was improved

Overlap-aware metrics were added for Stage 3c because strict string-only evaluation was unfair when annotations excluded some visible text.

### 4. An inspection-first pipeline mode was added

A visual artifact recorder now allows an end-to-end document run to be inspected stage by stage through saved outputs instead of only through benchmark summaries.

This is currently the most useful workflow for making decisions.

---

## Inspection-first run design

When the CLI runs on a document, it creates a timestamped folder under:

```text
artifacts/pipeline_runs/<filename>_<datetime>/
```

Outputs are stored for each major stage:

- `00_pages`
- `01_layout`
- `02_quality_classification`
- `03_preprocessed_regions`
- `04_rav_traces`
- `05_final_output`

### What gets saved in each folder

#### `00_pages`

The rendered page images are saved before any layout analysis.

#### `01_layout`

Page overlays with detected regions plus JSON metadata for each detected region are saved here.

#### `02_quality_classification`

Page overlays labeled with quality decisions are saved here, along with one folder per region containing the original crop.

#### `03_preprocessed_regions`

The original crop and processed crop for each region are saved here so preprocessing effects can be inspected directly.

#### `04_rav_traces`

For each region, this folder contains:

- original crop
- processed crop
- primary extraction summary
- fallback summary if triggered
- reconstruction summary
- fidelity scores
- provenance

For image and table regions, reconstructed visual artifacts are also saved where applicable.

#### `05_final_output`

This folder contains:

- final page overlays
- final `entity_records.json`
- run summary

This inspection layer is now central to how the system will be refined.

---

## Current implementation status

### Stable enough to use now

- end-to-end pipeline orchestration
- layout-first region flow
- region quality classification
- region preprocessing
- entity routing
- text extraction path
- image extraction path
- context enrichment
- image semantic enrichment when API key is available
- visual pipeline inspection outputs

### Implemented but still under revision

- table extraction and table fidelity comparison
- fidelity reliability study as a paper result
- fallback recovery study as a paper result

### Not yet the focus

- full benchmark expansion
- paper result population
- arXiv LaTeX conversion

---

## Current evaluation picture

Small real-data checks have already been run on the main extraction stages.

### Stage 3a: tables

The table benchmark path runs on real PubTabNet slices. It is technically operational, but row structure quality is still weak. Table improvements are intentionally not being prioritized right now.

### Stage 3b: images

The image benchmark runs on real ScanBank slices. Pixel-level extraction is stable, and semantic enrichment works when the API key is loaded from `experiment/.env`.

### Stage 3c: text

The text benchmark runs on real FUNSD slices. The evaluator now includes overlap-aware metrics, which better match the annotation-coverage objective.

### End-to-end inspection path

The pipeline has already run end to end on a sample local document image and produced a full timestamped inspection folder with stage outputs.

Right now, that matters more than squeezing out another round of isolated benchmark numbers.

---

## How evaluation is treated now

Component-wise evaluation still matters, but benchmark loops alone are no longer enough for design decisions.

From this point forward, two complementary lenses guide the work:

1. benchmark runs for coarse quantitative signals
2. inspection-first pipeline runs for qualitative and architectural debugging

If the benchmarks and the inspection folders disagree, the inspection evidence takes priority and the benchmark logic gets redesigned accordingly.

---

## Dependencies used

The runtime stack currently depends on:

- `docling`
- `pymupdf`
- `pydantic`
- `opencv-python-headless`
- `pillow`
- `pytesseract`
- `rapidocr`
- `onnxruntime`
- `pandas`
- `numpy`
- `scipy`
- `torch`
- `torchvision`
- `transformers`
- `timm`
- `openai`
- `python-dotenv`
- `beautifulsoup4`
- `apted`

The packaging metadata has already been updated to include the missing heavy runtime pieces used by the newer evaluation and extraction paths.

---

## Stage-to-code map

The following mapping connects paper stages to code:

| Stage | Purpose | Primary code |
|-------|---------|--------------|
| 1 | Quality assessment | `rav_idp/evaluation/stage1_quality.py`, `rav_idp/components/region_quality_classifier.py` |
| 2 | Layout detection | `rav_idp/evaluation/stage2_layout.py`, `rav_idp/components/layout_detector.py`, `rav_idp/components/page_renderer.py` |
| 3a | Table extraction | `rav_idp/evaluation/stage3a_tables.py`, `rav_idp/components/extractors/table.py`, `rav_idp/components/reconstructors/table.py`, `rav_idp/components/comparators/table.py` |
| 3b | Image extraction and semantics | `rav_idp/evaluation/stage3b_images.py`, `rav_idp/components/extractors/image.py`, `rav_idp/components/reconstructors/image.py`, `rav_idp/components/comparators/image.py`, `rav_idp/components/image_enricher.py` |
| 3c | Text extraction | `rav_idp/evaluation/stage3c_text.py`, `rav_idp/components/extractors/text.py`, `rav_idp/components/reconstructors/text.py`, `rav_idp/components/comparators/text.py` |
| 4 | Fidelity reliability study | `rav_idp/evaluation/stage4_fidelity.py` |
| 5 | GPT fallback recovery study | `rav_idp/evaluation/stage5_reextraction.py`, `rav_idp/components/fallback_extractor.py`, `rav_idp/pipeline.py` |
| 6 | End-to-end path | `rav_idp/evaluation/stage6_endtoend.py`, `rav_idp/pipeline.py` |

---

## Current priorities

The project is intentionally shifting from “benchmark everything first” to “inspect real pipeline behavior first.”

The next priorities are:

1. run the inspection pipeline on real target documents
2. review each stage folder manually
3. refine image semantic output for downstream RAG and automation use
4. refine text acceptance logic around annotation coverage and extra real text
5. return to larger benchmark runs only after inspection shows me what to fix

Table improvements are explicitly not being prioritized right now.

---

## How the pipeline runs now

```bash
cd /home/pritesh-jha/projects/rav-idp/experiment
/home/pritesh-jha/projects/rav-idp/.venv/bin/python -m rav_idp.cli /path/to/document.pdf
```

For only the final JSON and no inspection artifacts:

```bash
/home/pritesh-jha/projects/rav-idp/.venv/bin/python -m rav_idp.cli /path/to/document.pdf --no-visuals
```

---

## Main open questions

- How should image semantics be packaged so they are genuinely useful for retrieval and automation?
- How should text extraction be scored when the model finds real text that the annotator chose not to label?
- How should table structure validation eventually be redesigned so it penalizes structural failure more honestly?
- Which fidelity thresholds should be used for routing decisions in production?

---

## Design conclusion

At this stage, RaV-IDP is more than a benchmark project. It already has a working end-to-end system with a visual inspection trail, and that gives a much stronger basis for improving the service than isolated metric loops alone.

The near-term design strategy is simple:

- keep the pipeline modular
- inspect every stage visually
- improve the regions that matter for real documents
- use benchmarks to confirm progress, not to guess what the architecture should be
