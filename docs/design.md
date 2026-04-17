# RaV-IDP Design Document

Status as of April 2026. Use this as a reference for what is done and a
checklist for what comes next.

---

## What the system does

RaV-IDP is an intelligent document processing pipeline that wraps around
existing extraction tools and validates each extraction by reconstruction.

The core idea: after extracting an entity (table, image, text block) from a
document, reconstruct a representation of it and compare that reconstruction
against the original document region. If they match, the extraction is
trustworthy. If they diverge, trigger a fallback extractor (GPT-4o vision)
and try again.

The thing that makes this non-trivial: the comparator must always anchor
against the original document crop, never against the extraction itself.
Otherwise a wrong extraction that reconstructs cleanly will always pass. This
is called the bootstrap constraint.

---

## Architecture

Eight components in sequence:

```
Document
  └─ 1. Layout Detector          (Docling)
  └─ 2. Region Quality Classifier
  └─ 3. Region Pre-processor
  └─ 4. Entity Router            (Table / Image / Text / Formula / URL)
  └─ 5. Entity Extractors
  └─ 6. Reconstructors
  └─ 7. Comparators              → fidelity score f ∈ [0,1]
         ├─ f ≥ τ  → pass, attach score to record
         └─ f < τ  → GPT-4o vision fallback → re-run comparator
  └─ 8. Context Enricher
```

Important architectural change made during implementation: layout detection
(step 2) must run before quality classification and pre-processing, not after.
The layout detector needs the raw document. Pre-processing is applied only
to individual regions after they are detected. This change unblocked stage 3c.

### Fidelity formulas

Table:
```
# Current code path
f = 0.5 * grid_match + 0.2 * (1 - CER_headers) + 0.3 * (1 - CER_cells)

# Where:
# grid_match = average of row-count and column-count agreement against
# signals derived from the original crop.
#
# Note: this is the current implementation, not yet a settled final design
# for the paper. The table comparator is still under active revision.
```

Image:
```
f = 0.6 * pHash_similarity + 0.3 * sharpness_ratio + 0.1 * caption_check
```

Text:
```
f = max(0, 1 - CER)
```

### Thresholds

| Entity | Default τ |
|--------|----------|
| Table  | 0.75     |
| Image  | 0.70     |
| Text   | 0.85     |

---

## Repository structure

```
rav_idp/
  cli.py                     entry point: run pipeline on a document
  config.py                  settings via .env / environment variables
  models.py                  all Pydantic models
  pipeline.py                main orchestration: RaVIDPPipeline
  utils.py                   RapidOCR singleton, image helpers, PDF helpers

  components/
    quality_classifier.py    skew + sharpness → QualityClass enum
    layout_detector.py       Docling wrapper → DetectedRegion list
    preprocessor.py          page-level image enhancement by quality class
    region_preprocessor.py   region-level enhancement
    region_quality_classifier.py
    entity_router.py         partition regions by EntityType
    fallback_extractor.py    GPT-4o vision fallback
    context_enricher.py      caption + neighbour text attachment

    extractors/
      table.py               Docling table cells → DataFrame → TableContent
      image.py               PyMuPDF pixel extraction → ImageContent
      text.py                text + URL extraction → TextContent

    reconstructors/
      table.py               DataFrame → rendered grid image + structural signature
      image.py               pHash + sharpness + caption check
      text.py                re-OCR or PDF text stream

    comparators/
      table.py               grid/count match + OCR text agreement → FidelityResult
      image.py               pHash similarity + sharpness ratio → FidelityResult
      text.py                Levenshtein CER → FidelityResult

  evaluation/
    stage1_quality.py        PLACEHOLDER
    stage2_layout.py         IMPLEMENTED — DocLayNet, grouped IoU precision/recall/F1
    stage3a_tables.py        IMPLEMENTED — PubTabNet, TATR + OCR, GT CER + proxy TEDS
    stage3b_images.py        PLACEHOLDER
    stage3c_text.py          IMPLEMENTED — FUNSD, Tesseract OCR, CER + WER
    stage4_fidelity.py       PLACEHOLDER
    stage5_reextraction.py   PLACEHOLDER
    stage6_endtoend.py       PLACEHOLDER

  data/
    registry.py              dataset specs with download sources
    downloader.py            HTTP + HuggingFace download, archive extraction
    cli.py                   `python -m rav_idp.data fetch <dataset>`
```

---

## Dependencies

The packaging metadata should include the full runtime stack used by the
current evaluation codepaths. In particular, stage 3a requires the vision
stack below in addition to the lighter-weight pipeline dependencies.

| Package | Used in | Why missing |
|---------|---------|-------------|
| `torch >= 2.4` | stage3a (TATR) | heavy runtime dependency |
| `torchvision` | TATR (transformers dependency) | heavy runtime dependency |
| `transformers` | stage3a TATR model loading | heavy runtime dependency |
| `timm` | TATR backbone support | heavy runtime dependency |
| `rapidocr` | utils.py, comparators/table.py | OCR backend |
| `onnxruntime` | RapidOCR ONNX backend | OCR runtime |
| `scipy` | comparator peak/valley detection, correlations | scientific runtime |
| `beautifulsoup4` | proxy TEDS HTML parsing | benchmark runtime |
| `apted` | proxy TEDS tree distance | benchmark runtime |

Install order still matters in practice on clean machines because the TATR
stack pulls in large binary wheels.

```bash
uv pip install torch==2.4.1+cpu torchvision==0.19.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

uv pip install "transformers>=5.0" "rapidocr>=1.3" "onnxruntime" "timm"

uv pip install -e ".[dev]"
```

---

## Evaluation framework

Six stages, each independently runnable. The principle: end-to-end
benchmarks cannot isolate what a reconstruction-validation layer contributes.
Each stage is paired with a dataset and metric appropriate to exactly what
that component does.

| Stage | Component | Dataset | Metric | Status |
|-------|-----------|---------|--------|--------|
| 1 | Document quality classifier | SmartDoc-QA | Accuracy per class | PLACEHOLDER |
| 2 | Layout detector | DocLayNet | grouped IoU precision/recall/F1 | IMPLEMENTED |
| 3a | Table extractor | PubTabNet | row/col accuracy, GT CER, fidelity, proxy TEDS | IMPLEMENTED |
| 3b | Image extractor | ScanBank | IoU, caption match | PLACEHOLDER |
| 3c | Text extractor | FUNSD | CER, WER, fidelity-CER correlation | IMPLEMENTED |
| 4 | Fidelity scorer | PubTabNet (labelled subset) | Spearman ρ(fidelity, GT quality) | PLACEHOLDER |
| 5 | Re-extraction fallback | PubTabNet failed set | recovery rate | PLACEHOLDER |
| 6 | End-to-end | DocVQA | ANLS | PLACEHOLDER |

Run an implemented stage:
```bash
# stage 3a — requires PubTabNet archive at data/raw/pubtabnet/pubtabnet.tar.gz
python -m rav_idp.evaluation.stage3a_tables \
    --dataset-root data/raw/pubtabnet \
    --split val \
    --limit 200 \
    --output artifacts/stage3a_val.json

# stage 3c — requires FUNSD parquet at data/raw/funsd/data/
python -m rav_idp.evaluation.stage3c_text \
    --dataset-root data/raw/funsd \
    --split test \
    --output artifacts/stage3c_test.json
```

---

## Stage-to-Code Map

This is the current source-of-truth mapping from paper stage labels to the
latest code paths in this repository.

| Stage | Belongs to | Primary code | Current code status |
|-------|------------|--------------|---------------------|
| 1 | Document / region quality assessment | `rav_idp/evaluation/stage1_quality.py` plus `rav_idp/components/region_quality_classifier.py` | evaluation driver is placeholder; region classifier exists and is used by pipeline |
| 2 | Layout detection | `rav_idp/evaluation/stage2_layout.py` plus `rav_idp/components/layout_detector.py` and `rav_idp/components/page_renderer.py` | implemented and runnable |
| 3a | Table extraction on isolated table crops | `rav_idp/evaluation/stage3a_tables.py` plus `rav_idp/components/extractors/table.py`, `rav_idp/components/reconstructors/table.py`, `rav_idp/components/comparators/table.py` | implemented and runnable, but still under active revision |
| 3b | Image extraction | `rav_idp/evaluation/stage3b_images.py` plus `rav_idp/components/extractors/image.py`, `rav_idp/components/reconstructors/image.py`, `rav_idp/components/comparators/image.py` | evaluation driver placeholder; component path exists |
| 3c | Text extraction | `rav_idp/evaluation/stage3c_text.py` plus `rav_idp/components/extractors/text.py`, `rav_idp/components/reconstructors/text.py`, `rav_idp/components/comparators/text.py` | implemented and runnable |
| 4 | Fidelity reliability study | `rav_idp/evaluation/stage4_fidelity.py` | placeholder; not implemented yet |
| 5 | GPT fallback recovery study | `rav_idp/evaluation/stage5_reextraction.py` plus `rav_idp/components/fallback_extractor.py` and pipeline fallback logic in `rav_idp/pipeline.py` | fallback mechanism exists in pipeline; evaluation driver placeholder |
| 6 | End-to-end question answering | `rav_idp/evaluation/stage6_endtoend.py` plus full `rav_idp/pipeline.py` stack | evaluation driver placeholder; dataset access also pending |

Important interpretation notes:
- Stage 3a is benchmark-only code for standalone table crops. It is not the
  same as the production PDF pipeline path.
- Stage 5 depends on both a meaningful fidelity gate and an available
  `OPENAI_API_KEY`.
- Stage 6 depends on dataset access in addition to implementation work.

---

## Stage 3a: known issues and fixes applied

Full details in `docs/fix_3a.md`. Summary:

| Bug | File | Fix |
|-----|------|-----|
| Header/body row mismatch in GT counts | stage3a_tables.py | GT `row_count` aligned to extracted body rows |
| Header cells double-counted in GT text | stage3a_tables.py | header cells removed from body-cell CER stream |
| TATR extraction path unsuitable for isolated image crops when using Docling only | stage3a_tables.py | TATR primary, Docling fallback |
| Comparator count bug / false pass risk | comparators/table.py | independent OCR/count parsing and stricter crop-anchored checks |
| RapidOCR runtime instability on CPU / bundled models mismatch | utils.py | ONNX-backed RapidOCR singleton |

Current status:
- Stage 3a runs end to end on real PubTabNet crops.
- Column recovery is noticeably stronger than row recovery.
- The benchmark is now cleaner, but the table comparator and extraction path
  are still not strong enough to treat Stage 3a as a solved result.
- The TEDS value used in stage 3a is a benchmark-side proxy, not the official
  PubTabNet TEDS implementation, because the predicted side is reconstructed
  from a DataFrame and cannot encode merged cells exactly.

---

## Paper status

Written sections (in `Research papers/RaV-IDP/`):

| File | Section | Status |
|------|---------|--------|
| 08_abstract_and_section1_introduction.md | Abstract + Section 1 | Written. Placeholders: [X]%, [ρ], [Z] ANLS in abstract. |
| 07_section2_related_work.md | Section 2: Related Work | Written. |
| 05_section3_problem_formulation.md | Section 3: Problem Formulation | Written. |
| 06_section4_architecture.md | Section 4: Architecture | Written. |
| — | Section 5: Evaluation Framework | Written inline in draft. |
| — | Section 6: Results | PLACEHOLDER. Needs real numbers. |
| — | Section 7: Limitations | Written. |
| — | Section 8: Conclusion | Written. |

The full draft is `RaV-IDP_draft.html` — arXiv-style HTML with MathJax,
Georgia serif, 720px max-width. All 7 figures are embedded as relative paths
to `figures/fig*.png`.

The HTML is a readable intermediate. arXiv requires `.tex` source. Conversion
to LaTeX has not been done yet.

---

## What needs to happen next, in order

### 1. Run stage 3c on real FUNSD data

Stage 3c is already implemented and currently more stable than stage 3a.
This is the fastest path to trustworthy paper numbers.

### 2. Run stage 2 on DocLayNet

The layout benchmark is implemented and already gives strong signals about
where the architecture helps.

### 3. Improve stage 3a until the fidelity signal is usable

Real PubTabNet runs are already possible, but current results indicate that
table fidelity is not yet reliable enough to anchor Stage 4 cleanly.

### 4. Run stage 3a on larger real PubTabNet subsets

After the stage 3a debugging pass, rerun on larger validation subsets for
paper-quality numbers. PubTabNet is 11 GB; the val split has ~10,000 samples.

```bash
python -m rav_idp.data fetch pubtabnet     # downloads ~11 GB
python -m rav_idp.evaluation.stage3a_tables \
    --dataset-root data/raw/pubtabnet \
    --split val \
    --limit 500 \
    --output artifacts/stage3a_val_500.json
```

Target metrics for the paper: row_accuracy, col_accuracy,
exact_shape_accuracy, mean_cell_text_cer, mean_fidelity, pass_rate.

### 2. Run stage 3c on real FUNSD data

Stage 3c is implemented and passed end-to-end. Needs a full run to get
reportable numbers.

```bash
python -m rav_idp.data fetch funsd
python -m rav_idp.evaluation.stage3c_text \
    --dataset-root data/raw/funsd \
    --split test \
    --output artifacts/stage3c_test.json
```

Target metric: mean_CER, mean_fidelity, fidelity_cer_correlation (Pearson).

### 3. Implement stage 4: fidelity score validation

This is the central empirical claim of the paper: fidelity scores correlate
with ground-truth extraction quality. Without this, the paper has no
empirical foundation for the core claim.

Implementation plan:
- Use PubTabNet labelled val set where GT cell text is known
- For each sample, compute RaV fidelity score
- Compute actual extraction quality (CER against GT cells)
- Report Spearman ρ between fidelity and (1 - CER)

The abstract currently has `[ρ]` as a placeholder for this number.

### 4. Implement stage 5: re-extraction recovery rate

Measures how many failed extractions (fidelity < τ) are recovered by the
GPT-4o fallback. This maps to abstract placeholder `[X]%`.

Requires: OpenAI API key, a set of failed stage 3a samples.

### 5. Implement stage 2: layout detection evaluation on DocLayNet

Stage 2 code exists (293 lines). Run it against DocLayNet and report mAP.

```bash
python -m rav_idp.data fetch doclaynet
python -m rav_idp.evaluation.stage2_layout \
    --dataset-root data/raw/doclaynet \
    --output artifacts/stage2.json
```

### 6. Implement stage 6: end-to-end DocVQA

Maps to abstract placeholder `[Z] ANLS`. DocVQA requires registration at
docvqa.org. Once obtained:

```bash
python -m rav_idp.data stage --key docvqa --path /path/to/docvqa
python -m rav_idp.evaluation.stage6_endtoend \
    --dataset-root data/raw/docvqa \
    --output artifacts/stage6.json
```

### 7. Populate Section 6 with actual numbers

Once stages 3a, 3c, 4, 5 have results, fill in Section 6 in the HTML draft
and replace the three abstract placeholders.

### 8. Convert HTML draft to LaTeX for arXiv submission

arXiv requires `.tex` source. The HTML draft is a readable intermediate only.
Suggested approach: convert section-by-section, using the existing HTML as
the authoritative source for text content. MathJax equations need to be
converted back to raw LaTeX.

### 9. Fix pyproject.toml

Add the missing dependencies so `pip install -e .` covers everything:

```toml
"torch>=2.4",
"torchvision",
"transformers>=5.0",
"timm",
"rapidocr>=1.3",
"onnxruntime",
```

---

## Key design decisions on record

**Why reconstruction validates extraction:** If an extraction is faithful to
the source, rendering it back should produce something close to the original
region. The divergence between rendered and original is a grounded,
label-free quality signal.

**Why the comparator must anchor against the original crop:** Anchoring
against the extraction makes the validation circular. A wrong extraction that
reconstructs cleanly always passes. The original crop is stored at layout
detection time and never modified.

**Why per-stage evaluation rather than end-to-end only:** End-to-end
benchmarks like DocVQA mask component-level failures. A validation layer
that improves table extraction may show no improvement on DocVQA if the
question set does not exercise those tables. Per-stage evaluation is itself
a methodological contribution.

**Why GPT-4o as the fallback extractor:** The fallback operates on the
unmodified source crop with structured JSON prompts. It is entity-aware:
table prompts request headers/rows/notes; image prompts request
type/description/key_data_points; text prompts request plain transcription.
The RaV loop repeats on fallback output to provide a second fidelity
measurement.
