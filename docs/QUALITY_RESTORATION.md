# Quality and restoration experiments

This code is intentionally separate from the production extraction pipeline. The existing
`RaVIDPPipeline` remains assessment-only. Experimental restoration runs only through
`rav_idp.quality_cli restore --enable-restoration`.

## Phase coverage

- Phase 1: deterministic page/region profiler.
- Phase 2: controlled synthetic degradation benchmark and monotonicity report.
- Phase 3: deterministic planner and conservative page-level transforms.
- Phase 4: homogeneous v1-to-v2 mappings and restoration-integrity checks.
- Phase 5: entity-specific v3 refinement contract; image regions remain no-op.
- Phase 6: explicit `EvidenceUsage` and downstream evaluation-record contracts.
- Phase 7: learned scoring remains pending benchmark justification and labelled data.
- Phase 8: DocRes and DocEnTR are supported through isolated subprocess backends. Their
  official repositories and weights are not bundled.
- Phase 9: named ablation variants and result aggregation contracts are ready.

Later phases are foundations, not validated production features. Threshold calibration,
dataset evaluation, model integration, and extraction-gain measurements must happen before
they can be enabled in the main pipeline.

## Commands

From `experiment/`, activate the repository environment and use runtime paths:

```bash
source ../.venv/bin/activate

python -m rav_idp.quality_cli profile \
  "/path/to/document.pdf" \
  --output artifacts/quality-check/profile.json

python -m rav_idp.quality_cli restore \
  "/path/to/document.pdf" \
  --output-dir artifacts/evidence-identity

python -m rav_idp.quality_cli restore \
  "/path/to/document.pdf" \
  --output-dir artifacts/evidence-experimental \
  --enable-restoration

python -m rav_idp.quality_cli benchmark \
  "/path/to/validation-folder" \
  --output-dir artifacts/quality-benchmark \
  --max-documents 20 \
  --max-pages 3
```

## DocRes backend

DocRes runs in a separate environment because its official dependency versions conflict with
the main project. RaV-IDP does not download or vendor its repository or weights.

Expected external layout:

```text
/path/to/DocRes/inference.py
/path/to/DocRes/checkpoints/docres.pkl
/path/to/DocRes/data/MBD/checkpoint/mbd.pkl
```

The MBD weights are needed only for `dewarping` and `end2end`. Configure the official clone
and isolated Python at runtime:

```bash
export RAV_DOCRES_REPO="/path/to/DocRes"
export RAV_DOCRES_PYTHON="/path/to/docres/environment/bin/python"

python -m rav_idp.quality_cli restore \
  "/path/to/document.pdf" \
  --output-dir artifacts/evidence-docres \
  --enable-restoration \
  --backend docres \
  --docres-mode opencv_then_docres \
  --docres-max-dimension 1024 \
  --docres-task auto
```

Available tasks are `appearance`, `binarization`, `deblurring`, `deshadowing`, `dewarping`,
and `end2end`. Auto-routing selects one learned task from the deterministic plan. Run explicit
tasks as separate ablations when several degradations are present.

The default `opencv_then_docres` mode executes the planned OpenCV operations first and passes
that result into DocRes. Use `--docres-mode docres_only` for the ablation where only orientation/
affine deskew precede DocRes. Both modes use the same integrity gate and retain their candidate.
DocRes inference is bounded to a 1024-pixel maximum dimension by default to fit common 8 GB GPUs;
the learned output is restored to page dimensions before integrity evaluation. Lower this value
to 768 if memory remains insufficient, or set it to `0` only on hardware that can process the
full rendered page.

Gross orientation and minor affine deskew are applied before photometric DocRes tasks so an
explicit coordinate mapping remains available. DocRes `dewarping` and `end2end` use a dense,
non-affine remap that the official CLI does not export. Those results are therefore generated
but rejected as canonical v2 evidence until a dense mapping adapter is implemented.

Notebook usage:

```python
from pathlib import Path
from rav_idp.components.docres_backend import DocResConfig, DocResSubprocessBackend
from rav_idp.components.evidence_chain import build_page_evidence

docres = DocResSubprocessBackend(
    DocResConfig(
        repo_dir=Path("/path/to/DocRes"),
        python_bin="/path/to/docres/environment/bin/python",
        task="auto",  # or an explicit DocRes task
        mode="opencv_then_docres",
        max_input_dimension=1024,
    )
)

restored_evidence = build_page_evidence(
    pages,
    enable_restoration=True,
    backend=docres,
)
```

The adapter validates the official script and weights, uses argument-list subprocess execution,
isolates temporary inputs/outputs, serializes access to DocRes's shared working directory, checks
the generated image, and then applies the same restoration-integrity gate as OpenCV.

Every enabled run preserves `candidate_page_v2` for inspection. If integrity passes, it also
becomes canonical `working_page_v2`; otherwise `working_page_v2` falls back to the immutable v1
bytes while the rejected candidate remains available for analysis.

## DocEnTR benchmark backend

DocEnTR is integrated as an optional enhancement/binarization benchmark, not a general page
restorer. It uses the official `BinModel` and weights through a standalone tiled runner. Tiling
preserves the page dimensions instead of resizing the entire page to the model's 256-pixel input.

Expected external layout:

```text
/path/to/DocEnTR/models/binae.py
/path/to/DocEnTR/weights/model_8_2018_base.pt
```

Run it with its isolated environment:

```bash
export RAV_DOCENTR_REPO="/path/to/DocEnTR"
export RAV_DOCENTR_PYTHON="/path/to/docentr/environment/bin/python"
export RAV_DOCENTR_WEIGHTS="/path/to/model_8_2018_base.pt"

python -m rav_idp.quality_cli restore \
  "/path/to/document.pdf" \
  --output-dir artifacts/evidence-docentr \
  --enable-restoration \
  --backend docentr \
  --docentr-model-size base \
  --docentr-patch-size 8
```

The model-size and patch-size arguments must match the selected weights. DocEnTR produces binary
evidence, so it should be compared against both OpenCV and DocRes using OCR CER and downstream
fidelity rather than selected solely from visual appearance.

The profile, benchmark, and restoration manifests omit source filenames and paths. Inputs
are identified by SHA-256. The generated page image artifacts contain document pixels and
must still be handled as sensitive data.

Without `--enable-restoration`, `working_page_v2` is byte-identical to `source_page_v1` and
the coordinate mapping is identity. With the flag, a failed integrity check rejects v2 and
falls back to the original v1 bytes.
