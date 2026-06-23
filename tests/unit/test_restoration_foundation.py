from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from rav_idp.components.entity_refiner import refine_entity_input
from rav_idp.components.docres_backend import (
    DocResBackendError,
    DocResConfig,
    DocResSubprocessBackend,
    select_docres_task,
)
from rav_idp.components.docentr_backend import (
    DocEnTRConfig,
    DocEnTRSubprocessBackend,
)
from rav_idp.components.evidence_chain import build_page_evidence
from rav_idp.components.page_restorer import restore_page
from rav_idp.components.restoration_integrity import map_points
from rav_idp.components.restoration_planner import plan_page_restoration
from rav_idp.evaluation.quality_benchmark import run_quality_benchmark
from rav_idp.models import (
    AcquisitionMode,
    BoundingBox,
    DetectedRegion,
    EntityType,
    InputKind,
    PageRecord,
    QualityAssessment,
    QualityClass,
    QualityProfile,
    RestorationPlan,
    TransformKind,
    TransformSpec,
)


def _encode(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _page_image() -> np.ndarray:
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    cv2.putText(image, "VERSIONED EVIDENCE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.line(image, (10, 120), (240, 120), (0, 0, 0), 2)
    return image


def _page(profile: QualityProfile | None = None) -> PageRecord:
    raw = _encode(_page_image())
    assessment = (
        QualityAssessment(scope="page", page_index=0, profile=profile)
        if profile is not None
        else None
    )
    return PageRecord(
        page_index=0,
        quality_class=QualityClass.SCANNED_DEGRADED,
        raw_image=raw,
        processed_image=raw,
        quality_assessment=assessment,
    )


def test_disabled_evidence_chain_keeps_v2_byte_identical() -> None:
    page = _page(
        QualityProfile(
            input_kind=InputKind.RASTER_PDF,
            acquisition_mode=AcquisitionMode.SCANNED,
            blur=0.8,
            metric_version="test",
        )
    )

    evidence = build_page_evidence([page], enable_restoration=False)[0]

    assert evidence.working_page_v2 == evidence.source_page_v1
    assert evidence.candidate_page_v2 is None
    assert evidence.plan.operations == []
    assert evidence.mapping_v1_to_v2.matrix == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert evidence.integrity.passed


def test_planner_orders_geometry_before_photometric_transforms() -> None:
    page = _page(
        QualityProfile(
            input_kind=InputKind.RASTER_PDF,
            acquisition_mode=AcquisitionMode.SCANNED,
            rotation_required_degrees=270,
            orientation_confidence=0.8,
            skew=0.4,
            skew_angle_degrees=4.0,
            uneven_illumination=0.7,
            low_contrast=0.6,
            blur=0.8,
            metric_version="test",
        )
    )

    plan = plan_page_restoration(page)

    assert [operation.kind for operation in plan.operations] == [
        TransformKind.ROTATE_ORIENTATION,
        TransformKind.DESKEW,
        TransformKind.ILLUMINATION_NORMALIZATION,
        TransformKind.CLAHE,
        TransformKind.UNSHARP_MASK,
    ]


def test_enabled_chain_preserves_candidate_even_if_integrity_rejects_it() -> None:
    page = _page(
        QualityProfile(
            input_kind=InputKind.RASTER_PDF,
            acquisition_mode=AcquisitionMode.SCANNED,
            low_contrast=0.9,
            uneven_illumination=0.9,
            metric_version="test",
        )
    )

    evidence = build_page_evidence([page], enable_restoration=True)[0]

    assert evidence.candidate_page_v2 is not None
    if not evidence.integrity.passed:
        assert evidence.working_page_v2 == evidence.source_page_v1


def test_rotation_emits_mapping_and_swaps_dimensions() -> None:
    source = _encode(_page_image())
    plan = RestorationPlan(
        page_index=0,
        planner_version="test",
        operations=[
            TransformSpec(
                kind=TransformKind.ROTATE_ORIENTATION,
                parameters={"degrees_clockwise": 90},
                reason="test",
            )
        ],
    )

    _, mapping = restore_page(source, plan)

    assert (mapping.target_width, mapping.target_height) == (180, 260)
    assert map_points([(0.0, 0.0), (259.0, 179.0)], mapping) == [
        (179.0, 0.0),
        (0.0, 259.0),
    ]


def test_image_entity_refinement_is_noop() -> None:
    crop = _encode(_page_image())
    region = DetectedRegion(
        region_id="0_0",
        entity_type=EntityType.IMAGE,
        bbox=BoundingBox(x0=0, y0=0, x1=260, y1=180, page=0),
        original_crop=crop,
        raw_docling_record={},
        page_index=0,
    )

    evidence = refine_entity_input(region, crop)

    assert evidence.entity_input_v3 == crop
    assert evidence.operations == []


def test_benchmark_omits_source_names_and_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "private-name.png").write_bytes(_encode(_page_image()))

    summary = run_quality_benchmark(input_dir, tmp_path / "report")
    report_text = (tmp_path / "report" / "quality_benchmark_rows.csv").read_text(encoding="utf-8")
    summary_text = (tmp_path / "report" / "quality_benchmark_summary.json").read_text(encoding="utf-8")

    assert summary["observation_count"] == 18
    assert "private-name" not in report_text
    assert str(input_dir) not in report_text
    assert "private-name" not in summary_text
    json.loads(summary_text)


def _fake_docres_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "DocRes"
    (repo / "checkpoints").mkdir(parents=True)
    (repo / "checkpoints" / "docres.pkl").write_bytes(b"weights")
    (repo / "data" / "MBD" / "checkpoint").mkdir(parents=True)
    (repo / "data" / "MBD" / "checkpoint" / "mbd.pkl").write_bytes(b"weights")
    (repo / "inference.py").write_text(
        """\
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
parser.add_argument('--im_path')
parser.add_argument('--task')
parser.add_argument('--out_folder')
parser.add_argument('--save_dtsprompt')
args = parser.parse_args()
source = Path(args.im_path)
target = Path(args.out_folder) / f'{source.stem}_{args.task}{source.suffix}'
shutil.copyfile(source, target)
""",
        encoding="utf-8",
    )
    return repo


def test_docres_subprocess_backend_uses_official_output_contract(tmp_path: Path) -> None:
    repo = _fake_docres_repo(tmp_path)
    plan = RestorationPlan(
        page_index=0,
        planner_version="test",
        operations=[
            TransformSpec(kind=TransformKind.CLAHE, parameters={}, reason="test")
        ],
    )
    backend = DocResSubprocessBackend(
        DocResConfig(repo_dir=repo, python_bin=sys.executable, task="auto")
    )

    result = backend.restore(_encode(_page_image()), plan)

    assert result.task == "appearance"
    assert result.backend_name == "opencv-plus-docres-official-v1"
    assert result.mapping is not None
    assert result.image_bytes
    assert "docres_mode:opencv_then_docres" in result.warnings
    restored = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert not np.array_equal(restored, _page_image())


def test_docres_only_mode_skips_opencv_photometric_operations(tmp_path: Path) -> None:
    repo = _fake_docres_repo(tmp_path)
    source = _encode(_page_image())
    plan = RestorationPlan(
        page_index=0,
        planner_version="test",
        operations=[TransformSpec(kind=TransformKind.CLAHE, parameters={}, reason="test")],
    )
    backend = DocResSubprocessBackend(
        DocResConfig(
            repo_dir=repo,
            python_bin=sys.executable,
            task="appearance",
            mode="docres_only",
        )
    )

    result = backend.restore(source, plan)

    expected = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    restored = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(restored, expected)
    assert result.backend_name == "docres-official-subprocess-v1"
    assert "docres_mode:docres_only" in result.warnings


def test_docres_bounds_model_input_and_restores_page_dimensions(tmp_path: Path) -> None:
    repo = _fake_docres_repo(tmp_path)
    large = np.full((600, 1200, 3), 255, dtype=np.uint8)
    cv2.putText(large, "LARGE PAGE", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    backend = DocResSubprocessBackend(
        DocResConfig(
            repo_dir=repo,
            python_bin=sys.executable,
            task="deblurring",
            mode="docres_only",
            max_input_dimension=512,
        )
    )

    result = backend.restore(
        _encode(large),
        RestorationPlan(page_index=0, planner_version="test"),
    )

    restored = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert restored.shape[:2] == large.shape[:2]
    assert any(warning.startswith("docres_input_downscaled:1200x600->512x256") for warning in result.warnings)
    assert "docres_output_restored_to_page_dimensions" in result.warnings


def test_docres_geometric_task_is_rejected_without_dense_mapping(tmp_path: Path) -> None:
    repo = _fake_docres_repo(tmp_path)
    plan = RestorationPlan(page_index=0, planner_version="test")
    backend = DocResSubprocessBackend(
        DocResConfig(repo_dir=repo, python_bin=sys.executable, task="dewarping")
    )

    result = backend.restore(_encode(_page_image()), plan)

    assert result.mapping is None
    assert "docres_geometric_mapping_unavailable" in result.warnings


def test_docres_reports_missing_model_before_subprocess(tmp_path: Path) -> None:
    repo = _fake_docres_repo(tmp_path)
    (repo / "checkpoints" / "docres.pkl").unlink()
    backend = DocResSubprocessBackend(
        DocResConfig(repo_dir=repo, python_bin=sys.executable, task="appearance")
    )

    try:
        backend.restore(
            _encode(_page_image()),
            RestorationPlan(page_index=0, planner_version="test"),
        )
    except DocResBackendError as error:
        assert "model weights not found" in str(error)
    else:
        raise AssertionError("Expected missing DocRes weights to fail before inference.")


def test_docres_auto_task_prefers_deshadowing_for_illumination() -> None:
    plan = RestorationPlan(
        page_index=0,
        planner_version="test",
        operations=[
            TransformSpec(kind=TransformKind.UNSHARP_MASK, parameters={}, reason="blur"),
            TransformSpec(
                kind=TransformKind.ILLUMINATION_NORMALIZATION,
                parameters={},
                reason="illumination",
            ),
        ],
    )

    assert select_docres_task(plan) == "deshadowing"


def test_docentr_subprocess_backend_preserves_geometry_contract(tmp_path: Path) -> None:
    repo = tmp_path / "DocEnTR"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "binae.py").write_text("# fake", encoding="utf-8")
    weights = repo / "model.pt"
    weights.write_bytes(b"weights")
    runner = tmp_path / "fake_docentr.py"
    runner.write_text(
        """\
import argparse
import shutil

parser = argparse.ArgumentParser()
for name in ('repo-dir', 'weights', 'input', 'output', 'model-size', 'split-size',
             'patch-size', 'threshold', 'batch-size'):
    parser.add_argument('--' + name)
parser.add_argument('--device')
args = parser.parse_args()
shutil.copyfile(args.input, args.output)
""",
        encoding="utf-8",
    )
    backend = DocEnTRSubprocessBackend(
        DocEnTRConfig(
            repo_dir=repo,
            python_bin=sys.executable,
            weights_path=weights,
            runner_path=runner,
        )
    )

    result = backend.restore(
        _encode(_page_image()),
        RestorationPlan(page_index=0, planner_version="test"),
    )

    assert result.backend_name == "docentr-official-model-subprocess-v1"
    assert result.task == "binarization"
    assert result.mapping is not None
    assert result.mapping.target_width == 260
    assert "docentr_is_binarization_benchmark_backend" in result.warnings
