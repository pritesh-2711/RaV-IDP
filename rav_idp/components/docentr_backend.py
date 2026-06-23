"""Offline DocEnTR subprocess backend for binarization benchmarks."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import RestorationBackendResult, RestorationPlan, TransformKind
from ..utils import image_bytes_to_ndarray
from .page_restorer import restore_page


class DocEnTRBackendError(RuntimeError):
    """DocEnTR could not be configured or did not produce a valid result."""


@dataclass(frozen=True)
class DocEnTRConfig:
    repo_dir: Path
    python_bin: str
    weights_path: Path
    model_size: str = "base"
    split_size: int = 256
    patch_size: int = 8
    threshold: float = 0.5
    batch_size: int = 1
    device: str | None = None
    timeout_seconds: int = 600
    runner_path: Path | None = None

    @property
    def resolved_runner_path(self) -> Path:
        default = Path(__file__).resolve().parents[2] / "third_party" / "docentr_single_image.py"
        return (self.runner_path or default).expanduser().resolve()


class DocEnTRSubprocessBackend:
    """Run tiled single-image DocEnTR inference in its old, isolated environment."""

    name = "docentr-official-model-subprocess-v1"

    def __init__(self, config: DocEnTRConfig) -> None:
        self.config = config

    def _validate(self) -> tuple[Path, str, Path]:
        repo_dir = self.config.repo_dir.expanduser().resolve()
        if not (repo_dir / "models" / "binae.py").is_file():
            raise DocEnTRBackendError(f"DocEnTR models/binae.py not found under {repo_dir}")
        python_bin = self.config.python_bin
        if not Path(python_bin).is_file() and shutil.which(python_bin) is None:
            raise DocEnTRBackendError(f"DocEnTR Python executable not found: {python_bin}")
        weights = self.config.weights_path.expanduser().resolve()
        if not weights.is_file():
            raise DocEnTRBackendError(f"DocEnTR weights not found: {weights}")
        runner = self.config.resolved_runner_path
        if not runner.is_file():
            raise DocEnTRBackendError(f"DocEnTR inference runner not found: {runner}")
        if self.config.split_size % self.config.patch_size:
            raise DocEnTRBackendError("DocEnTR split_size must be divisible by patch_size.")
        if not 0.0 <= self.config.threshold <= 1.0:
            raise DocEnTRBackendError("DocEnTR threshold must be within [0, 1].")
        return repo_dir, python_bin, runner

    def restore(self, source_page_v1: bytes, plan: RestorationPlan) -> RestorationBackendResult:
        repo_dir, python_bin, runner = self._validate()
        geometry_plan = plan.model_copy(
            update={
                "operations": [
                    operation
                    for operation in plan.operations
                    if operation.kind in {TransformKind.ROTATE_ORIENTATION, TransformKind.DESKEW}
                ]
            }
        )
        geometry_bytes, mapping = restore_page(source_page_v1, geometry_plan)
        input_image = image_bytes_to_ndarray(geometry_bytes)
        if input_image is None:
            raise DocEnTRBackendError("Failed to decode the geometry-normalized DocEnTR input.")
        input_height, input_width = input_image.shape[:2]

        with tempfile.TemporaryDirectory(prefix="rav-docentr-") as temporary:
            work_dir = Path(temporary)
            input_path = work_dir / "input.png"
            output_path = work_dir / "output.png"
            input_path.write_bytes(geometry_bytes)
            command = [
                python_bin,
                str(runner),
                "--repo-dir",
                str(repo_dir),
                "--weights",
                str(self.config.weights_path.expanduser().resolve()),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--model-size",
                self.config.model_size,
                "--split-size",
                str(self.config.split_size),
                "--patch-size",
                str(self.config.patch_size),
                "--threshold",
                str(self.config.threshold),
                "--batch-size",
                str(self.config.batch_size),
            ]
            if self.config.device:
                command.extend(["--device", self.config.device])
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise DocEnTRBackendError(
                    f"DocEnTR inference exceeded {self.config.timeout_seconds} seconds."
                ) from error
            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout or "no diagnostic output")[-2000:]
                raise DocEnTRBackendError(
                    f"DocEnTR inference failed with exit code {completed.returncode}: {diagnostic}"
                )
            if not output_path.is_file():
                raise DocEnTRBackendError("DocEnTR inference completed without producing output.png.")
            output_bytes = output_path.read_bytes()

        restored = image_bytes_to_ndarray(output_bytes)
        if restored is None:
            raise DocEnTRBackendError("DocEnTR produced an unreadable image.")
        output_height, output_width = restored.shape[:2]
        scale = np.array(
            [
                [output_width / input_width, 0.0, 0.0],
                [0.0, output_height / input_height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        geometry_matrix = np.asarray(mapping.matrix, dtype=np.float64).reshape(3, 3)
        mapping = mapping.model_copy(
            update={
                "matrix": (scale @ geometry_matrix).reshape(-1).tolist(),
                "target_width": output_width,
                "target_height": output_height,
            }
        )
        return RestorationBackendResult(
            image_bytes=output_bytes,
            mapping=mapping,
            backend_name=self.name,
            task="binarization",
            warnings=["docentr_is_binarization_benchmark_backend"],
        )
