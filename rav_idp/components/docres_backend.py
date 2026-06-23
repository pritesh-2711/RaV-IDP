"""Offline DocRes subprocess backend isolated from the RaV-IDP Python environment."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ..models import (
    RestorationBackendResult,
    RestorationPlan,
    TransformKind,
)
from ..utils import image_bytes_to_ndarray, ndarray_to_png_bytes
from .page_restorer import restore_page


DocResTask = Literal[
    "auto",
    "dewarping",
    "deshadowing",
    "appearance",
    "deblurring",
    "binarization",
    "end2end",
]
DocResMode = Literal["opencv_then_docres", "docres_only"]

VALID_DOCRES_TASKS = {
    "dewarping",
    "deshadowing",
    "appearance",
    "deblurring",
    "binarization",
    "end2end",
}
GEOMETRIC_DOCRES_TASKS = {"dewarping", "end2end"}


class DocResBackendError(RuntimeError):
    """DocRes could not be configured or did not produce a valid result."""


@dataclass(frozen=True)
class DocResConfig:
    repo_dir: Path
    python_bin: str
    model_path: Path | None = None
    task: DocResTask = "auto"
    mode: DocResMode = "opencv_then_docres"
    max_input_dimension: int | None = 1024
    timeout_seconds: int = 600

    @property
    def resolved_model_path(self) -> Path:
        return (self.model_path or self.repo_dir / "checkpoints" / "docres.pkl").expanduser().resolve()

def select_docres_task(plan: RestorationPlan, requested: DocResTask = "auto") -> str:
    if requested != "auto":
        if requested not in VALID_DOCRES_TASKS:
            raise ValueError(f"Unsupported DocRes task: {requested}")
        return requested
    kinds = {operation.kind for operation in plan.operations}
    if TransformKind.ILLUMINATION_NORMALIZATION in kinds:
        return "deshadowing"
    if TransformKind.UNSHARP_MASK in kinds:
        return "deblurring"
    if TransformKind.CLAHE in kinds:
        return "appearance"
    if TransformKind.ADAPTIVE_BINARIZATION in kinds:
        return "binarization"
    if TransformKind.DESKEW in kinds:
        return "dewarping"
    return "appearance"


@contextmanager
def _repository_lock(repo_dir: Path):
    lock_path = repo_dir / ".rav_docres.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DocResSubprocessBackend:
    """Run the official DocRes inference script in its own environment."""

    def __init__(self, config: DocResConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        if self.config.mode == "opencv_then_docres":
            return "opencv-plus-docres-official-v1"
        return "docres-official-subprocess-v1"

    def _validate(self, task: str) -> tuple[Path, str]:
        repo_dir = self.config.repo_dir.expanduser().resolve()
        inference_script = repo_dir / "inference.py"
        if not inference_script.is_file():
            raise DocResBackendError(f"DocRes inference.py not found under {repo_dir}")
        python_bin = self.config.python_bin
        if not Path(python_bin).is_file() and shutil.which(python_bin) is None:
            raise DocResBackendError(f"DocRes Python executable not found: {python_bin}")
        if not self.config.resolved_model_path.is_file():
            raise DocResBackendError(f"DocRes model weights not found: {self.config.resolved_model_path}")
        mbd_model_path = repo_dir / "data" / "MBD" / "checkpoint" / "mbd.pkl"
        if task in GEOMETRIC_DOCRES_TASKS and not mbd_model_path.is_file():
            raise DocResBackendError(
                f"DocRes MBD weights required for {task}: {mbd_model_path}"
            )
        return repo_dir, python_bin

    def restore(self, source_page_v1: bytes, plan: RestorationPlan) -> RestorationBackendResult:
        task = select_docres_task(plan, self.config.task)
        repo_dir, python_bin = self._validate(task)

        if self.config.mode == "opencv_then_docres":
            preprocessing_plan = plan
        elif self.config.mode == "docres_only":
            preprocessing_plan = plan.model_copy(
                update={
                    "operations": [
                        operation
                        for operation in plan.operations
                        if operation.kind in {TransformKind.ROTATE_ORIENTATION, TransformKind.DESKEW}
                    ]
                }
            )
        else:
            raise DocResBackendError(f"Unsupported DocRes integration mode: {self.config.mode}")

        preprocessed_bytes, preprocessing_mapping = restore_page(source_page_v1, preprocessing_plan)
        preprocessed_image = image_bytes_to_ndarray(preprocessed_bytes)
        if preprocessed_image is None:
            raise DocResBackendError("Failed to decode the OpenCV-preprocessed DocRes input.")
        page_height, page_width = preprocessed_image.shape[:2]
        model_input = preprocessed_image
        warnings: list[str] = [f"docres_mode:{self.config.mode}"]
        maximum_dimension = self.config.max_input_dimension
        if maximum_dimension is not None:
            if maximum_dimension < 256:
                raise DocResBackendError("DocRes max_input_dimension must be at least 256 or None.")
            if max(page_width, page_height) > maximum_dimension:
                scale = maximum_dimension / max(page_width, page_height)
                model_width = max(8, round(page_width * scale / 8) * 8)
                model_height = max(8, round(page_height * scale / 8) * 8)
                model_input = cv2.resize(
                    preprocessed_image,
                    (model_width, model_height),
                    interpolation=cv2.INTER_AREA,
                )
                warnings.append(
                    f"docres_input_downscaled:{page_width}x{page_height}->{model_width}x{model_height}"
                )
        model_input_bytes = ndarray_to_png_bytes(model_input)

        with tempfile.TemporaryDirectory(prefix="rav-docres-") as temporary:
            work_dir = Path(temporary)
            input_path = work_dir / "input.png"
            output_dir = work_dir / "output"
            output_dir.mkdir()
            input_path.write_bytes(model_input_bytes)
            command = [
                python_bin,
                "inference.py",
                "--model_path",
                str(self.config.resolved_model_path),
                "--im_path",
                str(input_path),
                "--task",
                task,
                "--out_folder",
                str(output_dir),
                "--save_dtsprompt",
                "0",
            ]
            try:
                with _repository_lock(repo_dir):
                    (repo_dir / "restorted").mkdir(exist_ok=True)
                    environment = os.environ.copy()
                    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    completed = subprocess.run(
                        command,
                        cwd=repo_dir,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.config.timeout_seconds,
                        env=environment,
                    )
            except subprocess.TimeoutExpired as error:
                raise DocResBackendError(
                    f"DocRes inference exceeded {self.config.timeout_seconds} seconds."
                ) from error
            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout or "no diagnostic output")[-2000:]
                raise DocResBackendError(
                    f"DocRes inference failed with exit code {completed.returncode}: {diagnostic}"
                )

            output_path = output_dir / f"input_{task}.png"
            if not output_path.is_file():
                produced = sorted(path.name for path in output_dir.iterdir())
                raise DocResBackendError(
                    f"DocRes did not create {output_path.name}; produced files: {produced}"
                )
            output_bytes = output_path.read_bytes()

        restored = image_bytes_to_ndarray(output_bytes)
        if restored is None:
            raise DocResBackendError("DocRes produced an unreadable image.")
        output_height, output_width = restored.shape[:2]
        if (output_width, output_height) != (page_width, page_height):
            interpolation = cv2.INTER_NEAREST if task == "binarization" else cv2.INTER_CUBIC
            restored = cv2.resize(
                restored,
                (page_width, page_height),
                interpolation=interpolation,
            )
            output_bytes = ndarray_to_png_bytes(restored)
            output_width, output_height = page_width, page_height
            warnings.append("docres_output_restored_to_page_dimensions")
        if task in GEOMETRIC_DOCRES_TASKS:
            mapping = None
            warnings.append("docres_geometric_mapping_unavailable")
        else:
            preprocessing_matrix = np.asarray(
                preprocessing_mapping.matrix,
                dtype=np.float64,
            ).reshape(3, 3)
            mapping = preprocessing_mapping.model_copy(
                update={
                    "matrix": preprocessing_matrix.reshape(-1).tolist(),
                    "target_width": output_width,
                    "target_height": output_height,
                }
            )
        return RestorationBackendResult(
            image_bytes=output_bytes,
            mapping=mapping,
            backend_name=self.name,
            task=task,
            warnings=warnings,
        )
