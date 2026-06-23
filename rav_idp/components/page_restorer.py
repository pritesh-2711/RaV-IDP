"""Experimental page restorer with explicit v1-to-v2 coordinate mappings."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import CoordinateMapping, RestorationPlan, TransformKind
from ..utils import image_bytes_to_ndarray, ndarray_to_png_bytes


def identity_mapping(width: int, height: int) -> CoordinateMapping:
    return CoordinateMapping(
        matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        source_width=width,
        source_height=height,
        target_width=width,
        target_height=height,
    )


def _apply_luminance(image: np.ndarray, transform) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = transform(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _rotate_orientation(image: np.ndarray, degrees: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    degrees %= 360
    if degrees == 90:
        matrix = np.array([[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), matrix
    if degrees == 180:
        matrix = np.array([[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
        return cv2.rotate(image, cv2.ROTATE_180), matrix
    if degrees == 270:
        matrix = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0], [0.0, 0.0, 1.0]])
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), matrix
    return image, np.eye(3, dtype=np.float64)


def restore_page(source_page_v1: bytes, plan: RestorationPlan) -> tuple[bytes, CoordinateMapping]:
    """Execute a plan without mutating source bytes; return v2 and source-to-v2 mapping."""

    image = image_bytes_to_ndarray(source_page_v1)
    if image is None:
        raise ValueError("Unable to decode source_page_v1.")
    source_height, source_width = image.shape[:2]
    composed = np.eye(3, dtype=np.float64)

    for operation in plan.operations:
        if operation.kind == TransformKind.ROTATE_ORIENTATION:
            image, matrix = _rotate_orientation(
                image,
                int(operation.parameters.get("degrees_clockwise", 0)),
            )
            composed = matrix @ composed
        elif operation.kind == TransformKind.DESKEW:
            height, width = image.shape[:2]
            angle = -float(operation.parameters.get("angle_degrees", 0.0))
            affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image,
                affine,
                (width, height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
            composed = matrix @ composed
        elif operation.kind == TransformKind.ILLUMINATION_NORMALIZATION:
            def normalize(channel: np.ndarray) -> np.ndarray:
                background = cv2.GaussianBlur(channel, (0, 0), sigmaX=31, sigmaY=31)
                normalized = channel.astype(np.float32) - background.astype(np.float32) + 230.0
                return np.clip(normalized, 0, 255).astype(np.uint8)

            image = _apply_luminance(image, normalize)
        elif operation.kind == TransformKind.CLAHE:
            clip_limit = float(operation.parameters.get("clip_limit", 2.0))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            image = _apply_luminance(image, clahe.apply)
        elif operation.kind == TransformKind.DENOISE:
            image = cv2.bilateralFilter(image, 5, 35, 35)
        elif operation.kind == TransformKind.UNSHARP_MASK:
            amount = float(operation.parameters.get("amount", 0.35))
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            image = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        elif operation.kind == TransformKind.ADAPTIVE_BINARIZATION:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )
            image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    target_height, target_width = image.shape[:2]
    mapping = identity_mapping(source_width, source_height).model_copy(
        update={
            "matrix": composed.reshape(-1).tolist(),
            "target_width": target_width,
            "target_height": target_height,
        }
    )
    return ndarray_to_png_bytes(image), mapping
