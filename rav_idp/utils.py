"""Shared helper utilities."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import fitz
import numpy as np
from PIL import Image

from .config import DEFAULT_DPI
from .models import BoundingBox

if TYPE_CHECKING:
    from rapidocr import RapidOCR as _RapidOCRType

# ---------------------------------------------------------------------------
# RapidOCR singleton (torch backend) — loaded once, reused across components
# ---------------------------------------------------------------------------

_RAPIDOCR_INSTANCE: "_RapidOCRType | None" = None


def _rapidocr_model_dir() -> Path:
    """Return the directory where the RapidOCR torch model files live."""
    import rapidocr as _pkg
    return Path(_pkg.__file__).parent / "models"


def get_rapidocr() -> "_RapidOCRType":
    """Return a cached RapidOCR instance using the torch backend."""
    global _RAPIDOCR_INSTANCE
    if _RAPIDOCR_INSTANCE is None:
        from rapidocr import RapidOCR
        from rapidocr.inference_engine.base import EngineType
        model_dir = _rapidocr_model_dir()
        _RAPIDOCR_INSTANCE = RapidOCR(params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
            "Det.model_path": str(model_dir / "ch_PP-OCRv4_det_mobile.pth"),
            "Cls.model_path": str(model_dir / "ch_ptocr_mobile_v2.0_cls_mobile.pth"),
            "Rec.model_path": str(model_dir / "ch_PP-OCRv4_rec_mobile.pth"),
            "Rec.rec_keys_path": str(model_dir / "ppocr_keys_v1.txt"),
            "EngineConfig.torch.use_cuda": True,
        })
    return _RAPIDOCR_INSTANCE


def rapidocr_image_to_text(image: "Image.Image", min_height: int = 200) -> str:
    """Run RapidOCR on *image* and return all recognised text as a single string.

    Small images are upscaled to *min_height* pixels tall before inference so
    that the detector can find text regions.  The model is initialised once and
    reused across calls.
    """
    w, h = image.size
    if h < min_height:
        scale = min_height / h
        image = image.resize((max(1, int(w * scale)), min_height), Image.LANCZOS)
    image = image.convert("RGB")
    out = get_rapidocr()(np.array(image))
    return " ".join(out.txts) if out.txts else ""


def image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_bytes_to_ndarray(image_bytes: bytes, grayscale: bool = False) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(arr, flag)


def ndarray_to_png_bytes(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image as PNG.")
    return encoded.tobytes()


def render_page_to_png(page: fitz.Page, dpi: int = DEFAULT_DPI) -> bytes:
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pix.tobytes("png")


def has_pdf_text_layer(page: fitz.Page) -> bool:
    return bool(page.get_text("text").strip())


def bbox_centroid(bbox: BoundingBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def centroid_distance(a: BoundingBox, b: BoundingBox) -> float:
    ax, ay = bbox_centroid(a)
    bx, by = bbox_centroid(b)
    return math.hypot(ax - bx, ay - by)


def crop_image_bytes(image_bytes: bytes, bbox: BoundingBox) -> bytes:
    image = image_bytes_to_pil(image_bytes).convert("RGB")
    width, height = image.size
    left = max(0, min(int(round(bbox.x0)), width))
    top = max(0, min(int(round(bbox.y0)), height))
    right = max(left, min(int(round(bbox.x1)), width))
    bottom = max(top, min(int(round(bbox.y1)), height))
    if right <= left or bottom <= top:
        return b""
    crop = image.crop((left, top, right, bottom))
    return pil_to_png_bytes(crop)


def docling_bbox_to_pixel_bbox(
    bbox: object,
    page_height_points: float,
    page_index: int,
    dpi: int = DEFAULT_DPI,
    scale: float | None = None,
) -> BoundingBox:
    scale = scale if scale is not None else (dpi / 72.0)
    x0 = float(getattr(bbox, "l", 0.0)) * scale
    x1 = float(getattr(bbox, "r", 0.0)) * scale
    y0 = (page_height_points - float(getattr(bbox, "t", 0.0))) * scale
    y1 = (page_height_points - float(getattr(bbox, "b", 0.0))) * scale
    top, bottom = sorted((y0, y1))
    left, right = sorted((x0, x1))
    return BoundingBox(x0=left, y0=top, x1=right, y1=bottom, page=page_index)


def is_native_pdf(document_path: str | Path) -> bool:
    path = Path(document_path)
    return path.suffix.lower() == ".pdf"
