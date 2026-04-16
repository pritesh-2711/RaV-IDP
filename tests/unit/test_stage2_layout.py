from __future__ import annotations

from rav_idp.evaluation.stage2_layout import _iou
from rav_idp.models import BoundingBox


def test_iou_identical_boxes() -> None:
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10, page=0)
    assert _iou(bbox, bbox) == 1.0
