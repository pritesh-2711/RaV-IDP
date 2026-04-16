from __future__ import annotations

from rav_idp.components.extractors.table import extract_table
from rav_idp.models import BoundingBox, DetectedRegion, EntityType


class _Cell:
    def __init__(self, row: int, col: int, text: str, header: bool = False) -> None:
        self.start_row_offset_idx = row
        self.end_row_offset_idx = row
        self.start_col_offset_idx = col
        self.end_col_offset_idx = col
        self.text = text
        self.column_header = header

    def model_dump(self) -> dict:
        return {
            "start_row_offset_idx": self.start_row_offset_idx,
            "end_row_offset_idx": self.end_row_offset_idx,
            "start_col_offset_idx": self.start_col_offset_idx,
            "end_col_offset_idx": self.end_col_offset_idx,
            "text": self.text,
            "column_header": self.column_header,
        }


class _TableData:
    def __init__(self) -> None:
        self.table_cells = [
            _Cell(0, 0, "A", header=True),
            _Cell(0, 1, "B", header=True),
            _Cell(1, 0, "1"),
            _Cell(1, 1, "2"),
        ]

    def model_dump(self) -> dict:
        return {"table_cells": [cell.model_dump() for cell in self.table_cells]}


def test_extract_table_accepts_pydantic_like_docling_payload() -> None:
    region = DetectedRegion(
        region_id="r1",
        entity_type=EntityType.TABLE,
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10, page=0),
        original_crop=b"",
        processed_crop=b"",
        raw_docling_record={"data": _TableData()},
        page_index=0,
    )

    entity = extract_table(region)

    assert entity.content.headers == ["A", "B"]
    assert entity.content.row_count == 1
    assert entity.content.col_count == 2
