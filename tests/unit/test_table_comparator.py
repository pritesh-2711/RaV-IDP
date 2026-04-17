from __future__ import annotations

from rav_idp.components.comparators.table import _parse_ocr_to_signature


def test_parse_ocr_to_signature_uses_prediction_independent_header_split() -> None:
    ocr_text = "Header A  Header B\nrow1 col1  row1 col2\nrow2 col1  row2 col2\n"
    signature = _parse_ocr_to_signature(
        ocr_text,
        reference_signature={"headers": ["pred", "still", "ignored"]},
    )

    assert signature["row_count"] == 3
    assert signature["col_count"] >= 1
    assert signature["headers"] == ["Header A  Header B"]
    assert signature["cells"] == ["row1 col1  row1 col2", "row2 col1  row2 col2"]
