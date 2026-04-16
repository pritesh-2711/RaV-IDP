"""Table extractor."""

from __future__ import annotations

import pandas as pd

from ...models import DetectedRegion, EntityType, ExtractedEntity, TableContent


def _to_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "export_to_dict"):
        return value.export_to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _reconstruct_dataframe(docling_table_dict: dict) -> pd.DataFrame:
    table_record = _to_mapping(docling_table_dict)
    data = _to_mapping(table_record.get("data", {}))
    cells = data.get("table_cells", [])
    if not cells:
        return pd.DataFrame()

    normalized_cells = [_to_mapping(cell) for cell in cells]
    n_rows = max(cell.get("end_row_offset_idx", 0) for cell in normalized_cells) + 1
    n_cols = max(cell.get("end_col_offset_idx", 0) for cell in normalized_cells) + 1
    grid = [[""] * n_cols for _ in range(n_rows)]
    header_rows: set[int] = set()

    for cell in normalized_cells:
        row_idx = cell.get("start_row_offset_idx", 0)
        col_idx = cell.get("start_col_offset_idx", 0)
        grid[row_idx][col_idx] = str(cell.get("text", ""))
        if cell.get("column_header"):
            header_rows.add(row_idx)

    if header_rows:
        header_row_idx = sorted(header_rows)[0]
        columns = [str(value) for value in grid[header_row_idx]]
        data_rows = [row for idx, row in enumerate(grid) if idx not in header_rows]
    else:
        columns = [f"col_{index}" for index in range(n_cols)]
        data_rows = grid

    return pd.DataFrame(data_rows, columns=columns)


def extract_table(region: DetectedRegion) -> ExtractedEntity:
    """Extract a table into normalized serializations."""

    dataframe = _reconstruct_dataframe(region.raw_docling_record)
    content = TableContent(
        dataframe_json=dataframe.to_json(orient="split"),
        markdown=dataframe.to_markdown(index=False) if len(dataframe.columns) else "",
        csv=dataframe.to_csv(index=False),
        headers=[str(header) for header in dataframe.columns],
        row_count=len(dataframe.index),
        col_count=len(dataframe.columns),
    )
    return ExtractedEntity(
        region_id=region.region_id,
        entity_type=EntityType.TABLE,
        content=content,
        extractor_name="primary",
    )
