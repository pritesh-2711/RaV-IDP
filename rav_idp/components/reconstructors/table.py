"""Table reconstructor."""

from __future__ import annotations

import io

import pandas as pd
from PIL import Image, ImageDraw

from ...models import DetectedRegion, EntityType, ExtractedEntity, ReconstructedOutput, TableReconstruction


def render_dataframe_to_image(dataframe: pd.DataFrame) -> bytes:
    cell_width = 120
    cell_height = 28
    n_rows = len(dataframe) + 1
    n_cols = max(len(dataframe.columns), 1)
    image = Image.new("L", (cell_width * n_cols + 2, cell_height * n_rows + 2), color=255)
    draw = ImageDraw.Draw(image)

    headers = list(dataframe.columns) if len(dataframe.columns) else [""]
    for col_idx, header in enumerate(headers):
        x0 = col_idx * cell_width
        draw.rectangle([x0, 0, x0 + cell_width, cell_height], outline=0)
        draw.text((x0 + 4, 6), str(header)[:18], fill=0)

    for row_idx, row in enumerate(dataframe.itertuples(index=False), start=1):
        for col_idx, value in enumerate(row):
            x0 = col_idx * cell_width
            y0 = row_idx * cell_height
            draw.rectangle([x0, y0, x0 + cell_width, y0 + cell_height], outline=0)
            draw.text((x0 + 4, y0 + 6), str(value)[:18], fill=0)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_structural_signature(dataframe: pd.DataFrame) -> dict:
    return {
        "row_count": len(dataframe),
        "col_count": len(dataframe.columns),
        "headers": [str(header) for header in dataframe.columns],
        "cells": [str(value) for row in dataframe.itertuples(index=False) for value in row],
    }


def reconstruct_table(entity: ExtractedEntity, region: DetectedRegion) -> ReconstructedOutput:
    """Reconstruct a table for validation."""

    if entity.entity_type != EntityType.TABLE:
        raise ValueError("Table reconstruction requires a table entity.")
    dataframe = pd.read_json(io.StringIO(entity.content.dataframe_json), orient="split")
    content = TableReconstruction(
        rendered_image=render_dataframe_to_image(dataframe),
        structural_signature=build_structural_signature(dataframe),
    )
    return ReconstructedOutput(region_id=region.region_id, entity_type=EntityType.TABLE, content=content)
