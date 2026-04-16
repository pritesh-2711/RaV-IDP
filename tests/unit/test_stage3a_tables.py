from __future__ import annotations

from rav_idp.evaluation.stage3a_tables import _cluster_positions, _derive_ground_truth


def test_cluster_positions_groups_nearby_values() -> None:
    clusters = _cluster_positions([10.0, 11.0, 35.0, 36.5], tolerance=3.0)
    assert len(clusters) == 2
    assert round(clusters[0], 1) == 10.5
    assert round(clusters[1], 2) == 35.75


def test_derive_ground_truth_from_cell_bboxes() -> None:
    annotation = {
        "filename": "sample.png",
        "split": "test",
        "html": {
            "cells": [
                {"tokens": ["H1"], "bbox": [10, 10, 50, 20]},
                {"tokens": ["H2"], "bbox": [80, 10, 120, 20]},
                {"tokens": ["A"], "bbox": [10, 40, 50, 50]},
                {"tokens": ["B"], "bbox": [80, 40, 120, 50]},
            ]
        },
    }

    gt = _derive_ground_truth(annotation)

    assert gt.row_count == 2
    assert gt.col_count == 2
    assert gt.headers == ["H1", "H2"]
    assert gt.cell_texts == ["H1", "H2", "A", "B"]
