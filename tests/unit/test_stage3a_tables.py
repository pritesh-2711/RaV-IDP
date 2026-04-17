from __future__ import annotations

from rav_idp.evaluation.stage3a_tables import _cluster_positions, _compute_teds, _derive_ground_truth, _gt_annotation_to_html


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

    assert gt.row_count == 1
    assert gt.total_row_count == 2
    assert gt.header_row_count == 1
    assert gt.col_count == 2
    assert gt.headers == ["H1", "H2"]
    assert gt.cell_texts == ["A", "B"]


def test_gt_annotation_to_html_keeps_td_attributes() -> None:
    annotation = {
        "html": {
            "structure": {
                "tokens": ["<thead>", "<tr>", "<td colspan=\"2\">", "</td>", "</tr>", "</thead>"]
            },
            "cells": [{"tokens": ["H", "1"]}],
        }
    }

    html = _gt_annotation_to_html(annotation)

    assert "<td colspan=\"2\">H1</td>" in html


def test_compute_teds_proxy_identity() -> None:
    html = "<table><thead><tr><td>A</td></tr></thead><tbody><tr><td>B</td></tr></tbody></table>"
    assert _compute_teds(html, html) == 1.0
