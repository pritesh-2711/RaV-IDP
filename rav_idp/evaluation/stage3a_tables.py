"""Stage 3a: table extraction benchmark on PubTabNet."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import base64
import re

import pytesseract
import torch
import pandas as pd
from apted import APTED
from apted.helpers import Tree
from bs4 import BeautifulSoup
from Levenshtein import distance as lev_distance
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from ..components.comparators.table import compare_table
from ..components.extractors.table import extract_table
from ..components.region_preprocessor import preprocess_region
from ..components.region_quality_classifier import classify_region
from ..components.reconstructors.table import reconstruct_table, render_dataframe_to_image
from ..config import get_settings
from ..models import BoundingBox, DetectedRegion, EntityType, QualityClass

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency in tests
    DocumentConverter = None

# ---------------------------------------------------------------------------
# TableTransformer (TATR) — evaluation-only extractor for standalone image crops
# ---------------------------------------------------------------------------

_TATR_MODEL: Optional[TableTransformerForObjectDetection] = None
_TATR_PROCESSOR: Optional[AutoImageProcessor] = None
_TATR_MODEL_NAME = "microsoft/table-transformer-structure-recognition"
_TATR_MIN_IMAGE_DIM = 640   # pixels; upscale smaller images before TATR
_TATR_CELL_MIN_HEIGHT = 36  # pixels; upscale small cell crops before Tesseract
_TATR_THRESHOLD = 0.5       # detection confidence threshold


def _resolve_tatr_source() -> str:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshot_root = cache_root / "models--microsoft--table-transformer-structure-recognition" / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted((path for path in snapshot_root.iterdir() if path.is_dir()), key=os.path.getmtime, reverse=True)
        for snapshot in snapshots:
            if (snapshot / "preprocessor_config.json").exists() and (
                (snapshot / "model.safetensors").exists() or (snapshot / "pytorch_model.bin").exists()
            ):
                return str(snapshot)
    return _TATR_MODEL_NAME


def _load_tatr() -> tuple[TableTransformerForObjectDetection, AutoImageProcessor]:
    global _TATR_MODEL, _TATR_PROCESSOR
    if _TATR_MODEL is None:
        model_source = _resolve_tatr_source()
        local_only = model_source != _TATR_MODEL_NAME
        _TATR_PROCESSOR = AutoImageProcessor.from_pretrained(model_source, local_files_only=local_only)
        _TATR_MODEL = TableTransformerForObjectDetection.from_pretrained(model_source, local_files_only=local_only)
        _TATR_MODEL.eval()
        if torch.cuda.is_available():
            _TATR_MODEL = _TATR_MODEL.cuda()
    return _TATR_MODEL, _TATR_PROCESSOR


def _upscale_image(image: Image.Image, min_dim: int) -> Image.Image:
    w, h = image.size
    small = min(w, h)
    if small < min_dim:
        scale = min_dim / small
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return image


def _ocr_cell(cell_crop: Image.Image) -> str:
    w, h = cell_crop.size
    if h < _TATR_CELL_MIN_HEIGHT:
        scale = _TATR_CELL_MIN_HEIGHT / h
        cell_crop = cell_crop.resize((max(1, int(w * scale)), _TATR_CELL_MIN_HEIGHT), Image.LANCZOS)
    return pytesseract.image_to_string(cell_crop, config="--psm 6 --oem 3").strip()


def _edge_dark_ratio(gray: Image.Image, edge: str, thickness: int = 8) -> float:
    w, h = gray.size
    thickness = max(1, min(thickness, w, h))
    if edge == "bottom":
        strip = gray.crop((0, h - thickness, w, h))
    elif edge == "top":
        strip = gray.crop((0, 0, w, thickness))
    elif edge == "left":
        strip = gray.crop((0, 0, thickness, h))
    else:
        strip = gray.crop((w - thickness, 0, w, h))
    arr = torch.tensor(list(strip.getdata()), dtype=torch.float32)
    if arr.numel() == 0:
        return 0.0
    return float((255.0 - arr).mean().item() / 255.0)


def _pad_table_image(image: Image.Image) -> Image.Image:
    """Pad the table crop so border-adjacent text survives structure detection.

    This particularly helps when the last row sits very close to the lower
    crop edge and TATR truncates its row band.
    """
    gray = image.convert("L")
    w, h = image.size
    base_pad = max(8, int(round(min(w, h) * 0.03)))
    pads = {"left": base_pad, "top": base_pad, "right": base_pad, "bottom": base_pad}

    if _edge_dark_ratio(gray, "bottom") > 0.10:
        pads["bottom"] += max(10, int(round(h * 0.06)))
    if _edge_dark_ratio(gray, "top") > 0.10:
        pads["top"] += max(6, int(round(h * 0.03)))
    if _edge_dark_ratio(gray, "left") > 0.10:
        pads["left"] += max(6, int(round(w * 0.03)))
    if _edge_dark_ratio(gray, "right") > 0.10:
        pads["right"] += max(6, int(round(w * 0.03)))

    return ImageOps.expand(
        image,
        border=(pads["left"], pads["top"], pads["right"], pads["bottom"]),
        fill="white",
    )


def _merge_spanned_rows(
    rows: list[tuple[float, list[float]]],
    spanning_cells: list[list[float]],
    overlap_ratio: float = 0.5,
) -> list[tuple[float, list[float]]]:
    """Merge physical row bands that share a spanning cell into one logical row.

    TATR detects physical horizontal bands; a merged cell spanning N rows
    produces N separate bands. This collapses them back to one, which brings
    predicted row counts in line with the logical structure in the GT.
    """
    if not spanning_cells or not rows:
        return rows

    merge_groups: list[set[int]] = []
    for sx0, sy0, sx1, sy1 in spanning_cells:
        covered: list[int] = []
        for i, (_, (rx0, ry0, rx1, ry1)) in enumerate(rows):
            row_h = ry1 - ry0
            if row_h <= 0:
                continue
            ov0, ov1 = max(sy0, ry0), min(sy1, ry1)
            if ov1 > ov0 and (ov1 - ov0) / row_h >= overlap_ratio:
                covered.append(i)
        if len(covered) > 1:
            merge_groups.append(set(covered))

    if not merge_groups:
        return rows

    # Union-find: fuse groups that share any row index
    groups: list[set[int]] = []
    for pair in merge_groups:
        merged = False
        for group in groups:
            if group & pair:
                group |= pair
                merged = True
                break
        if not merged:
            groups.append(set(pair))

    consumed: set[int] = set()
    for g in groups:
        consumed |= g

    result: list[tuple[float, list[float]]] = [
        row for i, row in enumerate(rows) if i not in consumed
    ]
    for group in groups:
        boxes = [rows[i][1] for i in sorted(group)]
        mx0 = min(b[0] for b in boxes)
        my0 = min(b[1] for b in boxes)
        mx1 = max(b[2] for b in boxes)
        my1 = max(b[3] for b in boxes)
        result.append(((my0 + my1) / 2.0, [mx0, my0, mx1, my1]))

    result.sort(key=lambda r: r[0])
    return result


def _gpt_table_record(image_bytes: bytes) -> dict:
    """Extract a table from a low-quality crop using GPT-4o vision + JSONL streaming.

    Streams the response line-by-line: first line is {"headers": [...]} and each
    subsequent line is a data row [...].  This avoids context-limit issues on
    long tables and keeps token usage proportional to the actual table size.

    Returns a Docling-compatible table record, or empty dict on any failure.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        stream = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=4096,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract this table as JSON Lines (JSONL).\n"
                                "Line 1 MUST be: {\"headers\": [\"col1\", \"col2\", ...]}\n"
                                "Each subsequent line is one data row: [\"cell1\", \"cell2\", ...]\n"
                                "Use exactly the same number of values per row as the number of headers.\n"
                                "Output ONLY valid JSONL. No markdown, no explanation."
                            ),
                        },
                    ],
                }
            ],
        )

        buffer = ""
        headers: list[str] | None = None
        data_rows: list[list[str]] = []

        def _consume_line(line: str) -> None:
            nonlocal headers
            line = line.strip()
            if not line:
                return
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return
            if isinstance(obj, dict) and "headers" in obj:
                headers = [str(h) for h in obj["headers"]]
            elif isinstance(obj, list) and headers is not None:
                data_rows.append([str(v) for v in obj])

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            buffer += delta
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                _consume_line(line)
        _consume_line(buffer)

        if not headers:
            return {}

        n_cols = len(headers)
        cells = []
        for col_idx, h in enumerate(headers):
            cells.append({
                "start_row_offset_idx": 0,
                "end_row_offset_idx": 0,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx,
                "column_header": True,
                "text": h,
            })
        for row_idx, row in enumerate(data_rows, start=1):
            for col_idx, val in enumerate(row[:n_cols]):
                cells.append({
                    "start_row_offset_idx": row_idx,
                    "end_row_offset_idx": row_idx,
                    "start_col_offset_idx": col_idx,
                    "end_col_offset_idx": col_idx,
                    "column_header": False,
                    "text": val,
                })
        return {"data": {"table_cells": cells}}

    except Exception:
        return {}


def _tatr_table_record(image_bytes: bytes) -> dict:
    """Use TableTransformer structure recognition + Tesseract OCR to produce a
    Docling-compatible table record from a standalone table image crop.

    This is used only within the Stage 3a evaluation pipeline; the production
    pipeline uses Docling on full PDF documents where it works correctly.

    Returns an empty dict if TATR detects no rows or columns.
    """
    model, processor = _load_tatr()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _pad_table_image(image)
    image = _upscale_image(image, _TATR_MIN_IMAGE_DIM)
    w, h = image.size

    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs, threshold=_TATR_THRESHOLD, target_sizes=target_sizes
    )[0]

    label_map = model.config.id2label
    labels = results["labels"].cpu().tolist()
    boxes = results["boxes"].cpu().tolist()

    rows: list[tuple[float, list[float]]] = []   # (y_center, [x0,y0,x1,y1])
    cols: list[tuple[float, list[float]]] = []   # (x_center, [x0,y0,x1,y1])
    header_bands: list[tuple[float, float]] = [] # (y0, y1) of detected header regions
    spanning_cells: list[list[float]] = []       # [x0,y0,x1,y1] of spanning cell regions

    for label_id, box in zip(labels, boxes):
        name = label_map.get(label_id, "").lower()
        x0, y0, x1, y1 = box
        if "column header" in name:
            header_bands.append((y0, y1))
        elif "spanning" in name:
            spanning_cells.append([x0, y0, x1, y1])
        elif "row" in name and "header" not in name and "projected" not in name:
            rows.append(((y0 + y1) / 2.0, [x0, y0, x1, y1]))
        elif "column" in name and "header" not in name:
            cols.append(((x0 + x1) / 2.0, [x0, y0, x1, y1]))

    if not rows or not cols:
        return {}

    rows.sort(key=lambda r: r[0])
    cols.sort(key=lambda c: c[0])
    rows = _merge_spanned_rows(rows, spanning_cells)

    def _in_header(row_y_center: float) -> bool:
        return any(hy0 <= row_y_center <= hy1 for (hy0, hy1) in header_bands)

    cells = []
    for row_idx, (ry_center, (rx0, ry0, rx1, ry1)) in enumerate(rows):
        is_header = _in_header(ry_center)
        for col_idx, (_, (cx0, cy0, cx1, cy1)) in enumerate(cols):
            # Cell = intersection of row horizontal band x column vertical band
            cell_x0 = max(rx0, cx0)
            cell_y0 = max(ry0, cy0)
            cell_x1 = min(rx1, cx1)
            cell_y1 = min(ry1, cy1)
            if cell_x1 - cell_x0 < 1 or cell_y1 - cell_y0 < 1:
                continue
            cell_crop = image.crop((int(cell_x0), int(cell_y0), int(cell_x1), int(cell_y1)))
            cell_text = _ocr_cell(cell_crop)
            cells.append({
                "start_row_offset_idx": row_idx,
                "end_row_offset_idx": row_idx,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx,
                "column_header": is_header,
                "text": cell_text,
            })

    return {"data": {"table_cells": cells}}


def _tatr_col_count(tatr_record: dict) -> int | None:
    """Extract the number of distinct columns detected by TATR.

    Returns None when TATR produced no output so the caller can fall back to
    OCR-based estimation in the comparator.
    """
    if not tatr_record:
        return None
    cells = tatr_record.get("data", {}).get("table_cells", [])
    if not cells:
        return None
    col_indices = {cell["start_col_offset_idx"] for cell in cells}
    return len(col_indices) if col_indices else None


ANNOTATION_MEMBER_CANDIDATES = (
    "pubtabnet/PubTabNet_2.0.0.jsonl",
    "pubtabnet/PubTabNet_2.0.0.json",
    "pubtabnet/train.jsonl",
)


@dataclass
class TableGroundTruth:
    filename: str
    split: str
    row_count: int          # data rows only (excludes header rows)
    total_row_count: int    # all rows including header rows
    header_row_count: int   # header rows only
    col_count: int
    cell_texts: list[str]
    headers: list[str]


@dataclass
class TableBenchmarkRecord:
    sample_id: str
    filename: str
    ground_truth_rows: int
    predicted_rows: int
    ground_truth_cols: int
    predicted_cols: int
    ground_truth_nonempty_cells: int
    predicted_nonempty_cells: int
    row_match: bool
    col_match: bool
    cell_text_cer: float
    fidelity_score: float
    passed_threshold: bool


@dataclass
class TableBenchmarkSummary:
    split: str
    num_samples: int
    row_accuracy: float
    col_accuracy: float
    exact_shape_accuracy: float
    mean_row_abs_error: float
    mean_col_abs_error: float
    mean_cell_text_cer: float
    mean_fidelity: float
    pass_rate: float


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _tokens_to_text(tokens: list[str] | tuple[str, ...]) -> str:
    return _normalize_text("".join(str(token) for token in tokens))


def _cluster_positions(values: list[float], tolerance: float = 12.0) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _nearest_cluster(value: float, centers: list[float]) -> int:
    if not centers:
        return 0
    return min(range(len(centers)), key=lambda idx: abs(centers[idx] - value))


def _count_rows_from_structure(tokens: list[str]) -> tuple[int, int]:
    """Count <tr> elements inside <thead> and <tbody> from PubTabNet structure tokens.

    Returns (header_rows, data_rows).  Using the HTML structure is authoritative;
    bbox y-clustering is unreliable when rows are closely spaced because the
    greedy chaining groups adjacent rows into a single cluster.
    """
    in_thead = False
    thead_tr = tbody_tr = 0
    for token in tokens:
        if token == "<thead>":
            in_thead = True
        elif token == "</thead>":
            in_thead = False
        elif token == "<tr>":
            if in_thead:
                thead_tr += 1
            else:
                tbody_tr += 1
    return thead_tr, tbody_tr


def _derive_ground_truth(annotation: dict) -> TableGroundTruth:
    html = annotation.get("html", {})
    structure_tokens = html.get("structure", {}).get("tokens", [])
    cells = html.get("cells", html.get("cell", []))
    nonempty_cells = [cell for cell in cells if "bbox" in cell]
    ordered_cells = sorted(nonempty_cells, key=lambda cell: (cell["bbox"][1], cell["bbox"][0]))

    # Column count: cluster x-positions of bbox cells (still reliable).
    xs = [float(cell["bbox"][0]) for cell in ordered_cells]
    col_centers = _cluster_positions(xs)

    # Row counts: use HTML structure tokens — authoritative and immune to
    # the chaining artefact that afflicts bbox y-clustering on tight rows.
    header_row_count, data_row_count = _count_rows_from_structure(structure_tokens)
    if not structure_tokens:
        # Fallback for annotations that lack structure tokens.
        ys = [float(cell["bbox"][1]) for cell in ordered_cells]
        row_centers = _cluster_positions(ys)
        total_fallback = max(len(row_centers), 1 if ordered_cells else 0)
        header_row_count = 1 if ordered_cells and row_centers else 0
        data_row_count = max(0, total_fallback - header_row_count)
    total_row_count = header_row_count + data_row_count

    # Header texts: first y-cluster among bbox cells.
    headers: list[str] = []
    if ordered_cells:
        ys = [float(cell["bbox"][1]) for cell in ordered_cells]
        row_centers = _cluster_positions(ys)
        if row_centers:
            header_cells = [
                cell for cell in ordered_cells
                if _nearest_cluster(float(cell["bbox"][1]), row_centers) == 0
            ]
            headers = [_tokens_to_text(cell.get("tokens", [])) for cell in header_cells]

    # Data cell texts: bbox cells not in the first y-cluster.
    data_cells: list[dict] = []
    if ordered_cells:
        ys = [float(cell["bbox"][1]) for cell in ordered_cells]
        row_centers = _cluster_positions(ys)
        data_cells = [
            cell for cell in ordered_cells
            if not row_centers or _nearest_cluster(float(cell["bbox"][1]), row_centers) != 0
        ]

    return TableGroundTruth(
        filename=str(annotation["filename"]),
        split=str(annotation["split"]),
        row_count=data_row_count,
        total_row_count=total_row_count,
        header_row_count=header_row_count,
        col_count=max(len(col_centers), 1 if ordered_cells else 0),
        cell_texts=[_tokens_to_text(cell.get("tokens", [])) for cell in data_cells],
        headers=headers,
    )


def _gt_annotation_to_html(annotation: dict) -> str:
    """Reconstruct a PubTabNet HTML table from its annotation dict.

    Handles <td colspan="X"> / <td rowspan="Y"> style tokens by preserving
    the opening-tag attributes and injecting cell text content.
    """
    structure_tokens = annotation.get("html", {}).get("structure", {}).get("tokens", [])
    cells = annotation.get("html", {}).get("cells", [])
    cell_iter = iter(cells)
    html_parts = ["<table>"]
    skip_close = False
    for token in structure_tokens:
        if skip_close:
            if token == "</td>":
                skip_close = False
            continue
        if token.startswith("<td"):
            try:
                cell = next(cell_iter)
            except StopIteration:
                cell = {"tokens": []}
            content = "".join(str(t) for t in cell.get("tokens", []))
            html_parts.append(f"{token}{content}</td>")
            skip_close = True
        else:
            html_parts.append(token)
    html_parts.append("</table>")
    return "".join(html_parts)


def _html_to_apted_tree(html: str) -> Tree:
    soup = BeautifulSoup(html, "html.parser")

    def _node(element) -> str:
        if isinstance(element, str):
            text = element.strip()
            if not text:
                return ""
            text = re.sub(r"[{}]", "", text)
            return "{" + text + "}" if text else ""
        tag = element.name or "unknown"
        children = "".join(_node(c) for c in element.children)
        return "{" + tag + children + "}"

    table = soup.find("table")
    if table is None:
        return Tree.from_text("{empty}")
    return Tree.from_text(_node(table))


def _compute_teds(pred_html: str, gt_html: str) -> float:
    pred_tree = _html_to_apted_tree(pred_html)
    gt_tree = _html_to_apted_tree(gt_html)
    n_pred = str(pred_tree).count("{")
    n_gt = str(gt_tree).count("{")
    denom = max(n_pred, n_gt)
    if denom == 0:
        return 1.0
    ted = APTED(pred_tree, gt_tree).compute_edit_distance()
    return max(0.0, 1.0 - ted / denom)


def _annotation_member(archive: tarfile.TarFile) -> tarfile.TarInfo:
    candidate_set = set(ANNOTATION_MEMBER_CANDIDATES)
    for member in archive:
        if member.name in candidate_set:
            return member
    raise FileNotFoundError(
        "Could not locate a PubTabNet annotation jsonl member inside the archive. "
        f"Tried: {', '.join(ANNOTATION_MEMBER_CANDIDATES)}"
    )


def _annotation_cache_path(dataset_root: Path) -> Path:
    return dataset_root / "PubTabNet_2.0.0.jsonl"


def _ensure_annotation_cache(dataset_root: Path, archive_path: Path) -> Path:
    cache_path = _annotation_cache_path(dataset_root)
    if cache_path.exists():
        return cache_path

    with tarfile.open(archive_path, "r:gz") as archive:
        member = _annotation_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(member.name)
        cache_path.write_bytes(handle.read())
    return cache_path


def _iter_annotations(dataset_root: Path, archive_path: Path, split: str, limit: int | None) -> list[dict]:
    records: list[dict] = []
    annotation_path = _ensure_annotation_cache(dataset_root, archive_path)
    with annotation_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            payload = json.loads(raw_line)
            if payload.get("split") != split:
                continue
            records.append(payload)
            if limit is not None and len(records) >= limit:
                break
    return records


def _load_image_bytes_batch(archive_path: Path, split: str, filenames: list[str]) -> dict[str, bytes]:
    remaining = set(filenames)
    found: dict[str, bytes] = {}
    candidates = {f"pubtabnet/{split}/{filename}": filename for filename in filenames}
    candidates.update({f"pubtabnet/{filename}": filename for filename in filenames})

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            target_name = candidates.get(member.name)
            if target_name is None:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            found[target_name] = handle.read()
            remaining.discard(target_name)
            if not remaining:
                break

    if remaining:
        raise FileNotFoundError(
            f"Could not locate table images for: {', '.join(sorted(remaining))} in split {split}."
        )
    return found


def _make_region(sample_id: str, image_bytes: bytes) -> DetectedRegion:
    width, height = Image.open(io.BytesIO(image_bytes)).size
    return DetectedRegion(
        region_id=sample_id,
        entity_type=EntityType.TABLE,
        bbox=BoundingBox(x0=0, y0=0, x1=width, y1=height, page=0),
        original_crop=image_bytes,
        processed_crop=image_bytes,
        raw_docling_record={},
        page_index=0,
    )


def _docling_table_record(image_bytes: bytes) -> dict:
    if DocumentConverter is None:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as handle:
        handle.write(image_bytes)
        handle.flush()
        result = DocumentConverter().convert(handle.name)
    tables = getattr(result.document, "tables", []) or []
    if not tables:
        return {}
    table = tables[0]
    if hasattr(table, "export_to_dict"):
        return table.export_to_dict()
    return getattr(table, "__dict__", {})


def _dataframe_signature(frame_json: str) -> tuple[int, int, list[str], list[str]]:
    frame = pd.read_json(io.StringIO(frame_json), orient="split")
    headers = [_normalize_text(str(header)) for header in frame.columns]
    cell_texts = [
        _normalize_text(str(value))
        for row in frame.itertuples(index=False)
        for value in row
        if str(value).strip()
    ]
    return len(frame.index), len(frame.columns), headers, cell_texts


def _cer(reference_items: list[str], hypothesis_items: list[str]) -> float:
    reference = "\n".join(reference_items)
    hypothesis = "\n".join(hypothesis_items)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return lev_distance(hypothesis, reference) / len(reference)


def _save_mismatch_visual(
    record: "TableBenchmarkRecord",
    original_bytes: bytes,
    pred_df_json: str,
    out_dir: Path,
    tatr_row_bands: list[list[float]] | None = None,
    tatr_col_bands: list[list[float]] | None = None,
) -> None:
    """Save a side-by-side comparison PNG for a row/col mismatch sample.

    Left panel: original crop with TATR row/col band lines overlaid (when
    tatr_row_bands / tatr_col_bands are supplied).  This is the grid overlay
    that makes phantom splits vs real rows immediately visible without any
    API cost.

    Right panel: extracted DataFrame rendered as a grid.
    """
    from PIL import ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)

    orig = Image.open(io.BytesIO(original_bytes)).convert("RGB")

    # Draw TATR band lines onto a copy of the original crop.
    annotated = orig.copy()
    if tatr_row_bands or tatr_col_bands:
        adraw = ImageDraw.Draw(annotated)
        ow, oh = orig.size
        for _, (bx0, by0, bx1, by1) in (tatr_row_bands or []):
            # Top edge of each row band
            adraw.line([(0, int(by0)), (ow, int(by0))], fill=(255, 50, 50), width=2)
        for _, (bx0, by0, bx1, by1) in (tatr_col_bands or []):
            # Left edge of each col band
            adraw.line([(int(bx0), 0), (int(bx0), oh)], fill=(50, 50, 255), width=2)

    frame = pd.read_json(io.StringIO(pred_df_json), orient="split")
    pred_bytes = render_dataframe_to_image(frame)
    pred_img = Image.open(io.BytesIO(pred_bytes)).convert("RGB")

    title_h = 50
    target_h = max(annotated.height, pred_img.height, 200)
    orig_scaled = annotated.resize(
        (max(1, int(annotated.width * target_h / annotated.height)), target_h), Image.LANCZOS
    )
    pred_scaled = pred_img.resize(
        (max(1, int(pred_img.width * target_h / pred_img.height)), target_h), Image.LANCZOS
    )

    total_w = orig_scaled.width + pred_scaled.width + 10
    total_h = title_h + target_h
    canvas = Image.new("RGB", (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    row_ok = "✓" if record.row_match else "✗"
    col_ok = "✓" if record.col_match else "✗"
    overlay_note = " (red=rows blue=cols)" if (tatr_row_bands or tatr_col_bands) else ""
    title = (
        f"{record.filename}  |  "
        f"rows GT={record.ground_truth_rows} pred={record.predicted_rows} {row_ok}  |  "
        f"cols GT={record.ground_truth_cols} pred={record.predicted_cols} {col_ok}  |  "
        f"CER={record.cell_text_cer:.3f}"
    )
    draw.text((8, 10), title, fill=(30, 30, 30))
    draw.text((8, 30), f"Original crop{overlay_note}  →  Predicted grid", fill=(80, 80, 80))

    canvas.paste(orig_scaled, (0, title_h))
    canvas.paste(pred_scaled, (orig_scaled.width + 10, title_h))

    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", record.sample_id)
    canvas.save(out_dir / f"{safe_id}.png")


_MIN_EVAL_DIM = 150  # Skip table crops smaller than this on the short side in evaluation.
                     # Very small images produce cell crops too tiny for reliable OCR.
                     # The production pipeline always processes all regions regardless of size.
_CANDIDATE_OVERSAMPLE = 3  # Read this many candidates per desired sample to absorb filter losses.


def run_table_benchmark(
    dataset_root: str | Path,
    split: str = "test",
    limit: int | None = None,
    mismatches_dir: Path | None = None,
) -> tuple[TableBenchmarkSummary, list[TableBenchmarkRecord]]:
    """Run a table-region benchmark on PubTabNet table crops."""

    settings = get_settings()
    dataset_root = Path(dataset_root)
    archive_path = dataset_root / "pubtabnet.tar.gz"
    candidate_limit = limit * _CANDIDATE_OVERSAMPLE if limit is not None else None
    annotations = _iter_annotations(dataset_root, archive_path, split=split, limit=candidate_limit)
    image_bytes_by_name = _load_image_bytes_batch(
        archive_path,
        split=split,
        filenames=[str(annotation["filename"]) for annotation in annotations],
    )

    records: list[TableBenchmarkRecord] = []
    for annotation in annotations:
        if limit is not None and len(records) >= limit:
            break

        gt = _derive_ground_truth(annotation)
        sample_id = str(annotation.get("imgid", gt.filename))
        image_bytes = image_bytes_by_name[gt.filename]

        img_w, img_h = Image.open(io.BytesIO(image_bytes)).size
        if min(img_w, img_h) < _MIN_EVAL_DIM:
            continue

        region = preprocess_region(classify_region(_make_region(sample_id, image_bytes)))

        crop_bytes = region.processed_crop or image_bytes
        is_degraded = region.quality_class == QualityClass.SCANNED_DEGRADED

        tatr_row_bands: list[list[float]] | None = None
        tatr_col_bands: list[list[float]] | None = None

        if is_degraded:
            # Low-quality crop: try GPT-4o first for better cell-level accuracy.
            # GPT streams rows as JSONL so long tables don't hit context limits.
            table_record = _gpt_table_record(crop_bytes)
            detected_cols = _tatr_col_count(table_record) if table_record else None
            if not table_record:
                # GPT unavailable or failed — fall through to TATR
                tatr_record = _tatr_table_record(crop_bytes)
                detected_cols = _tatr_col_count(tatr_record)
                table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)
        else:
            # TATR is the primary structure recogniser for standalone PubTabNet image crops.
            # The production pipeline uses Docling on full PDF documents where it works correctly.
            tatr_record = _tatr_table_record(crop_bytes)

            # Capture TATR-detected column count before the record is consumed by
            # extract_table. This count is passed to compare_table so the comparator
            # uses a reliable detector-supplied value instead of guessing from OCR
            # whitespace tokenisation.
            detected_cols = _tatr_col_count(tatr_record)
            table_record = tatr_record if tatr_record else _docling_table_record(crop_bytes)

        region = region.model_copy(update={"raw_docling_record": table_record})
        entity = extract_table(region)
        reconstruction = reconstruct_table(entity, region)

        # skip_visual=True: SSIM between a rendered grid and a real table photo
        # is meaningless regardless of extraction quality. Score on structural
        # fidelity only when evaluating against image crops.
        fidelity = compare_table(
            reconstruction.content,
            region,
            settings.threshold_table,
            skip_visual=True,
            detected_col_count=detected_cols,
        )

        predicted_rows, predicted_cols, predicted_headers, predicted_cells = _dataframe_signature(
            entity.content.dataframe_json
        )
        combined_gt_cells = gt.headers + gt.cell_texts
        combined_pred_cells = predicted_headers + predicted_cells

        rec = TableBenchmarkRecord(
            sample_id=sample_id,
            filename=gt.filename,
            ground_truth_rows=gt.row_count,
            predicted_rows=predicted_rows,
            ground_truth_cols=gt.col_count,
            predicted_cols=predicted_cols,
            ground_truth_nonempty_cells=len(gt.cell_texts),
            predicted_nonempty_cells=len(predicted_cells),
            row_match=gt.row_count == predicted_rows,
            col_match=gt.col_count == predicted_cols,
            cell_text_cer=_cer(combined_gt_cells, combined_pred_cells),
            fidelity_score=fidelity.fidelity_score,
            passed_threshold=fidelity.passed_threshold,
        )
        records.append(rec)

        if mismatches_dir is not None and (not rec.row_match or not rec.col_match):
            _save_mismatch_visual(rec, image_bytes, entity.content.dataframe_json, mismatches_dir)

    if not records:
        raise ValueError(f"No PubTabNet samples found for split={split!r}.")

    row_accuracy = sum(record.row_match for record in records) / len(records)
    col_accuracy = sum(record.col_match for record in records) / len(records)
    exact_shape_accuracy = sum(
        record.row_match and record.col_match for record in records
    ) / len(records)
    mean_row_abs_error = sum(
        abs(record.predicted_rows - record.ground_truth_rows) for record in records
    ) / len(records)
    mean_col_abs_error = sum(
        abs(record.predicted_cols - record.ground_truth_cols) for record in records
    ) / len(records)
    mean_cell_text_cer = sum(record.cell_text_cer for record in records) / len(records)
    mean_fidelity = sum(record.fidelity_score for record in records) / len(records)
    pass_rate = sum(record.passed_threshold for record in records) / len(records)

    return (
        TableBenchmarkSummary(
            split=split,
            num_samples=len(records),
            row_accuracy=row_accuracy,
            col_accuracy=col_accuracy,
            exact_shape_accuracy=exact_shape_accuracy,
            mean_row_abs_error=mean_row_abs_error,
            mean_col_abs_error=mean_col_abs_error,
            mean_cell_text_cer=mean_cell_text_cer,
            mean_fidelity=mean_fidelity,
            pass_rate=pass_rate,
        ),
        records,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 3a PubTabNet table benchmark.")
    parser.add_argument("--dataset-root", default="data/raw/pubtabnet", help="Path to the PubTabNet dataset root.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--save-mismatches",
        default=None,
        metavar="DIR",
        help="Directory to save side-by-side comparison images for row/col mismatch samples.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    mismatches_dir = Path(args.save_mismatches) if args.save_mismatches else None
    summary, records = run_table_benchmark(
        args.dataset_root, split=args.split, limit=args.limit, mismatches_dir=mismatches_dir
    )
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
