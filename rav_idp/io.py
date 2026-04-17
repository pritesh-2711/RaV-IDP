"""I/O helpers for pipeline runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import get_settings
from .models import EntityRecord


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def default_document_output(document_path: str | Path, suffix: str = ".entities.json") -> Path:
    settings = get_settings()
    doc_path = Path(document_path)
    return (settings.results_root / f"{doc_path.stem}{suffix}").resolve()


def default_visual_run_dir(document_path: str | Path) -> Path:
    settings = get_settings()
    doc_path = Path(document_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (settings.results_root / "pipeline_runs" / f"{doc_path.stem}_{stamp}").resolve()


def write_entity_records(records: list[EntityRecord], output_path: str | Path) -> Path:
    serialized = []
    for record in records:
        payload = record.model_dump(mode="python")
        content = payload.get("content", {})
        if isinstance(content, dict):
            content.pop("crop_bytes", None)
        serialized.append(payload)
    target = ensure_parent(output_path)
    target.write_text(
        json.dumps(serialized, indent=2, default=str),
        encoding="utf-8",
    )
    return target
