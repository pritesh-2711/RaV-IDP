"""I/O helpers for pipeline runs."""

from __future__ import annotations

import json
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


def write_entity_records(records: list[EntityRecord], output_path: str | Path) -> Path:
    target = ensure_parent(output_path)
    target.write_text(
        json.dumps([record.model_dump(mode="json") for record in records], indent=2),
        encoding="utf-8",
    )
    return target
