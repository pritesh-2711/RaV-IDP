"""Dataset registry derived from the paper's experimental plan."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class DatasetAccess(str, Enum):
    PUBLIC = "public"
    MANUAL = "manual"
    CONTACT = "contact-authors"
    RESTRICTED = "restricted"


class DatasetSource(BaseModel):
    kind: Literal["http", "huggingface", "manual"]
    url: str
    filename: str | None = None
    allow_patterns: list[str] = Field(default_factory=list)
    ignore_patterns: list[str] = Field(default_factory=list)
    note: str | None = None


class DatasetSpec(BaseModel):
    key: str
    display_name: str
    stage: str
    access: DatasetAccess
    description: str
    expected_artifacts: list[str] = Field(default_factory=list)
    sources: list[DatasetSource] = Field(default_factory=list)
    license_note: str | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "smartdoc_qa": DatasetSpec(
        key="smartdoc_qa",
        display_name="SmartDoc-QA",
        stage="stage1_quality",
        access=DatasetAccess.PUBLIC,
        description="Mobile-captured document images for quality classification and pre-processing evaluation.",
        expected_artifacts=["images/", "annotations/"],
        sources=[
            DatasetSource(
                kind="huggingface",
                url="https://huggingface.co/datasets/34data/SmartDoc-QA",
                note="Public mirror for SmartDoc-QA.",
            ),
        ],
    ),
    "dociq": DatasetSpec(
        key="dociq",
        display_name="DocIQ",
        stage="stage1_quality",
        access=DatasetAccess.MANUAL,
        description="Document image quality dataset referenced in the paper; likely requires manual acquisition.",
        expected_artifacts=["images/", "labels/"],
        sources=[
            DatasetSource(kind="manual", url="manual://dociq", note="Add local dataset files manually once acquired."),
        ],
    ),
    "doclaynet": DatasetSpec(
        key="doclaynet",
        display_name="DocLayNet",
        stage="stage2_layout",
        access=DatasetAccess.PUBLIC,
        description="Document layout detection benchmark with page images and annotations.",
        expected_artifacts=["README.md", "data/"],
        sources=[
            DatasetSource(
                kind="huggingface",
                url="https://huggingface.co/datasets/ds4sd/DocLayNet",
                note="Clone or snapshot from Hugging Face.",
            ),
        ],
    ),
    "pubtabnet": DatasetSpec(
        key="pubtabnet",
        display_name="PubTabNet",
        stage="stage3a_tables",
        access=DatasetAccess.PUBLIC,
        description="Table extraction benchmark with HTML structure annotations.",
        expected_artifacts=["train.jsonl", "val.jsonl", "test.jsonl"],
        sources=[
            DatasetSource(
                kind="huggingface",
                url="https://huggingface.co/datasets/ajimeno/PubTabNet",
                note="Public Hugging Face mirror for PubTabNet.",
            ),
        ],
    ),
    "fintabnet": DatasetSpec(
        key="fintabnet",
        display_name="FinTabNet",
        stage="stage3a_tables",
        access=DatasetAccess.MANUAL,
        description="Financial table extraction benchmark for domain diversity.",
        expected_artifacts=["pdf/", "annotations/"],
        sources=[
            DatasetSource(kind="manual", url="manual://fintabnet", note="Add after separate acquisition."),
        ],
    ),
    "scanbank": DatasetSpec(
        key="scanbank",
        display_name="ScanBank",
        stage="stage3b_images",
        access=DatasetAccess.CONTACT,
        description="Figure extraction benchmark requiring author contact.",
        expected_artifacts=["images/", "annotations/"],
        sources=[
            DatasetSource(kind="manual", url="https://scanbank.cs.umd.edu", note="Contact authors for access."),
        ],
    ),
    "omnidocbench": DatasetSpec(
        key="omnidocbench",
        display_name="OmniDocBench",
        stage="stage3b_images",
        access=DatasetAccess.MANUAL,
        description="Document vision benchmark for image extraction and figure analysis.",
        expected_artifacts=["images/", "annotations/"],
        sources=[
            DatasetSource(kind="manual", url="manual://omnidocbench", note="Add local copy once obtained."),
        ],
    ),
    "funsd": DatasetSpec(
        key="funsd",
        display_name="FUNSD",
        stage="stage3c_text",
        access=DatasetAccess.PUBLIC,
        description="Form understanding dataset used for OCR/text extraction evaluation.",
        expected_artifacts=["dataset/", "training_data/", "testing_data/"],
        sources=[
            DatasetSource(
                kind="huggingface",
                url="https://huggingface.co/datasets/nielsr/funsd",
                note="Public Hugging Face mirror for FUNSD.",
            ),
        ],
    ),
    "sroie": DatasetSpec(
        key="sroie",
        display_name="SROIE",
        stage="stage3c_text",
        access=DatasetAccess.MANUAL,
        description="Receipt OCR benchmark frequently used for text extraction.",
        expected_artifacts=["images/", "annotations/"],
        sources=[
            DatasetSource(kind="manual", url="manual://sroie", note="Add after manual acquisition."),
        ],
    ),
    "docvqa": DatasetSpec(
        key="docvqa",
        display_name="DocVQA",
        stage="stage6_endtoend",
        access=DatasetAccess.RESTRICTED,
        description="End-to-end benchmark for document question answering.",
        expected_artifacts=["train/", "val/", "test/"],
        sources=[
            DatasetSource(kind="manual", url="https://www.docvqa.org/", note="Requires registration."),
        ],
    ),
}


def list_datasets() -> list[DatasetSpec]:
    """Return all registered dataset specs."""

    return list(DATASET_REGISTRY.values())


def get_dataset_spec(key: str) -> DatasetSpec:
    """Return a single dataset spec by key."""

    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset key: {key}") from exc
