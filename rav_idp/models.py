"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class EntityType(str, Enum):
    TABLE = "table"
    IMAGE = "image"
    TEXT = "text"
    FORMULA = "formula"
    URL = "url"


class QualityClass(str, Enum):
    CLEAN = "clean"
    SCANNED_CLEAN = "scanned-clean"
    SCANNED_DEGRADED = "scanned-degraded"
    PHOTOGRAPHED = "photographed"
    HANDWRITTEN = "handwritten"
    OVERLAPPING = "overlapping"


class PageRecord(BaseModel):
    page_index: int
    quality_class: QualityClass
    secondary_flags: list[QualityClass] = Field(default_factory=list)
    raw_image: bytes
    processed_image: bytes


class DetectedRegion(BaseModel):
    region_id: str
    entity_type: EntityType
    bbox: BoundingBox
    original_crop: bytes
    processed_crop: bytes | None = None
    quality_class: QualityClass | None = None
    secondary_flags: list[QualityClass] = Field(default_factory=list)
    raw_docling_record: dict
    page_index: int


class TableContent(BaseModel):
    dataframe_json: str
    markdown: str
    csv: str
    headers: list[str]
    row_count: int
    col_count: int


class ImageContent(BaseModel):
    crop_bytes: bytes
    classification_label: str | None
    classification_confidence: float | None
    description: str | None = None


class TextContent(BaseModel):
    text: str
    urls: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    region_id: str
    entity_type: EntityType
    content: TableContent | ImageContent | TextContent
    extractor_name: str


class TableReconstruction(BaseModel):
    rendered_image: bytes
    structural_signature: dict


class ImageReconstruction(BaseModel):
    phash_hex: str
    sharpness_crop: float
    sharpness_original: float
    caption_found: bool


class TextReconstruction(BaseModel):
    reocr_text: str


class ReconstructedOutput(BaseModel):
    region_id: str
    entity_type: EntityType
    content: TableReconstruction | ImageReconstruction | TextReconstruction


class FidelityResult(BaseModel):
    region_id: str
    entity_type: EntityType
    fidelity_score: float
    passed_threshold: bool
    threshold_used: float
    component_scores: dict
    extractor_name: str


class ProvenanceRecord(BaseModel):
    region_id: str
    primary_fidelity: float | None
    fallback_triggered: bool = False
    fallback_fidelity: float | None = None
    final_extractor: str
    final_fidelity: float
    low_confidence_flag: bool = False


class ContextRecord(BaseModel):
    region_id: str
    caption_text: str | None
    preceding_text: list[str] = Field(default_factory=list)
    following_text: list[str] = Field(default_factory=list)
    neighbor_region_ids: list[str] = Field(default_factory=list)


class EntityRecord(BaseModel):
    region_id: str
    page_index: int
    entity_type: EntityType
    bbox: BoundingBox
    content: TableContent | ImageContent | TextContent
    fidelity_score: float
    low_confidence_flag: bool
    context: ContextRecord
    provenance: ProvenanceRecord
