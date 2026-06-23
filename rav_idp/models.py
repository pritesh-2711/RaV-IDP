"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

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


class InputKind(str, Enum):
    NATIVE_PDF = "native-pdf"
    RASTER_PDF = "raster-pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


class AcquisitionMode(str, Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    PHOTOGRAPHED = "photographed"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class EvidenceVersion(str, Enum):
    SOURCE_PAGE_V1 = "source_page_v1"
    WORKING_PAGE_V2 = "working_page_v2"
    SOURCE_CROP_V1 = "source_crop_v1"
    WORKING_CROP_V2 = "working_crop_v2"
    ENTITY_INPUT_V3 = "entity_input_v3"


class TransformKind(str, Enum):
    ROTATE_ORIENTATION = "rotate_orientation"
    DESKEW = "deskew"
    ILLUMINATION_NORMALIZATION = "illumination_normalization"
    CLAHE = "clahe"
    DENOISE = "denoise"
    UNSHARP_MASK = "unsharp_mask"
    ADAPTIVE_BINARIZATION = "adaptive_binarization"


class TransformSpec(BaseModel):
    kind: TransformKind
    parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    reason: str


class CoordinateMapping(BaseModel):
    """Homogeneous source-to-target pixel mapping, row-major 3x3."""

    matrix: list[float] = Field(min_length=9, max_length=9)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)


class RestorationPlan(BaseModel):
    page_index: int
    source_version: EvidenceVersion = EvidenceVersion.SOURCE_PAGE_V1
    target_version: EvidenceVersion = EvidenceVersion.WORKING_PAGE_V2
    operations: list[TransformSpec] = Field(default_factory=list)
    planner_version: str
    warnings: list[str] = Field(default_factory=list)


class RestorationIntegrity(BaseModel):
    page_index: int
    passed: bool
    threshold: float = Field(ge=0.0, le=1.0)
    structural_similarity: float = Field(ge=0.0, le=1.0)
    foreground_retention: float = Field(ge=0.0)
    edge_retention: float = Field(ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class PageEvidence(BaseModel):
    page_index: int
    source_page_v1: bytes
    candidate_page_v2: bytes | None = None
    working_page_v2: bytes
    plan: RestorationPlan
    mapping_v1_to_v2: CoordinateMapping
    integrity: RestorationIntegrity
    backend_name: str = "deterministic-opencv-v1"
    backend_task: str | None = None


class RestorationBackendResult(BaseModel):
    image_bytes: bytes
    mapping: CoordinateMapping | None
    backend_name: str
    task: str | None = None
    warnings: list[str] = Field(default_factory=list)


class EntityEvidence(BaseModel):
    region_id: str
    entity_type: EntityType
    source_crop_v1: bytes
    working_crop_v2: bytes
    entity_input_v3: bytes
    operations: list[TransformSpec] = Field(default_factory=list)


class EvidenceUsage(BaseModel):
    """Explicit evidence declaration required before measuring extraction gains."""

    restoration_anchor: EvidenceVersion
    layout_input: EvidenceVersion
    primary_extraction_input: EvidenceVersion
    validation_reference: EvidenceVersion
    fallback_inputs: list[EvidenceVersion] = Field(default_factory=list)


class RestorationEvaluationRecord(BaseModel):
    input_id: str
    page_index: int
    variant: str
    integrity_passed: bool
    delta_cer: float | None = None
    delta_layout_f1: float | None = None
    delta_table_accuracy: float | None = None
    delta_extraction_fidelity: float | None = None
    delta_final_trust_rate: float | None = None
    clean_regression: bool | None = None
    evidence_usage: EvidenceUsage


class AblationVariant(BaseModel):
    name: str
    enabled_transforms: list[TransformKind] = Field(default_factory=list)
    backend_name: str = "deterministic"


class QualityProfile(BaseModel):
    """Versioned, multi-dimensional document image quality measurements."""

    input_kind: InputKind = InputKind.UNKNOWN
    acquisition_mode: AcquisitionMode = AcquisitionMode.UNKNOWN

    blur: float | None = Field(default=None, ge=0.0, le=1.0)
    low_contrast: float | None = Field(default=None, ge=0.0, le=1.0)
    underexposure: float | None = Field(default=None, ge=0.0, le=1.0)
    overexposure: float | None = Field(default=None, ge=0.0, le=1.0)
    uneven_illumination: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow: float | None = Field(default=None, ge=0.0, le=1.0)
    noise: float | None = Field(default=None, ge=0.0, le=1.0)
    skew: float | None = Field(default=None, ge=0.0, le=1.0)
    perspective: float | None = Field(default=None, ge=0.0, le=1.0)
    blockiness: float | None = Field(default=None, ge=0.0, le=1.0)

    skew_angle_degrees: float | None = None
    rotation_required_degrees: int | None = None
    orientation_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    brightness_mean: float | None = None
    text_density: float | None = Field(default=None, ge=0.0, le=1.0)
    handwriting_likelihood: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap_likelihood: float | None = Field(default=None, ge=0.0, le=1.0)

    derived_label: str = "unknown"
    metric_version: str
    raw_measurements: dict[str, float | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class QualityAssessment(BaseModel):
    scope: Literal["page", "region"]
    page_index: int
    region_id: str | None = None
    entity_type: EntityType | None = None
    bbox: BoundingBox | None = None
    profile: QualityProfile


class PageRecord(BaseModel):
    page_index: int
    quality_class: QualityClass
    secondary_flags: list[QualityClass] = Field(default_factory=list)
    raw_image: bytes
    processed_image: bytes
    quality_assessment: QualityAssessment | None = None


class DetectedRegion(BaseModel):
    region_id: str
    entity_type: EntityType
    bbox: BoundingBox
    original_crop: bytes
    processed_crop: bytes | None = None
    quality_class: QualityClass | None = None
    secondary_flags: list[QualityClass] = Field(default_factory=list)
    quality_assessment: QualityAssessment | None = None
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
    # classification_label / confidence come from Docling's structural classifier
    classification_label: str | None
    classification_confidence: float | None
    # fields below are populated by the image enricher after fidelity validation
    image_type: str | None = None          # photo / chart / diagram / flowchart / logo / screenshot / table_as_image / other
    description: str | None = None         # natural language description of the image content
    extracted_text: str | None = None      # verbatim text visible within the image (OCR via vision model)
    structured_data: dict | None = None    # chart: {title, axes, data_points, trend}; None for non-chart types


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


class PipelineTraceRecord(BaseModel):
    region_id: str
    entity_type: EntityType
    primary_entity: ExtractedEntity
    primary_reconstruction: ReconstructedOutput
    primary_fidelity: FidelityResult
    fallback_entity: ExtractedEntity | None = None
    fallback_reconstruction: ReconstructedOutput | None = None
    fallback_fidelity: FidelityResult | None = None
    final_entity: ExtractedEntity
    final_fidelity: FidelityResult
    provenance: ProvenanceRecord
    context_text: str | None = None
