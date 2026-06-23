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
    expected_size_bytes: int | None = None
    checksum: str | None = None
    download_parts: int = Field(default=1, ge=1, le=16)
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
        description=(
            "Mobile-captured documents with capture parameters, transcriptions, OCR output, "
            "and single/multiple distortions for quality and enhancement evaluation."
        ),
        expected_artifacts=["Dataset SmartDoc-QA.zip"],
        sources=[
            DatasetSource(
                kind="http",
                url="https://zenodo.org/api/records/5293201/files/Dataset%20SmartDoc-QA.zip/content",
                filename="Dataset SmartDoc-QA.zip",
                expected_size_bytes=13659581479,
                checksum="md5:643c5e54606626694b17ac5b8984baab",
                download_parts=8,
                note="Official open Zenodo record supplied by the dataset authors.",
            ),
        ],
        license_note="CC BY 4.0; citation required by the dataset record.",
    ),
    "docunet": DatasetSpec(
        key="docunet",
        display_name="DocUNet Benchmark",
        stage="restoration_geometry_eval",
        access=DatasetAccess.MANUAL,
        description="Real photographed/cropped documents paired with flatbed-scanned references for dewarping evaluation.",
        expected_artifacts=["original.zip", "crop.zip", "scan.zip"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://vision.cs.stonybrook.edu/~kema/docwarp/original.zip",
                filename="original.zip",
                note="Official Stony Brook CVLab benchmark archive; listed as 328 MB. TLS certificate verification currently fails on the host.",
            ),
            DatasetSource(
                kind="manual",
                url="https://vision.cs.stonybrook.edu/~kema/docwarp/crop.zip",
                filename="crop.zip",
                note="Official Stony Brook CVLab benchmark archive; listed as 281 MB. TLS certificate verification currently fails on the host.",
            ),
            DatasetSource(
                kind="manual",
                url="https://vision.cs.stonybrook.edu/~kema/docwarp/scan.zip",
                filename="scan.zip",
                note="Official Stony Brook CVLab benchmark archive; listed as 416 MB. TLS certificate verification currently fails on the host.",
            ),
        ],
        license_note="Research benchmark; cite DocUNet. The official page does not state a separate data license.",
    ),
    "uvdoc_benchmark": DatasetSpec(
        key="uvdoc_benchmark",
        display_name="UVDoc Benchmark",
        stage="restoration_geometry_eval",
        access=DatasetAccess.PUBLIC,
        description="Official pseudo-photorealistic UVDoc evaluation benchmark with geometric annotations.",
        expected_artifacts=["UVDoc_benchmark.zip", "UVDoc_benchmark/"],
        sources=[
            DatasetSource(
                kind="http",
                url="https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip",
                filename="UVDoc_benchmark.zip",
                expected_size_bytes=775129594,
                note="Official archive linked by the authors' UVDoc-Dataset repository.",
            ),
        ],
        license_note="Author repository is MIT; review third-party texture provenance before redistribution.",
    ),
    "doc3d": DatasetSpec(
        key="doc3d",
        display_name="Doc3D",
        stage="restoration_geometry_training",
        access=DatasetAccess.MANUAL,
        description="100K-image synthetic document-dewarping training set with backward maps and 3D supervision.",
        expected_artifacts=["doc3d/img_*.zip", "doc3d/bm_*.zip"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://huggingface.co/datasets/StonyBrook-CVLab/doc3D-dataset",
                note=(
                    "Official Stony Brook CVLab distribution. Backward-map shards alone exceed available "
                    "workspace storage; select modalities manually rather than snapshotting the whole repo."
                ),
            ),
        ],
        license_note="Official repository is MIT; constituent document textures retain their source licenses.",
    ),
    "dir300": DatasetSpec(
        key="dir300",
        display_name="DIR300",
        stage="restoration_geometry_eval",
        access=DatasetAccess.MANUAL,
        description="300-image real-world document rectification benchmark released by the DocGeoNet authors.",
        expected_artifacts=["distorted/", "gt/"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://github.com/fh2019ustc/DocGeoNet",
                note="Official repository links the test set through Google Drive; acquire manually.",
            ),
        ],
    ),
    "docreal": DatasetSpec(
        key="docreal",
        display_name="DocReal Benchmark",
        stage="restoration_geometry_eval",
        access=DatasetAccess.MANUAL,
        description="Real-life Chinese document dewarping benchmark from the DocReal paper.",
        expected_artifacts=["distorted/", "reference/"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://openaccess.thecvf.com/content/WACV2024/html/Yu_DocReal_Robust_Document_Dewarping_of_Real-Life_Images_via_Attention-Enhanced_Control_WACV_2024_paper.html",
                note="No directly downloadable official archive was confirmed; do not use an unverified mirror.",
            ),
        ],
    ),
    "dibco_series": DatasetSpec(
        key="dibco_series",
        display_name="DIBCO / H-DIBCO 2016-2018",
        stage="restoration_binarization_eval",
        access=DatasetAccess.PUBLIC,
        description="Official degraded document binarization inputs and ground truths from three competition years.",
        expected_artifacts=["DIBCO2016_dataset-original.zip", "DIBCO2017_Dataset.7z", "dibco2018_Dataset.zip"],
        sources=[
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
                filename="DIBCO2016_dataset-original.zip",
                expected_size_bytes=8985981,
            ),
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip",
                filename="DIBCO2016_dataset-GT.zip",
                expected_size_bytes=240396,
            ),
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z",
                filename="DIBCO2017_Dataset.7z",
                expected_size_bytes=43868529,
            ),
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z",
                filename="DIBCO2017_GT.7z",
                expected_size_bytes=892437,
            ),
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018_Dataset.zip",
                filename="dibco2018_Dataset.zip",
                expected_size_bytes=22311122,
            ),
            DatasetSource(
                kind="http",
                url="https://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018-GT.zip",
                filename="dibco2018-GT.zip",
                expected_size_bytes=3102047,
            ),
        ],
        license_note="Public competition benchmarks; cite the corresponding DIBCO/H-DIBCO reports.",
    ),
    "k_watermark": DatasetSpec(
        key="k_watermark",
        display_name="K-Watermark",
        stage="overlap_watermark_eval",
        access=DatasetAccess.MANUAL,
        description="Synthetic watermark text spotting benchmark generated from document pages.",
        expected_artifacts=["train/", "validation/", "test/"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://arxiv.org/abs/2401.05167",
                note="Paper confirmed, but no authoritative public dataset archive was located.",
            ),
        ],
    ),
    "mot_overlap": DatasetSpec(
        key="mot_overlap",
        display_name="Multi-scenario Overlapping Text (MOT)",
        stage="overlap_text_eval",
        access=DatasetAccess.MANUAL,
        description="1,250-image overlapping text segmentation benchmark spanning documents and scene text.",
        expected_artifacts=["images/", "masks/", "annotations/"],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Multi-scenario_Overlapping_Text_Segmentation_with_Depth_Awareness_ICCV_2025_paper.html",
                note="Official paper confirmed, but no authoritative public dataset archive was located.",
            ),
        ],
    ),
    "docres_references": DatasetSpec(
        key="docres_references",
        display_name="DocRes Referenced Benchmarks",
        stage="restoration_multitask_eval",
        access=DatasetAccess.MANUAL,
        description="An umbrella reference, not a single dataset: DocRes evaluates separate dewarping, deshadowing, enhancement, deblurring, and binarization corpora.",
        expected_artifacts=[],
        sources=[
            DatasetSource(
                kind="manual",
                url="https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_DocRes_A_Generalist_Model_Toward_Unifying_Document_Image_Restoration_Tasks_CVPR_2024_paper.html",
                note="Acquire each underlying benchmark from its original owner; do not treat DocRes as a dataset archive.",
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
        access=DatasetAccess.PUBLIC,
        description=(
            "Document figure extraction benchmark. Each row is a document page image "
            "with COCO-style bounding box annotations for embedded figures. "
            "Available as WKLI22/scanbank_hf on HuggingFace (MIT license). "
            "Columns: image_id, image, width, height, objects {area, bbox, category, id}."
        ),
        expected_artifacts=["data/"],
        sources=[
            DatasetSource(
                kind="huggingface",
                url="https://huggingface.co/datasets/WKLI22/scanbank_hf",
                allow_patterns=["data/*.parquet"],
                note="MIT license. ~564 MB. Train split: 10.1K rows. Test split: 102 rows.",
            ),
        ],
        license_note="MIT",
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
