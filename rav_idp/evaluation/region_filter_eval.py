"""Evaluate the spatial text-inside-image suppression filter.

Runs layout detection + preprocessing on all available test documents,
applies the containment filter, and reports before/after statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..components.layout_detector import detect_layout
from ..components.page_renderer import render_document_pages
from ..components.region_filter import _containment_ratio, suppress_text_inside_images
from ..components.region_preprocessor import preprocess_regions
from ..models import DetectedRegion, EntityType

_TEXT_TYPES = {EntityType.TEXT, EntityType.FORMULA, EntityType.URL}
_CONTAINER_TYPES = {EntityType.IMAGE, EntityType.TABLE}

_TEST_DOCS = [
    Path(__file__).parents[3] / "test_docs" / "MIT_Technology_Agentic_AI_in_Banking_1773060434.pdf",
    Path(__file__).parents[3] / "test_docs" / "embedded-images-tables.pdf",
    Path(__file__).parents[3] / "test_docs" / "MIT_pg358.pdf",
]


def _region_summary(region: DetectedRegion) -> dict:
    b = region.bbox
    return {
        "region_id": region.region_id,
        "type": region.entity_type.value,
        "page": region.page_index,
        "bbox": [round(b.x0), round(b.y0), round(b.x1), round(b.y1)],
    }


def _best_container(region: DetectedRegion, containers: list[DetectedRegion]) -> dict | None:
    best_ratio = 0.0
    best = None
    for c in containers:
        r = _containment_ratio(region.bbox, c.bbox)
        if r > best_ratio:
            best_ratio = r
            best = c
    if best is None:
        return None
    return {
        "container_id": best.region_id,
        "container_type": best.entity_type.value,
        "containment_ratio": round(best_ratio, 4),
    }


def run_eval(containment_threshold: float = 0.85) -> dict:
    all_suppressed: list[dict] = []
    doc_results: list[dict] = []

    for doc_path in _TEST_DOCS:
        if not doc_path.exists():
            print(f"  SKIP (not found): {doc_path.name}")
            continue
        print(f"\nProcessing: {doc_path.name}")
        page_records = render_document_pages(doc_path)
        regions = detect_layout(doc_path, page_records)
        regions = preprocess_regions(regions)

        n_before = len(regions)
        n_text_before = sum(1 for r in regions if r.entity_type in _TEXT_TYPES)
        n_image_before = sum(1 for r in regions if r.entity_type in _CONTAINER_TYPES)

        containers = [r for r in regions if r.entity_type in _CONTAINER_TYPES]
        suppressed_details: list[dict] = []
        for region in regions:
            if region.entity_type not in _TEXT_TYPES:
                continue
            info = _best_container(region, containers)
            if info and info["containment_ratio"] >= containment_threshold:
                entry = _region_summary(region)
                entry.update(info)
                entry["doc"] = doc_path.name
                suppressed_details.append(entry)

        filtered, suppressed_ids = suppress_text_inside_images(regions, containment_threshold)
        n_after = len(filtered)
        n_text_after = sum(1 for r in filtered if r.entity_type in _TEXT_TYPES)

        print(f"  Pages: {len(page_records)}")
        print(f"  Regions before: {n_before}  (text/formula/url={n_text_before}, image/table={n_image_before})")
        print(f"  Regions after:  {n_after}  (text/formula/url={n_text_after})")
        print(f"  Suppressed:     {len(suppressed_ids)}")

        # containment ratio distribution for suppressed
        ratios = [e["containment_ratio"] for e in suppressed_details]
        if ratios:
            print(f"  Containment ratios — min={min(ratios):.3f}  mean={sum(ratios)/len(ratios):.3f}  max={max(ratios):.3f}")

        all_suppressed.extend(suppressed_details)
        doc_results.append({
            "doc": doc_path.name,
            "pages": len(page_records),
            "regions_before": n_before,
            "regions_after": n_after,
            "text_before": n_text_before,
            "text_after": n_text_after,
            "image_table_count": n_image_before,
            "suppressed_count": len(suppressed_ids),
            "suppressed_ids": suppressed_ids,
        })

    total_before = sum(d["regions_before"] for d in doc_results)
    total_after = sum(d["regions_after"] for d in doc_results)
    total_text_before = sum(d["text_before"] for d in doc_results)
    total_text_after = sum(d["text_after"] for d in doc_results)
    total_suppressed = sum(d["suppressed_count"] for d in doc_results)
    total_pages = sum(d["pages"] for d in doc_results)

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print(f"  Documents:          {len(doc_results)}")
    print(f"  Pages:              {total_pages}")
    print(f"  Total regions before: {total_before}  (text={total_text_before})")
    print(f"  Total regions after:  {total_after}  (text={total_text_after})")
    print(f"  Total suppressed:     {total_suppressed}  ({100*total_suppressed/max(total_text_before,1):.1f}% of text regions)")
    if total_suppressed > 0:
        ratios_all = [e["containment_ratio"] for e in all_suppressed]
        print(f"  Containment ratios — min={min(ratios_all):.3f}  mean={sum(ratios_all)/len(ratios_all):.3f}  max={max(ratios_all):.3f}")

    print("\nSample suppressed regions (first 10):")
    for entry in all_suppressed[:10]:
        print(f"  [{entry['doc']} p{entry['page']}] {entry['region_id']}  ratio={entry['containment_ratio']}  inside {entry['container_id']} ({entry['container_type']})")

    result = {
        "containment_threshold": containment_threshold,
        "aggregate": {
            "documents": len(doc_results),
            "pages": total_pages,
            "regions_before": total_before,
            "regions_after": total_after,
            "text_regions_before": total_text_before,
            "text_regions_after": total_text_after,
            "suppressed_count": total_suppressed,
            "suppressed_pct_of_text": round(100 * total_suppressed / max(total_text_before, 1), 2),
        },
        "per_doc": doc_results,
        "suppressed_regions": all_suppressed,
    }

    out_path = Path(__file__).parents[2] / "artifacts" / "region_filter_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nArtifact: {out_path}")
    return result


if __name__ == "__main__":
    run_eval()
