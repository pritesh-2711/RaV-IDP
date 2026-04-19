"""GPT fallback extractor."""

from __future__ import annotations

import base64
import json
import os

import pandas as pd

from ..config import get_settings
from ..models import DetectedRegion, EntityType, ExtractedEntity, ImageContent, TableContent, TextContent

SYSTEM_PROMPTS = {
    EntityType.TABLE: (
        "You are a precise document extraction assistant. "
        "Extract the complete table from the provided image. "
        "Return a JSON object only. No prose."
    ),
    EntityType.IMAGE: (
        "You are a precise document analysis assistant. "
        "Describe the figure in the provided image. "
        "Return a JSON object only. No prose."
    ),
    EntityType.TEXT: (
        "You are a precise OCR assistant. "
        "Transcribe all visible text in the provided image exactly as written. "
        "Preserve line breaks. Return a JSON object only. No prose."
    ),
}

USER_PROMPTS = {
    EntityType.TABLE: (
        'Extract the complete table. Return JSON with exactly these keys:\n'
        '  "headers": list of column header strings (in order)\n'
        '  "rows": list of lists — each inner list is one data row, values in column order\n'
        '  "notes": list of any footnotes, spanning cell annotations, or special markers\n\n'
        "Surrounding document context:\n{context}"
    ),
    EntityType.IMAGE: (
        'Describe this figure. Return JSON with exactly these keys:\n'
        '  "type": one of: photograph, chart, diagram, flowchart, logo, screenshot, table_as_image, other\n'
        '  "description": concise factual description\n'
        '  "extracted_text": all text visible within the image as a single string; empty string if none\n'
        '  "key_data_points": list of important numerical or categorical values visible in the image\n'
        '  "structured_data": for chart or diagram types only — object with keys "title", "axes", "data_points", "trend"; null for all other types\n'
        '  "document_intent": one sentence — what this figure illustrates in the context of the document\n\n'
        "Surrounding document context:\n{context}"
    ),
    EntityType.TEXT: (
        'Transcribe all text in the image exactly as written. Preserve line breaks. Return JSON with exactly one key:\n'
        '  "text": the full transcription as a single string\n\n'
        "Surrounding document context:\n{context}"
    ),
}


def _client():
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def _table_content_from_json(payload: dict) -> TableContent:
    headers = payload["headers"]
    rows = payload["rows"]
    dataframe = pd.DataFrame(rows, columns=headers)
    return TableContent(
        dataframe_json=dataframe.to_json(orient="split"),
        markdown=dataframe.to_markdown(index=False) if len(dataframe.columns) else "",
        csv=dataframe.to_csv(index=False),
        headers=[str(header) for header in dataframe.columns],
        row_count=len(dataframe),
        col_count=len(dataframe.columns),
    )


def _parse_fallback_response(payload: dict, region: DetectedRegion) -> ExtractedEntity:
    if region.entity_type == EntityType.TABLE:
        content = _table_content_from_json(payload)
    elif region.entity_type == EntityType.IMAGE:
        description = " ".join(
            filter(
                None,
                [
                    payload.get("description"),
                    "Key points: " + ", ".join(payload.get("key_data_points", [])) if payload.get("key_data_points") else "",
                    payload.get("document_intent"),
                ],
            )
        ).strip()
        extracted_text = payload.get("extracted_text") or None
        if isinstance(extracted_text, str) and not extracted_text.strip():
            extracted_text = None
        structured_data = payload.get("structured_data") or None
        if isinstance(structured_data, dict) and not any(structured_data.values()):
            structured_data = None
        content = ImageContent(
            crop_bytes=region.original_crop,
            classification_label=payload.get("type"),
            classification_confidence=None,
            image_type=payload.get("type"),
            description=description or None,
            extracted_text=extracted_text,
            structured_data=structured_data,
        )
    else:
        content = TextContent(text=payload["text"], urls=[])

    return ExtractedEntity(
        region_id=region.region_id,
        entity_type=region.entity_type,
        content=content,
        extractor_name="fallback",
    )


def call_vision_fallback(region: DetectedRegion, context_text: str = "") -> ExtractedEntity:
    """Call GPT vision fallback with structured prompts."""

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for fallback extraction.")

    image_b64 = base64.b64encode(region.original_crop).decode("utf-8")
    entity_type = region.entity_type if region.entity_type in SYSTEM_PROMPTS else EntityType.TEXT
    response = _client().chat.completions.create(
        model=settings.openai_model,
        max_tokens=int(os.getenv("OPENAI_VISION_MAX_TOKENS", str(settings.openai_vision_max_tokens))),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[entity_type]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
                    },
                    {"type": "text", "text": USER_PROMPTS[entity_type].format(context=context_text or "No surrounding context available.")},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return _parse_fallback_response(payload, region)
