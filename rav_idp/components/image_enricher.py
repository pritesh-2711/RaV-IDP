"""Semantic enrichment for extracted image entities.

Called for every image entity after fidelity validation, regardless of
whether the extraction passed or failed. Populates ImageContent fields that
are opaque to downstream retrieval systems (description, extracted_text,
structured_data, image_type) using a vision model.

If OPENAI_API_KEY is not configured, the function returns the entity
unchanged without raising an error. This keeps the pipeline runnable
without an API key at the cost of unpopulated enrichment fields.
"""

from __future__ import annotations

import base64
import json

from ..config import get_settings
from ..models import EntityType, ExtractedEntity, ImageContent

_SYSTEM_PROMPT = (
    "You are a document analysis assistant specialising in semantic content extraction. "
    "You receive a single image region cropped from a document page. "
    "Return a JSON object only. No prose, no explanation."
)

_USER_PROMPT = (
    "Analyse this image and return JSON with exactly these keys:\n"
    '  "image_type": one of: photograph, chart, diagram, flowchart, logo, '
    "screenshot, table_as_image, other\n"
    '  "description": concise factual description of what the image shows\n'
    '  "extracted_text": all text visible within the image as a single string; '
    "empty string if none\n"
    '  "structured_data": for chart or diagram types only — an object with keys '
    '"title" (string or null), "axes" (object describing axis labels and ranges '
    'or null), "data_points" (list of notable values), "trend" (one-sentence '
    "summary of the trend or pattern shown); null for all other image types\n\n"
    "Document context (text surrounding this image in the source document):\n"
    "{context}"
)


def enrich_image(
    entity: ExtractedEntity,
    context_text: str = "",
) -> ExtractedEntity:
    """Enrich an image entity with semantic content via vision model.

    Returns the entity unchanged if:
    - entity type is not IMAGE
    - no crop bytes are present
    - OPENAI_API_KEY is not configured
    - the API call fails for any reason
    """
    if entity.entity_type != EntityType.IMAGE:
        return entity

    content: ImageContent = entity.content
    if not content.crop_bytes:
        return entity

    settings = get_settings()
    if not settings.openai_api_key:
        return entity

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        image_b64 = base64.b64encode(content.crop_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.openai_vision_max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": _USER_PROMPT.format(
                                context=context_text or "No surrounding context available."
                            ),
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
    except Exception:
        return entity

    extracted_text = payload.get("extracted_text") or None
    if isinstance(extracted_text, str) and not extracted_text.strip():
        extracted_text = None

    structured_data = payload.get("structured_data") or None
    if isinstance(structured_data, dict) and not any(structured_data.values()):
        structured_data = None

    enriched = content.model_copy(update={
        "image_type": payload.get("image_type") or content.classification_label,
        "description": payload.get("description") or content.description,
        "extracted_text": extracted_text,
        "structured_data": structured_data,
    })
    return entity.model_copy(update={"content": enriched})
