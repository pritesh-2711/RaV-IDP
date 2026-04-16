"""Entity extractors."""

from .image import extract_image
from .table import extract_table
from .text import extract_text

__all__ = ["extract_image", "extract_table", "extract_text"]
