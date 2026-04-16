from __future__ import annotations

from rav_idp.evaluation.stage3c_text import _normalize_text, _word_error_rate


def test_normalize_text_collapses_whitespace() -> None:
    assert _normalize_text(" A   B\nC ") == "A B C"


def test_word_error_rate_exact_match() -> None:
    assert _word_error_rate("hello world", "hello world") == 0.0
