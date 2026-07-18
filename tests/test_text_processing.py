"""
Basic unit tests for the parsing utilities.

These don't require a GPU or LLM backend, so they're a good place to
start a CI pipeline (e.g. GitHub Actions) for this project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.text_processing import (
    extract_code,
    extract_text_up_to_code,
    is_valid_python_script,
    trim_long_string,
    wrap_code,
)


def test_wrap_code():
    assert wrap_code("print(1)") == "```python\nprint(1)\n```"


def test_is_valid_python_script():
    assert is_valid_python_script("x = 1 + 1") is True
    assert is_valid_python_script("x = ") is False


def test_extract_code_from_fenced_block():
    text = "Here's a plan.\n\n```python\nx = 1\nprint(x)\n```\n"
    code = extract_code(text)
    assert "x = 1" in code
    assert "print(x)" in code


def test_extract_code_no_fence_falls_back_to_whole_text():
    text = "x = 1\nprint(x)"
    code = extract_code(text)
    assert "x = 1" in code


def test_extract_text_up_to_code():
    text = "Plan: do the thing.\n```python\nx = 1\n```"
    plan = extract_text_up_to_code(text)
    assert plan == "Plan: do the thing."


def test_trim_long_string():
    s = "a" * 10000
    trimmed = trim_long_string(s, threshold=100, k=10)
    assert "characters truncated" in trimmed
    assert len(trimmed) < len(s)


def test_trim_long_string_below_threshold_unchanged():
    s = "short string"
    assert trim_long_string(s, threshold=100, k=10) == s
