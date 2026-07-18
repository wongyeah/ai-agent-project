"""
Unit tests for src/llm/structured.py using a fake LLMBackend that returns
scripted responses — no real model or GPU required.
"""

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.backend import LLMBackend
from src.llm.structured import generate_structured


class ScriptedBackend(LLMBackend):
    """Returns each response in `responses` in order, one per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def generate_response(self, messages):
        self.call_count += 1
        return self._responses.pop(0)


class Animal(BaseModel):
    name: str
    legs: int


def test_generate_structured_parses_clean_json():
    backend = ScriptedBackend(['{"name": "dog", "legs": 4}'])
    result = generate_structured(backend, "sys", "user", Animal)
    assert result.name == "dog"
    assert result.legs == 4
    assert backend.call_count == 1


def test_generate_structured_handles_fenced_json():
    backend = ScriptedBackend(
        ['Sure, here you go:\n```json\n{"name": "cat", "legs": 4}\n```\nHope that helps!']
    )
    result = generate_structured(backend, "sys", "user", Animal)
    assert result.name == "cat"


def test_generate_structured_retries_on_bad_json_then_succeeds():
    backend = ScriptedBackend(
        [
            "not json at all",
            '{"name": "spider", "legs": 8}',
        ]
    )
    result = generate_structured(backend, "sys", "user", Animal, retries=3)
    assert result.name == "spider"
    assert result.legs == 8
    assert backend.call_count == 2


def test_generate_structured_retries_on_schema_violation_then_succeeds():
    backend = ScriptedBackend(
        [
            '{"name": "bird"}',  # missing required "legs" field
            '{"name": "bird", "legs": 2}',
        ]
    )
    result = generate_structured(backend, "sys", "user", Animal, retries=3)
    assert result.legs == 2


def test_generate_structured_raises_after_exhausting_retries():
    backend = ScriptedBackend(["nope", "still nope", "nope again"])
    with pytest.raises(ValueError):
        generate_structured(backend, "sys", "user", Animal, retries=3)
