"""
Unit tests for Agent._reflect_and_revise using a scripted fake LLM backend
— no real model call required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.journal import Journal
from src.agent.node import Node
from src.llm.backend import LLMBackend
from src.utils.config import Config


class ScriptedBackend(LLMBackend):
    """Returns each response in `responses` in order, one per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def generate_response(self, messages):
        self.call_count += 1
        return self._responses.pop(0)


def make_cfg(enabled=True, max_revisions=1):
    return Config(
        {
            "task_goal": "dummy task",
            "agent": {
                "reflection": {"enabled": enabled, "max_revisions": max_revisions},
                "search": {"debug_prob": 0.0, "num_drafts": 1, "exploration_constant": 1.0},
            },
        }
    )


def test_reflection_disabled_leaves_node_untouched():
    backend = ScriptedBackend([])  # should never be called
    journal = Journal()
    agent = Agent(cfg=make_cfg(enabled=False), journal=journal, llm=backend)

    node = Node(code="original code", plan="original plan")
    result = agent._reflect_and_revise(node)

    assert result.code == "original code"
    assert result.reflection is None
    assert backend.call_count == 0


def test_reflection_no_issues_keeps_code():
    backend = ScriptedBackend(['{"has_issues": false, "feedback": "Looks fine."}'])
    journal = Journal()
    agent = Agent(cfg=make_cfg(), journal=journal, llm=backend)

    node = Node(code="original code", plan="original plan")
    result = agent._reflect_and_revise(node)

    assert result.code == "original code"
    assert result.reflection == "Looks fine."
    assert backend.call_count == 1  # only the review call, no revision needed


def test_reflection_with_issues_triggers_revision():
    backend = ScriptedBackend(
        [
            '{"has_issues": true, "feedback": "Missing submission.csv output."}',
            "Here is the fix.\n```python\nimport pandas as pd\ndf.to_csv('/content/submission.csv')\n```",
        ]
    )
    journal = Journal()
    agent = Agent(cfg=make_cfg(), journal=journal, llm=backend)

    node = Node(code="original code", plan="original plan")
    result = agent._reflect_and_revise(node)

    assert "submission.csv" in result.code
    assert result.reflection == "Missing submission.csv output."
    assert backend.call_count == 2  # review call + revision call


def test_reflection_respects_max_revisions():
    """Even if the critic keeps finding issues, we stop after max_revisions rounds."""
    backend = ScriptedBackend(
        [
            '{"has_issues": true, "feedback": "issue 1"}',
            "```python\ncode_v2 = True\n```",
            '{"has_issues": true, "feedback": "issue 2"}',
            "```python\ncode_v3 = True\n```",
        ]
    )
    journal = Journal()
    agent = Agent(cfg=make_cfg(max_revisions=2), journal=journal, llm=backend)

    node = Node(code="original code", plan="original plan")
    result = agent._reflect_and_revise(node)

    assert "code_v3 = True" in result.code
    assert backend.call_count == 4  # 2 rounds x (review + revision)


def test_reflection_review_failure_does_not_crash():
    """If the critic's response can never be parsed, skip reflection gracefully."""
    backend = ScriptedBackend(["not json", "still not json", "nope"])
    journal = Journal()
    agent = Agent(cfg=make_cfg(), journal=journal, llm=backend)

    node = Node(code="original code", plan="original plan")
    result = agent._reflect_and_revise(node)

    assert result.code == "original code"  # unchanged, no crash
