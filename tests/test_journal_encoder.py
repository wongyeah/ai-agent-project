"""
Unit tests for JournalJSONEncoder (src/utils/journal_encoder.py), the
explicit replacement for the old `json.dump(..., default=str)`.

Covers the three claims made in that module's docstring: (1) today's real
Node/Journal shapes never hit the fallback path at all, (2) numpy scalar
types -- the concrete "metric came back from sklearn, not a plain float"
scenario -- get converted exactly rather than stringified, and (3)
genuinely unknown types still serialize successfully but log a warning
instead of failing silently.
"""

import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.journal import Journal
from src.agent.node import Node
from src.utils.journal_encoder import JournalJSONEncoder


def _build_realistic_tree() -> Journal:
    """A draft -> debug -> improve tree shaped like a real run's output,
    including exec results with the exact exc_info/exc_stack shapes
    src/interpreter/interpreter.py's exception_summary() produces."""
    journal = Journal()

    draft = Node(code="raise ValueError('bad')", plan="first attempt")
    journal.append(draft)
    draft.is_buggy = True
    draft.metric = None
    draft.exc_type = "ValueError"
    draft.exc_info = {"args": ["bad"]}
    draft.exc_stack = [("runfile.py", 1, "<module>", "raise ValueError('bad')")]
    draft._term_out = ["Traceback ...\nValueError: bad\n"]

    fix = Node(code="print('VALIDATION_MSE=1.5')", plan="fix it", parent=draft)
    journal.append(fix)
    fix.is_buggy = False
    fix.metric = 1.5
    fix._term_out = ["VALIDATION_MSE=1.5\n"]

    return journal


def test_current_node_shapes_never_hit_the_fallback(caplog):
    """
    Documents (and enforces) the "nothing in this codebase currently
    needs the fallback path" claim in the module docstring. If a future
    change makes this false, this test fails loudly instead of the type
    corruption happening silently in production.
    """
    journal = _build_realistic_tree()
    with caplog.at_level(logging.WARNING, logger="src.utils.journal_encoder"):
        serialized = json.dumps(journal.to_dict(), cls=JournalJSONEncoder)
    assert caplog.records == []
    assert json.loads(serialized)  # round-trips as valid JSON


def test_numpy_float64_metric_round_trips_as_a_real_float_not_a_string():
    """
    The concrete failure mode this class exists to guard against:
    sklearn's mean_squared_error() returns numpy.float64, not a plain
    Python float. default=str would have silently turned this into a
    JSON string; this encoder must instead extract the native value.
    """
    journal = Journal()
    node = Node(code="print(1)", plan="p")
    journal.append(node)
    node.is_buggy = False
    node.metric = np.float64(1903868428.9049313)  # simulates an unwrapped sklearn return

    serialized = json.dumps(journal.to_dict(), cls=JournalJSONEncoder)
    raw = json.loads(serialized)
    # It must be a JSON number, not a string, in the raw serialized form.
    assert isinstance(raw["nodes"][0]["metric"], float)

    reloaded = Journal.from_dict(json.loads(serialized))
    assert isinstance(reloaded.nodes[0].metric, float)
    assert reloaded.nodes[0].metric == pytest.approx(1903868428.9049313)
    # And it must actually be usable in the numeric comparisons the rest
    # of the codebase relies on (Journal.get_best_node(), Agent._node_value()).
    assert reloaded.get_best_node().metric < 1903868429


def test_decimal_is_converted_to_a_real_float():
    journal = Journal()
    node = Node(code="print(1)", plan="p")
    journal.append(node)
    node.is_buggy = False
    node.metric = Decimal("2.5")

    serialized = json.dumps(journal.to_dict(), cls=JournalJSONEncoder)
    raw = json.loads(serialized)
    assert isinstance(raw["nodes"][0]["metric"], float)
    assert raw["nodes"][0]["metric"] == 2.5


def test_genuinely_unknown_type_still_serializes_but_logs_a_warning(caplog):
    class Mystery:
        def __repr__(self):
            return "<Mystery obj>"

    journal = Journal()
    node = Node(code="print(1)", plan="p")
    journal.append(node)
    node.analysis = Mystery()  # a field with no defined "correct" numeric type

    with caplog.at_level(logging.WARNING, logger="src.utils.journal_encoder"):
        serialized = json.dumps(journal.to_dict(), cls=JournalJSONEncoder)

    # Must not crash, and must produce valid, loadable JSON...
    raw = json.loads(serialized)
    assert raw["nodes"][0]["analysis"] == "<Mystery obj>"
    # ...but MUST be observable, unlike a silent default=str fallback.
    assert len(caplog.records) == 1
    assert "Mystery" in caplog.records[0].message
