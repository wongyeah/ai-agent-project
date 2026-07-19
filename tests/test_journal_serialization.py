"""
Regression test for a RecursionError bug found while running real
multi-step search experiments: Journal.to_dict() (what main.py's
save_run() calls after EVERY step) used to recurse forever
(parent -> children -> parent -> children -> ...) the moment the tree
had any parent/child pair -- i.e. from the very first debug/improve node
onward. See the comments on Node.__post_init__ (src/agent/node.py) and
Journal.from_dict (src/agent/journal.py) for the root cause and fix.

Without this test, the bug is easy to miss: every unit test that
constructs a Node directly (tests/test_search_policy.py, etc.) never
calls journal.to_dict(), and the only config.yaml shipped with the
project runs agent.steps=1 / num_drafts=1 by default, which never
produces a second-level node either -- so save_run() was never actually
exercised past a single draft node before this test existed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.journal import Journal
from src.agent.node import Node


def _build_sample_tree() -> Journal:
    """draft -> (buggy) -> debug -> (good) -> improve, a 3-node tree with
    real parent/child links, mirroring what a real multi-step run
    produces."""
    journal = Journal()

    draft = Node(code="raise ValueError('oops')", plan="first attempt")
    journal.append(draft)
    draft.is_buggy = True
    draft.metric = None

    fix = Node(code="print('fixed')", plan="fix the bug", parent=draft)
    journal.append(fix)
    fix.is_buggy = False
    fix.metric = 0.42

    improved = Node(code="print('better')", plan="tune it", parent=fix)
    journal.append(improved)
    improved.is_buggy = False
    improved.metric = 0.10

    return journal


def test_to_dict_does_not_recurse_on_a_multi_level_tree():
    journal = _build_sample_tree()
    # This is the exact call main.py's save_run() makes after every step.
    # It used to RecursionError here on any tree with a parent/child pair.
    d = journal.to_dict()
    # Must be actually JSON-serializable too (save_run() also does this).
    s = json.dumps(d, default=str)
    assert len(s) > 0


def test_round_trip_preserves_tree_structure_and_derived_properties():
    journal = _build_sample_tree()
    reloaded = Journal.from_dict(json.loads(json.dumps(journal.to_dict(), default=str)))

    assert len(reloaded.nodes) == 3
    draft, fix, improved = reloaded.nodes

    # Object identity must be shared across the flat node list (not each
    # node holding its own disposable deserialized copy of its parent) --
    # this is what makes subtree_size/is_leaf/stage_name correct again.
    assert fix.parent is draft
    assert improved.parent is fix

    assert draft.subtree_size == 3
    assert draft.is_leaf is False
    assert improved.is_leaf is True
    assert improved.subtree_size == 1

    assert draft.stage_name == "draft"
    assert fix.stage_name == "debug"  # draft.is_buggy was True
    assert improved.stage_name == "improve"  # fix.is_buggy was False

    assert reloaded.get_best_node().metric == 0.10
