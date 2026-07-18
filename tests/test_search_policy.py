"""
Unit tests for Node.subtree_size and Agent's UCB-based search policy.

These don't require any real LLM call — search_policy() and the UCB
scoring math are pure logic over the Journal/Node tree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.agent import Agent
from src.agent.journal import Journal
from src.agent.node import Node
from src.utils.config import Config


def make_cfg(debug_prob=0.0, num_drafts=1, exploration_constant=1.0):
    return Config(
        {
            "task_goal": "dummy task",
            "agent": {
                "search": {
                    "debug_prob": debug_prob,
                    "num_drafts": num_drafts,
                    "exploration_constant": exploration_constant,
                }
            },
        }
    )


def test_subtree_size_leaf_node_is_one():
    n = Node(code="x = 1")
    assert n.subtree_size == 1


def test_subtree_size_counts_all_descendants():
    root = Node(code="root")
    child_a = Node(code="a", parent=root)
    child_b = Node(code="b", parent=root)
    grandchild = Node(code="c", parent=child_a)

    assert root.subtree_size == 4  # root + a + b + c
    assert child_a.subtree_size == 2  # a + c
    assert child_b.subtree_size == 1


def test_node_value_is_higher_for_lower_mse():
    journal = Journal()
    agent = Agent(cfg=make_cfg(), journal=journal, llm=None)

    good_node = Node(code="x", is_buggy=False, metric=0.1)
    bad_node = Node(code="y", is_buggy=False, metric=10.0)

    assert agent._node_value(good_node) > agent._node_value(bad_node)


def test_search_policy_returns_none_before_enough_drafts():
    journal = Journal()
    agent = Agent(cfg=make_cfg(num_drafts=2), journal=journal, llm=None)
    journal.append(Node(code="draft1", is_buggy=False, metric=1.0))

    # Only 1 draft so far, need 2 -> should draft again.
    assert agent.search_policy() is None


def test_search_policy_debugs_buggy_leaf_when_debug_prob_is_one():
    journal = Journal()
    agent = Agent(cfg=make_cfg(debug_prob=1.0, num_drafts=1), journal=journal, llm=None)

    good = Node(code="good", is_buggy=False, metric=0.5)
    buggy = Node(code="buggy", is_buggy=True, metric=None)
    journal.append(good)
    journal.append(buggy)

    result = agent.search_policy()
    assert result is buggy


def test_search_policy_exploration_favors_underexplored_branch():
    """
    Two good nodes with similar metrics: one has been heavily improved
    already (large subtree), the other hasn't been touched. With a
    meaningful exploration_constant, UCB should be able to prefer the
    less-explored one over pure greedy "always pick best metric".
    """
    journal = Journal()
    agent = Agent(
        cfg=make_cfg(debug_prob=0.0, num_drafts=1, exploration_constant=5.0),
        journal=journal,
        llm=None,
    )

    # heavily_explored has a slightly better metric but a big subtree
    # (already explored a lot).
    heavily_explored = Node(code="a", is_buggy=False, metric=0.09)
    for _ in range(10):
        Node(code="child", is_buggy=False, metric=0.5, parent=heavily_explored)

    # barely_explored has a slightly worse metric but no children yet.
    barely_explored = Node(code="b", is_buggy=False, metric=0.1)

    journal.append(heavily_explored)
    for c in heavily_explored.children:
        journal.append(c)
    journal.append(barely_explored)

    chosen = agent.search_policy()
    assert chosen is barely_explored


def test_search_policy_greedy_when_exploration_constant_is_zero():
    """With exploration_constant=0, UCB degenerates to pure greedy — the
    best-metric node should always win regardless of subtree size."""
    journal = Journal()
    agent = Agent(
        cfg=make_cfg(debug_prob=0.0, num_drafts=1, exploration_constant=0.0),
        journal=journal,
        llm=None,
    )

    best = Node(code="best", is_buggy=False, metric=0.01)
    worse = Node(code="worse", is_buggy=False, metric=5.0)
    journal.append(best)
    journal.append(worse)

    assert agent.search_policy() is best
