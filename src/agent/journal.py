"""
Journal: the full collection of nodes explored by the agent, i.e. the
solution search tree flattened into a list, plus convenience accessors.
"""

from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import DataClassJsonMixin

from src.agent.node import Node


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    @classmethod
    def from_dict(cls, kvs: dict[str, Any], **kwargs) -> "Journal":
        """
        Reconstruct a Journal from a dict (e.g. json.load()'d from a file
        main.py's save_run() wrote), repairing the parent/children object
        graph as it goes.

        dataclasses_json's generated from_dict() builds each Node's
        `parent` by recursively constructing a *brand-new* Node object
        from the nested dict, rather than pointing it at the actual
        parent object elsewhere in `nodes` — so naively, two nodes that
        share a parent in the real tree would end up with two separate,
        non-identical Python copies of "the same" parent after a reload
        (their `.id` still matches, since Node.__eq__ compares by id, but
        object identity and — critically — the shared, mutated `.children`
        set do not carry over). `children` itself is excluded from JSON
        entirely (see node.py) since it's redundant with `parent` and
        naive round-tripping both together is what caused a
        RecursionError on any tree deeper than one level in the first
        place.

        This override does the default reconstruction, then walks the
        flat node list once to: (1) point every node's `.parent` at the
        real, shared Node object from `nodes` (matched by `.id`) instead
        of its disposable deserialized copy, and (2) rebuild each real
        parent's `.children` set. This restores exactly the object graph
        `Node.__post_init__` builds during normal (non-reload) operation,
        so `subtree_size`/`is_leaf`/`stage_name`/`debug_depth` — and
        therefore UCB1 node selection — are all correct on a reloaded
        Journal, not just a freshly-built in-memory one.
        """
        journal: Journal = super().from_dict(kvs, **kwargs)

        by_id: dict[str, Node] = {n.id: n for n in journal.nodes}
        for node in journal.nodes:
            node.children = set()
        for node in journal.nodes:
            if node.parent is not None:
                real_parent = by_id.get(node.parent.id)
                node.parent = real_parent
                if real_parent is not None:
                    real_parent.children.add(node)
        return journal

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Add a new node, recording its step index."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Nodes that are initial drafts (no parent)."""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Nodes the agent judged to be buggy."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Nodes the agent judged to be non-buggy."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[float]:
        """Metric values across all nodes, in insertion order."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good: bool = True) -> None | Node:
        """
        Return the best solution found so far.

        NOTE: this currently assumes lower metric is better (e.g. MSE).
        If you generalize this project beyond a single-metric regression
        task, make "better" direction configurable instead of hardcoding
        `min`.
        """
        nodes = self.good_nodes if only_good else self.nodes
        if only_good and not nodes:
            return None
        return min(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a text summary of good nodes, for use in agent prompts."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric (Mean Squared Error): {n.metric}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)
