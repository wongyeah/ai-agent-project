"""
Journal: the full collection of nodes explored by the agent, i.e. the
solution search tree flattened into a list, plus convenience accessors.
"""

from dataclasses import dataclass, field

from dataclasses_json import DataClassJsonMixin

from src.agent.node import Node


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

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
