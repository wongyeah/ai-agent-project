"""
Node: a single point in the agent's solution search tree.

Each Node holds a plan + code, the result of executing that code, and the
agent's evaluation of whether the code is buggy / how good the metric is.
Nodes link to their parent, forming a tree of draft -> debug/improve steps.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin

from src.interpreter.interpreter import ExecutionResult
from src.utils.text_processing import trim_long_string


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)
    exec_time: float = field(default=None, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    analysis: str = field(default=None, kw_only=True)
    metric: float = field(default=None, kw_only=True)
    is_buggy: bool = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """Which stage produced this node."""
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult) -> None:
        """Store the result of executing this node's code."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Terminal output of the execution, truncated if very long."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Whether this node has no children yet."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """Length of the current consecutive-debugging chain."""
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1
