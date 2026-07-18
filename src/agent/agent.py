"""
Agent: the core draft -> debug -> improve loop.

This is where most of the interesting "AI agent" design decisions live:
    - search_policy(): decide what to work on next (currently: mostly
      random between debug/improve/greedy-on-best). See TODO for a
      proposed MCTS/UCB-style replacement.
    - _draft / _improve / _debug: the three prompt-construction + LLM-call
      paths that produce new Nodes.
    - parse_exec_result(): turns raw execution output into a judgement
      (buggy? what's the metric?) via another LLM call.
"""

import random
from typing import Callable

from src.agent.journal import Journal
from src.agent.node import Node
from src.agent.schemas import ExecutionEvaluation
from src.interpreter.interpreter import ExecutionResult
from src.llm.backend import LLMBackend
from src.llm.structured import generate_structured
from src.utils.data_preview import data_preview_generate
from src.utils.text_processing import extract_code, extract_text_up_to_code, wrap_code

ExecCallbackType = Callable[[str, bool], ExecutionResult]


class Agent:
    def __init__(self, cfg, journal: Journal, llm: LLMBackend):
        self.cfg = cfg
        self.journal = journal
        self.llm = llm
        self.data_preview: str | None = None

    def search_policy(self) -> Node | None:
        """
        Select a node to work on (or None to draft a new node).

        TODO(search-strategy): this is currently a simple heuristic
        (random debug-vs-improve choice, then greedy on the best node).
        A stronger version of this project would replace it with a
        proper tree search, e.g.:
            - UCB1 score per node: metric_estimate + c * sqrt(ln(N) / n_i)
            - or MCTS with a rollout/simulation step
        which is a much better "algorithms" story for interviews than
        the current random policy.
        """
        search_cfg = self.cfg.agent.search

        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            return None

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [n for n in self.journal.buggy_nodes if n.is_leaf]
            if debuggable_nodes:
                return random.choice(debuggable_nodes)

        good_nodes = self.journal.good_nodes
        if not good_nodes:
            return None

        return self.journal.get_best_node()

    def plan_and_code_query(
        self, system_message: str, user_message: str, retries: int = 3
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in one LLM call, then split them."""
        completion_text = None
        for _ in range(retries):
            completion_text = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
            )
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code:
                return nl_text, code

        return "", completion_text

    def _draft(self) -> Node:
        system_prompt = "You are an AI agent."

        user_prompt = "\n".join(
            [
                "You have to come up with a solution for a machine learning task "
                "and then implement this solution in Python.",
                f"The task is to {self.cfg.task_goal}",
                f'All the provided input data is stored in "{self.cfg.data_dir}" directory.',
                f"{self.data_preview}",
                'You have to save the predictions result on testing set in "/content/submission.csv".',
                "Note that the testing file DOES NOT have the target column.",
            ]
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        system_prompt = "You are an AI assistant."

        user_prompt = " ".join(
            [
                f"Task description: {self.cfg.task_goal} "
                f"Memory: {self.journal.generate_summary()} "
                f"Previous solution: Code: {wrap_code(parent_node.code)} "
            ]
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _debug(self, parent_node: Node) -> Node:
        system_prompt = "You are an AI agent."

        user_prompt = " ".join(
            [
                f"Task description: {self.cfg.task_goal}\n\n",
                f"Previous (buggy) implementation: {wrap_code(parent_node.code)}\n\n",
                f"Execution output: {wrap_code(parent_node.term_out, lang='')}\n\n",
                str(self.data_preview),
            ]
        )

        plan, code = self.plan_and_code_query(system_prompt, user_prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(self) -> None:
        self.data_preview = data_preview_generate(self.cfg.data_dir)

    def step(self, exec_callback: ExecCallbackType) -> None:
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> None:
        """Judge whether the executed code is buggy and extract its metric.

        Uses a structured LLM call (see src/llm/structured.py) so the model
        returns a validated ExecutionEvaluation object instead of free text
        we'd have to regex out. The final is_buggy decision combines the
        LLM judgement with two hard signals we already have for free:
        node.exc_type is not None means the interpreter caught an
        exception, so it is unconditionally buggy regardless of what the
        LLM thinks; evaluation.metric is None means the LLM could not
        find or verify a metric, which we also treat as buggy.
        """
        node.absorb_exec_result(exec_result)

        system_prompt = (
            "You are an AI assistant that evaluates the output of a "
            "machine learning code execution. Judge honestly and "
            "conservatively: if you are not sure the code succeeded, or "
            "you cannot find a validation metric in the output, mark it "
            "as buggy."
        )
        user_prompt = (
            f"The task is:\n{self.cfg.task_goal}\n\n"
            f"The code implementation is:\n{wrap_code(node.code)}\n\n"
            f'The execution output is:\n{wrap_code(node.term_out, lang="")}'
        )

        try:
            evaluation = generate_structured(
                llm=self.llm,
                system_message=system_prompt,
                user_message=user_prompt,
                response_model=ExecutionEvaluation,
            )
        except ValueError as e:
            node.is_buggy = True
            node.metric = None
            node.analysis = f"Evaluation failed: {e}"
            return

        node.is_buggy = (
            node.exc_type is not None
            or evaluation.is_buggy
            or evaluation.metric is None
        )
        node.metric = evaluation.metric if not node.is_buggy else None
        node.analysis = evaluation.summary
