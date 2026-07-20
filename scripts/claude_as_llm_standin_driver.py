"""
Driver library for the "Claude-as-LLM stand-in" experiment.

PROVIDED FOR TRANSPARENCY, NOT PART OF THE NORMAL PIPELINE: this is the
actual script used to produce runs/eval_california_housing_journal.json
and the results documented in README.md's "Evaluation results" section,
kept here so the methodology is fully inspectable/reproducible in spirit
(even though a real run would use `python main.py --config ...` instead
-- see "Reproducing this locally" in README.md for that normal path).

This deliberately reuses the project's REAL Journal/Node/Agent(search
logic)/Interpreter classes unmodified -- the only thing being substituted
for this run is the LLM call itself (Agent.plan_and_code_query /
parse_exec_result would normally call an OpenAI/Anthropic/Coze API; here
Claude authors the code and judges the execution result directly, in the
same conversation turn, instead of that API call happening). The search
policy (UCB1 node selection, draft/debug/improve branching via
Agent.search_policy()) is the project's real, un-modified code, run for
real against real execution results from the real Interpreter sandbox.

Journal state is persisted to JOURNAL_PATH between steps (each step is a
separate script invocation) using the project's own Node/Journal
DataClassJsonMixin (to_dict/from_dict) -- so the on-disk file is bit-for-
bit the same format main.py's save_run() produces.

This was run with PROJECT_ROOT checked out at /tmp/aiagentproject/ai-agent-project
and a scratch journal dir at /tmp/standin_exp -- both paths below are left
as-used for an accurate record; adjust them if you actually invoke this
file rather than just reading it.
"""

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import Agent  # noqa: E402
from src.agent.journal import Journal  # noqa: E402
from src.agent.node import Node  # noqa: E402
from src.interpreter.interpreter import Interpreter  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.journal_encoder import JournalJSONEncoder  # noqa: E402

JOURNAL_PATH = PROJECT_ROOT / "runs" / "eval_california_housing_journal.json"
DATA_DIR = PROJECT_ROOT / "data" / "california-housing"

# Loaded from the real config file (rather than a hand-rolled partial
# dict) for two reasons: (1) Config is now a strict, validated schema
# with task_goal/data_dir/llm/interpreter as real fields -- a
# hand-rolled dict with only `agent.search` set would either fail
# validation (task_goal has no default) or silently drift out of sync
# with what configs/eval_california_housing.yaml actually specifies;
# (2) this also removes a duplication risk that existed before: the old
# hand-rolled dict's search params had to be kept in sync with that YAML
# file BY HAND, with no guarantee they stayed identical. Loading the file
# directly makes that impossible to get out of sync.
with open(PROJECT_ROOT / "configs" / "eval_california_housing.yaml") as f:
    SEARCH_CFG = Config(yaml.safe_load(f))


def load_journal() -> Journal:
    if JOURNAL_PATH.exists():
        return Journal.from_dict(json.loads(JOURNAL_PATH.read_text()))
    return Journal()


def save_journal(journal: Journal) -> None:
    JOURNAL_PATH.write_text(json.dumps(journal.to_dict(), indent=2, cls=JournalJSONEncoder))


def get_agent(journal: Journal) -> Agent:
    # llm=None is fine: search_policy() never touches self.llm, only the
    # plan_and_code_query/_draft/_improve/_debug/parse_exec_result methods
    # do (all of which are being replaced by Claude authoring content
    # directly in this stand-in run, so they're never called here).
    return Agent(cfg=SEARCH_CFG, journal=journal, llm=None)


def decide_next() -> None:
    """
    Print what the REAL search_policy() wants to happen next, plus enough
    context (parent code + exec output) to write the next step by hand.
    """
    journal = load_journal()
    agent = get_agent(journal)

    print(f"=== journal has {len(journal)} nodes so far ===")
    for n in journal.nodes:
        tag = "BUGGY" if n.is_buggy else "ok"
        print(f"  step {n.step} | id={n.id[:8]} | stage={n.stage_name} | {tag} | metric={n.metric}")

    if len(journal.draft_nodes) < SEARCH_CFG.agent.search.num_drafts:
        print(f"\n>>> DECISION: draft a NEW fresh solution "
              f"(draft {len(journal.draft_nodes) + 1} of {SEARCH_CFG.agent.search.num_drafts})")
        return

    action = agent.search_policy()
    if action is None:
        print("\n>>> DECISION: draft a NEW fresh solution (no good nodes yet / policy fallback)")
        return

    if action.is_buggy:
        print(f"\n>>> DECISION: DEBUG node {action.id[:8]} (step {action.step})")
    else:
        print(f"\n>>> DECISION: IMPROVE node {action.id[:8]} (step {action.step}, "
              f"metric={action.metric})")

    print(f"\n--- parent node {action.id[:8]} plan ---\n{action.plan}")
    print(f"\n--- parent node {action.id[:8]} code ---\n{action.code}")
    print(f"\n--- parent node {action.id[:8]} exec output (last 3000 chars) ---")
    print(action.term_out[-3000:])
    print(f"\n--- parent node {action.id[:8]} exc_type: {action.exc_type} ---")


def run_and_record_step(
    code: str,
    plan: str,
    parent_id: str | None,
    is_buggy: bool,
    metric: float | None,
    analysis: str,
    reflection: str | None = None,
) -> None:
    """
    Execute `code` for real via the project's Interpreter (real sandboxed
    subprocess, real resource limits), then append a real Node to the
    journal with the given evaluation (is_buggy/metric/analysis --
    Claude's judgement of the just-printed execution output, standing in
    for what Agent.parse_exec_result's LLM call would normally decide).
    """
    journal = load_journal()
    parent = None
    if parent_id is not None:
        parent = next(n for n in journal.nodes if n.id.startswith(parent_id))

    # NOTE: max_memory_mb must comfortably exceed what THIS driver process
    # (which imports torch transitively via src.utils.config) already has
    # mapped at fork() time -- ~3.2GB observed. Same caveat their own
    # tests/test_interpreter.py docstring calls out for the POSIX RLIMIT_AS
    # path. 2048 caused every step to fail to even start (child crash-
    # loops on memory allocation before it can send "state:ready",
    # manifesting as an opaque 10s "REPL child process failed to start
    # execution" timeout) until raised.
    interpreter = Interpreter(timeout=120, block_network=True, max_memory_mb=6144)
    result = interpreter.run(code, reset_session=True)
    interpreter.cleanup_session()

    print("=== EXECUTION OUTPUT ===")
    print("".join(result.term_out))
    print(f"=== exc_type: {result.exc_type} | exec_time: {result.exec_time:.2f}s ===")

    node = Node(code=code, plan=plan, parent=parent, reflection=reflection)
    node.absorb_exec_result(result)
    # Hard rule matching Agent.parse_exec_result's real behavior: an actual
    # interpreter exception always overrides into "buggy", regardless of
    # what's passed in.
    node.is_buggy = is_buggy or (result.exc_type is not None)
    node.metric = metric if not node.is_buggy else None
    node.analysis = analysis

    journal.append(node)
    save_journal(journal)
    print(f"\n=== RECORDED node {node.id[:8]} as step {node.step} "
          f"(is_buggy={node.is_buggy}, metric={node.metric}) ===")


def step(code: str, plan: str, parent_id: str | None, analysis: str, reflection: str | None = None) -> None:
    """
    One-shot version of run_and_record_step + finalize_last_node: expects
    the code to print a line like "VALIDATION_MSE=<float>" on success.
    Parses that automatically, so the whole step (execute, judge bugginess
    from the real exc_type, extract the metric, append+save the node) is
    a single call given code that's already known-working. Falls back to
    the two-step run_and_record_step/finalize_last_node flow (still
    defined above) for cases worth inspecting output before judging, e.g.
    a suspected-buggy debug attempt.
    """
    import re

    journal = load_journal()
    parent = None
    if parent_id is not None:
        parent = next(n for n in journal.nodes if n.id.startswith(parent_id))

    interpreter = Interpreter(timeout=120, block_network=True, max_memory_mb=6144)
    result = interpreter.run(code, reset_session=True)
    interpreter.cleanup_session()

    print("=== EXECUTION OUTPUT ===")
    print("".join(result.term_out))
    print(f"=== exc_type: {result.exc_type} | exec_time: {result.exec_time:.2f}s ===")

    node = Node(code=code, plan=plan, parent=parent)
    node.absorb_exec_result(result)
    node.is_buggy = result.exc_type is not None

    metric = None
    if not node.is_buggy:
        m = re.search(r"VALIDATION_MSE=([0-9.eE+-]+)", node.term_out)
        if m:
            metric = float(m.group(1))
        else:
            node.is_buggy = True  # couldn't find a metric -> treat as buggy, matches
            # Agent.parse_exec_result's real behavior of failing safe.
    node.metric = metric
    node.analysis = analysis

    journal.append(node)
    save_journal(journal)
    print(f"\n=== RECORDED node {node.id[:8]} as step {node.step} "
          f"(is_buggy={node.is_buggy}, metric={node.metric}) ===")


def finalize_last_node(metric: float | None, analysis: str, is_buggy: bool | None = None) -> None:
    """
    Patch the just-recorded (last) node's metric/analysis now that the
    real execution output has been read and judged -- mirrors the second
    half of what Agent.parse_exec_result's LLM call would do in the real
    pipeline (the first half, catching an outright interpreter exception,
    already happened automatically in run_and_record_step).
    """
    journal = load_journal()
    node = journal.nodes[-1]
    if is_buggy is not None:
        node.is_buggy = is_buggy or (node.exc_type is not None)
    node.metric = metric if not node.is_buggy else None
    node.analysis = analysis
    save_journal(journal)
    print(f"=== FINALIZED node {node.id[:8]} (step {node.step}): "
          f"is_buggy={node.is_buggy}, metric={node.metric} ===")
