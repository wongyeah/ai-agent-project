"""
Run the "one-shot baseline" control-group experiment: a SINGLE LLM call
that drafts + codes a solution once, executed once, with NO multi-round
UCB search/iteration/reflection — for direct comparison against a real
`python main.py --config <same config>` search run's best-so-far metric.

This reuses the project's real Agent._draft() (same prompt-construction
the search loop's very first draft uses) and Agent.parse_exec_result()
(same structured LLM-judged evaluation the search loop uses for every
node) — so the ONLY difference from a real search run is that this script
calls step-equivalent logic exactly once and never iterates. That keeps
the comparison fair: any metric gap between this baseline and a real
search run's best node is attributable to the search/iteration process
itself, not to a differently-written prompt or a different evaluation
method.

Usage:
    python scripts/run_one_shot_baseline.py --config configs/eval_california_housing.yaml

Writes a JSON file to runs/<exp_name>_baseline.json with the same shape
scripts/plot_search_trajectory.py expects.
"""

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Make `src` importable when run as `python scripts/run_one_shot_baseline.py`
# from the project root.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import build_llm_backend  # noqa: E402
from src.agent.agent import Agent  # noqa: E402
from src.agent.journal import Journal  # noqa: E402
from src.interpreter.interpreter import Interpreter  # noqa: E402
from src.utils.config import Config, set_seed  # noqa: E402


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="runs")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = Config(raw_cfg)

    set_seed()

    llm = build_llm_backend(cfg)
    journal = Journal()  # empty — this run never accumulates a search tree
    agent = Agent(cfg=cfg, journal=journal, llm=llm)
    interpreter = Interpreter(
        timeout=cfg.interpreter.timeout,
        max_memory_mb=cfg.interpreter.max_memory_mb,
        max_cpu_seconds=cfg.interpreter.max_cpu_seconds,
        block_network=cfg.interpreter.block_network,
    )

    agent.update_data_preview()
    node = agent._draft()  # ONE LLM call, no reflection, no search loop
    exec_result = interpreter.run(node.code, reset_session=True)
    interpreter.cleanup_session()
    agent.parse_exec_result(node=node, exec_result=exec_result)

    print(f"is_buggy={node.is_buggy}  metric={node.metric}")
    print(f"analysis: {node.analysis}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.exp_name}_baseline.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "mse": node.metric,
                "is_buggy": node.is_buggy,
                "exec_time": node.exec_time,
                "plan": node.plan,
                "code": node.code,
                "analysis": node.analysis,
                "description": (
                    "One-shot baseline: a single Agent._draft() + one "
                    "execution + one Agent.parse_exec_result() call, no "
                    "multi-round UCB search/iteration/reflection."
                ),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved baseline result to {out_path}")


if __name__ == "__main__":
    main()
