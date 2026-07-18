"""
Entry point: wires together config, LLM backend, journal, agent, and
interpreter, then runs the draft -> debug -> improve loop for
`cfg.agent.steps` iterations.

Usage:
    python main.py --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from src.agent.agent import Agent
from src.agent.journal import Journal
from src.interpreter.interpreter import Interpreter
from src.llm.backend import LlamaCppBackend
from src.utils.config import Config, set_seed


def build_llm_backend(cfg: Config):
    backend = cfg.llm.backend
    if backend == "llama_cpp":
        return LlamaCppBackend(
            model_path=cfg.llm.model_path,
            n_gpu_layers=cfg.llm.n_gpu_layers,
            n_ctx=cfg.llm.n_ctx,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )
    # TODO(multi-backend): wire up OpenAIBackend / AnthropicBackend here
    # once implemented in src/llm/backend.py.
    raise ValueError(f"Unknown llm backend: {backend}")


def save_run(cfg: Config, journal: Journal, out_dir: str = "runs") -> None:
    """Persist the journal (and best solution) for this run to disk."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_path = Path(out_dir) / f"{cfg.exp_name}_journal.json"
    with open(run_path, "w") as f:
        json.dump(journal.to_dict(), f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = Config(raw_cfg)

    set_seed()

    llm = build_llm_backend(cfg)
    journal = Journal()
    agent = Agent(cfg=cfg, journal=journal, llm=llm)
    interpreter = Interpreter()

    def exec_callback(*call_args, **call_kwargs):
        return interpreter.run(*call_args, **call_kwargs)

    global_step = len(journal)
    while global_step < cfg.agent.steps:
        agent.step(exec_callback=exec_callback)
        save_run(cfg, journal)
        global_step = len(journal)

    interpreter.cleanup_session()

    best = journal.get_best_node()
    if best is not None:
        print(f"Best metric: {best.metric}")
    else:
        print("No successful (non-buggy) solution found.")


if __name__ == "__main__":
    main()
