"""
Entry point: wires together config, LLM backend, journal, agent, and
interpreter, then runs the draft -> debug -> improve loop for
`cfg.agent.steps` iterations.

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --fresh   # ignore any existing checkpoint
"""

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agent.agent import Agent
from src.agent.journal import Journal
from src.interpreter.interpreter import Interpreter
from src.llm.backend import AnthropicBackend, CozeBackend, LlamaCppBackend, OpenAIBackend
from src.utils.config import Config, require_llm_config, set_seed
from src.utils.journal_encoder import JournalJSONEncoder


def build_llm_backend(cfg: Config):
    # Fails fast (before constructing any client / spending an API call)
    # if this backend is missing a field it actually needs -- see
    # require_llm_config()'s docstring in src/utils/config.py for why
    # this lives here instead of on LLMConfig itself.
    require_llm_config(cfg.llm)

    backend = cfg.llm.backend
    if backend == "llama_cpp":
        return LlamaCppBackend(
            model_path=cfg.llm.model_path,
            n_gpu_layers=cfg.llm.n_gpu_layers,
            n_ctx=cfg.llm.n_ctx,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )
    if backend == "openai":
        return OpenAIBackend(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )
    if backend == "anthropic":
        return AnthropicBackend(
            model=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )
    if backend == "coze":
        # llm.model doubles as the Coze bot_id here (Coze has no separate
        # "model name" concept from the caller's side — see
        # CozeBackend's docstring in src/llm/backend.py).
        return CozeBackend(
            bot_id=cfg.llm.model,
            max_tokens=cfg.llm.max_tokens,
            temperature=cfg.llm.temperature,
        )
    raise ValueError(f"Unknown llm backend: {backend}")


def _journal_path(exp_name: str, out_dir: str) -> Path:
    return Path(out_dir) / f"{exp_name}_journal.json"


def _meta_path(exp_name: str, out_dir: str) -> Path:
    return Path(out_dir) / f"{exp_name}_meta.json"


def save_run(cfg: Config, journal: Journal, out_dir: str = "runs") -> None:
    """
    Persist the journal (and a small config fingerprint) for this run to
    disk, so an interrupted/crashed run can be resumed instead of
    re-spending LLM calls (and money, once a real paid backend is wired
    up) on steps that already completed.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # cls=JournalJSONEncoder instead of default=str: see
    # src/utils/journal_encoder.py's docstring for why the blanket
    # "stringify anything json doesn't recognize" fallback was replaced
    # with an explicit, logged one.
    with open(_journal_path(cfg.exp_name, out_dir), "w") as f:
        json.dump(journal.to_dict(), f, indent=2, cls=JournalJSONEncoder)

    # A tiny sidecar recording what task this checkpoint belongs to. Not
    # required to resume (the journal file alone is enough), but lets
    # load_or_create_journal() warn you if you point --config at a
    # different task while reusing the same exp_name, instead of
    # silently resuming UCB1 search state built for a different problem.
    with open(_meta_path(cfg.exp_name, out_dir), "w") as f:
        json.dump({"data_dir": cfg.data_dir, "task_goal": cfg.task_goal}, f, indent=2)


def load_or_create_journal(cfg: Config, out_dir: str = "runs") -> tuple[Journal, bool]:
    """
    Load a previous run's Journal from disk if one exists for this
    exp_name, so main() can pick up where a previous (possibly crashed
    or manually interrupted) run left off.

    Note there's no separate "steps completed" counter to load — main()'s
    own `len(journal)` already IS that count (it's how the resumed run
    knows how many more steps it still needs), so the only state that
    ever needed persisting was the journal itself.

    Returns (journal, resumed) so the caller can log whether this is a
    fresh start or a resume, and warn on a probable exp_name/task
    mismatch (see save_run()'s sidecar meta file).
    """
    journal_path = _journal_path(cfg.exp_name, out_dir)
    if not journal_path.exists():
        return Journal(), False

    with open(journal_path) as f:
        journal = Journal.from_dict(json.load(f))

    meta_path = _meta_path(cfg.exp_name, out_dir)
    if meta_path.exists():
        with open(meta_path) as f:
            old_meta = json.load(f)
        if old_meta.get("data_dir") != cfg.data_dir or old_meta.get("task_goal") != cfg.task_goal:
            print(
                f"WARNING: resuming '{cfg.exp_name}' from {journal_path}, but its "
                "recorded data_dir/task_goal don't match the config you just "
                "loaded. You're probably resuming state built for a different "
                "task under the same exp_name. Pass --fresh to start over, or "
                "give this run a different exp_name if it's genuinely unrelated."
            )
    # else: meta sidecar predates this feature (e.g. a journal checked in
    # before save_run() started writing it) — nothing to compare against,
    # so resume silently rather than raising a false-positive warning.

    return journal, True


def main():
    # Loads variables from a local .env file (if present) into the process
    # environment — e.g. OPENAI_API_KEY. Never overwrites a variable that's
    # already set in the shell, so `export`/`$env:` still takes priority.
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint for this exp_name and start a brand-new run "
        "(overwrites the old journal/meta files on the first save).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = Config(raw_cfg)

    set_seed()

    if args.fresh:
        journal, resumed = Journal(), False
    else:
        journal, resumed = load_or_create_journal(cfg)

    if resumed:
        print(f"Resumed {len(journal)} previous step(s) for '{cfg.exp_name}' from runs/{cfg.exp_name}_journal.json")
        if len(journal) >= cfg.agent.steps:
            print(
                f"Already have {len(journal)} step(s) (>= configured agent.steps={cfg.agent.steps}); "
                "nothing left to do. Raise agent.steps in the config if you want more, "
                "or pass --fresh to start over."
            )
    else:
        print(f"Starting a new run for '{cfg.exp_name}' (no existing checkpoint found).")

    llm = build_llm_backend(cfg)
    agent = Agent(cfg=cfg, journal=journal, llm=llm)
    interpreter = Interpreter(
        timeout=cfg.interpreter.timeout,
        max_memory_mb=cfg.interpreter.max_memory_mb,
        max_cpu_seconds=cfg.interpreter.max_cpu_seconds,
        block_network=cfg.interpreter.block_network,
    )

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
