# ML Coding Agent

An LLM-driven agent that autonomously drafts, executes, debugs, and
iteratively improves Python solutions to a machine learning task —
inspired by [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/pdf/2502.13138).

Originally built as a course assignment (NTU ML 2025 Spring, HW2); this repo
is a refactor of that assignment into a modular, backend-agnostic project,
used here as the basis for further AI-agent-focused improvements.

## What it does

Given a task description and a directory of data, the agent:

1. **Drafts** an initial plan + code solution via an LLM call.
2. **Executes** the code in an isolated subprocess (`Interpreter`), capturing
   stdout/stderr/exceptions with a timeout.
3. **Evaluates** the result — currently a stub (see `Agent.parse_exec_result`,
   flagged as the top TODO), intended to judge bugginess and extract a metric.
4. Based on the search policy, either **debugs** a buggy solution, **improves**
   the current best one, or starts a fresh **draft** — repeating for a
   configured number of steps.
5. Tracks every attempt in a `Journal` (a tree of `Node`s) and reports the
   best solution found.

## Project structure

```
ai-agent-project/
├── configs/
│   └── config.yaml       # task description, data dir, LLM + search settings
├── src/
│   ├── llm/backend.py     # LLM backend interface (local GGUF now; OpenAI/Anthropic stubs)
│   ├── agent/
│   │   ├── node.py         # a single solution attempt in the search tree
│   │   ├── journal.py      # the full search tree + accessors
│   │   └── agent.py        # draft / debug / improve loop + search policy
│   ├── interpreter/
│   │   └── interpreter.py  # sandboxed subprocess code execution
│   └── utils/
│       ├── text_processing.py  # code/JSON extraction from LLM output
│       ├── data_preview.py     # lightweight dataset summaries for prompts
│       └── config.py           # dict -> dot-access config object
├── main.py                 # CLI entry point
├── tests/                  # unit tests (no GPU/LLM required)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download a GGUF model (any model works, pick one from the
[Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
in GGUF format) and place its path in `configs/config.yaml` under `llm.model_path`.

Put your task's train/test CSVs under `data/` and point `data_dir` in the
config at that folder.

## Run

```bash
python main.py --config configs/config.yaml
```

## Test

```bash
python -m pytest tests/
```

## Known limitations / roadmap

- ✅ **Real evaluation** (`Agent.parse_exec_result`): now uses structured
  LLM output (Pydantic schema + JSON-mode prompting with validation
  retries, see `src/llm/structured.py`) instead of a hardcoded stub. Bug
  detection combines the LLM's judgement with a hard rule (an actual
  interpreter exception always overrides the LLM into "buggy").
- ✅ **Smarter search** (`Agent.search_policy`): node selection for
  "improve" now uses a UCB1-style score (`Agent._ucb_score`) balancing
  the node's metric against how much its branch has already been
  explored (`Node.subtree_size` as a visit-count proxy), instead of
  always greedily picking the single best metric seen so far.
- **Reflection step** (not yet done): add a "critic" LLM pass between
  drafting and execution to catch obvious issues before spending an
  execution budget.
- **Multi-backend LLM support**: `src/llm/backend.py` defines the
  interface and has commented-out stubs for OpenAI/Anthropic backends —
  wiring these up would let the agent run without a local GPU.
- **Sandbox hardening**: the `Interpreter` isolates execution in a
  subprocess but doesn't yet enforce memory/CPU/network limits; running
  it in a disposable container would be a more production-realistic
  execution sandbox.

## Credit

Core execution/search scaffolding adapted from the
[AIDE](https://arxiv.org/pdf/2502.13138) project, as provided in the
NTU ML 2025 Spring HW2 course template.
