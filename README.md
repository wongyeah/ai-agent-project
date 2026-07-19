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
3. **Evaluates** the result via a structured LLM call (`Agent.parse_exec_result`),
   judging bugginess and extracting a metric — combined with a hard rule
   that any real interpreter exception always counts as buggy.
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
│   ├── llm/backend.py     # LLM backend interface (OpenAI by default; local GGUF supported too)
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

By default this project uses the OpenAI API (`configs/config.yaml` ->
`llm.backend: "openai"`). Set your API key by copying `.env.example` to
`.env` and filling in your real key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`.env` is gitignored — it will never be committed. `main.py` loads it
automatically at startup via `python-dotenv`. (You can also set the
environment variable directly in your shell instead if you prefer; either
way works, and a shell-level variable takes priority over `.env`.)

If you'd rather run a local model instead (e.g. no API budget, or want to
experiment offline), set `llm.backend: "llama_cpp"` in the config, install
`llama-cpp-python`, download a GGUF model (any model works — pick one from
the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
in GGUF format), and point `llm.model_path` at it.

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
- ✅ **Reflection step**: a "critic" LLM pass (`Agent._reflect_and_revise`)
  reviews the plan + code for obvious problems before execution, and can
  trigger up to `reflection.max_revisions` rounds of self-revision — see
  `src/agent/schemas.py::CodeReview`. Fails open (skips reflection rather
  than crashing) if the critic's own response can't be parsed.
- ✅ **Multi-backend LLM support** (partial): `src/llm/backend.py` now has
  a working `OpenAIBackend` (the default) alongside the local
  `LlamaCppBackend`. An `AnthropicBackend` stub is sketched in comments
  for anyone wanting to add it.
- ✅ **Sandbox hardening**: the `Interpreter` now enforces a memory cap
  (`RLIMIT_AS`) and an optional CPU-time cap (`RLIMIT_CPU`, via a
  `SIGXCPU` handler that turns into a clean `ResourceLimitExceeded`
  exception rather than an abrupt kill) on POSIX systems, plus best-effort
  network blocking (monkeypatches `socket.socket` in the child process).
  Not a substitute for real container isolation in production, but a
  meaningful, dependency-free layer of defense for a local/Colab
  environment — see `src/interpreter/interpreter.py` for the platform
  caveats (POSIX-only for the resource limits).

## Credit

Core execution/search scaffolding adapted from the
[AIDE](https://arxiv.org/pdf/2502.13138) project, as provided in the
NTU ML 2025 Spring HW2 course template.
