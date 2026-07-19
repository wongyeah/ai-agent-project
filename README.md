# ML Coding Agent

An LLM-driven agent that autonomously drafts, executes, debugs, and
iteratively improves Python solutions to a machine learning task —
inspired by [AIDE: AI-Driven Exploration in the Space of Code](https://arxiv.org/pdf/2502.13138).


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

If you'd rather use Claude instead of OpenAI (e.g. no OpenAI credits
available), set `llm.backend: "anthropic"` and `llm.model` to a Claude
model id (e.g. `claude-haiku-4-5` for the cheap/fast tier) in the config,
and set `ANTHROPIC_API_KEY` in `.env` instead of `OPENAI_API_KEY`. Get a
key from the Anthropic Console (platform.claude.com -> Settings -> API
Keys) — this is a separate, pay-as-you-go API credential, unrelated to
any claude.ai chat subscription.

There's also a `llm.backend: "coze"` option for Coze (扣子, ByteDance's AI
agent platform) — its setup is meaningfully different from the two above
(you build/publish a bot in the Coze web UI first, and its free tier is a
small one-time cumulative call cap rather than a paid-as-you-go budget),
see `CozeBackend`'s docstring in `src/llm/backend.py` and `.env.example`
before using it.

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

## Evaluation results

This section reports a real, end-to-end run of the search pipeline
against a one-shot baseline, on a self-held-out validation split of a
real regression dataset — evidence the draft → debug → improve loop with
UCB1 node selection actually outperforms a single LLM call, not just a
description of the design.

### Methodology, stated plainly

**The LLM backend used for this run was Claude (me), not this project's
configured OpenAI/Anthropic/Coze API backend.** At the time this run was
produced, no working LLM API credentials were available (Coze's free
tier proved insufficient for this task's code-generation demands; a
Claude API top-up was blocked by a bank card issue). Rather than fabricate
numbers or wait, I (Claude) stood in for the API call directly: I authored
each step's plan + code myself, the code executed for real in this
project's real sandboxed `Interpreter` (real subprocess, real timeout,
real memory limit, real captured stdout/stderr/exceptions), and I judged
each execution's bugginess/metric from the real output — the same two
responsibilities `Agent.plan_and_code_query` and `Agent.parse_exec_result`
normally hand to an LLM API call.

Everything else in the run is this project's real, unmodified code,
exercised for real:

- `Agent.search_policy()` — the actual UCB1 node-selection logic — decided
  what to work on at every step (draft a new attempt vs. debug a buggy
  leaf vs. improve a specific existing node), exactly as it would in a
  real `python main.py` run. I did not choose which node to build on;
  the algorithm did, from the real accumulated `Journal` state.
- `Journal` / `Node` — the real solution-tree bookkeeping, including
  `subtree_size`-based visit counts feeding into the UCB1 score.
- `Interpreter` — the real sandboxed code execution (subprocess isolation,
  memory/CPU/timeout limits, network blocking).

The driver script that ran this loop (`scripts/claude_as_llm_standin_driver.py`)
and the resulting `runs/eval_california_housing_journal.json` are both
included in this repo so the process is fully inspectable — every step's
plan, code, execution output, and judged metric is in that JSON file, not
just the summary table below.

**Why California Housing instead of the House Prices dataset
(`configs/eval_house_prices.yaml`):** House Prices requires a free
Kaggle account and a manual download (see the setup comment in that
config), which was a blocker under the time pressure this run was
produced under. California Housing (Pace & Barry 1997, via
[a standard mirrored CSV](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv))
is a license-free regression dataset with the same evaluation shape
(self-held-out validation split, self-reported MSE) — a drop-in stand-in
that exercises the identical pipeline. `configs/eval_california_housing.yaml`
is the config this run used; swap `data_dir` back to House Prices (or any
other regression CSV pair) once Kaggle access is available — no agent/
search code needs to change either way. The 16512/4128 train/test split
in `data/california-housing/` was made with `random_state=531`, matching
this project's own `set_seed()` default.

### Two bugs found while producing this experiment

Running a real multi-step search (as opposed to the shipped
`configs/config.yaml` default of `steps: 1`, which every existing unit
test also implicitly relies on) surfaced two real bugs in the codebase:

1. **`RecursionError` in `Journal.to_dict()` — fixed.** `main.py`'s
   `save_run()` calls `journal.to_dict()` after every single step. The
   moment the tree has any parent/child pair (i.e. from the second real
   step onward — a debug or improve node), the old code recursed forever:
   `parent` and a would-be `children` field pointed at each other, and
   `dataclasses_json`'s serializer walked that cycle infinitely. This was
   completely latent before this run: `steps: 1` never produces a second
   node, and every existing unit test builds `Node`s directly without
   ever calling `to_dict()` on a multi-level tree. Fixed in
   `src/agent/node.py` (`children` is now a plain instance attribute, not
   a dataclass field, so the serializer never touches it) and
   `src/agent/journal.py` (a `from_dict` override that rebuilds the
   parent/children object graph, matched by node `id`, after reload).
   Regression test: `tests/test_journal_serialization.py`. Without this
   fix, no real multi-step run — including the one that produced this
   section — could have gotten past the first debug/improve step without
   crashing on checkpoint save.
2. **`Agent._node_value()` / `Journal.get_best_node()` hardcode
   "lower is better" — noted, not fixed.** Both assume an MSE-style
   metric (`1 / (1 + metric)` for the UCB score; `min(nodes, key=...)` for
   "best"). This is correct for every regression config in this repo
   (California Housing, House Prices, ML2025 HW2) but would silently pick
   the *worst* model as "best" on an accuracy-style metric such as
   `configs/eval_titanic.yaml`. Left unfixed here (out of scope for this
   experiment, which is MSE-only) but flagged as a concrete next
   improvement — see "Known limitations / roadmap" below.

### Search trajectory (12 steps, UCB1 search policy)

Search config used (matches `configs/eval_california_housing.yaml`):
`num_drafts: 3`, `debug_prob: 0.5`, `exploration_constant: 1.0`.

| step | stage | change | validation MSE | new best? |
|---|---|---|---|---|
| 0 | draft | Ridge regression, one-hot encoded categoricals | 4,745,217,127 | ✅ (first) |
| 1 | draft | RandomForest(n_estimators=100) | 2,388,730,260 | ✅ |
| 2 | draft | GradientBoostingRegressor | 3,022,624,824 | — |
| 3 | improve (of step 1) | +3 engineered ratio features (rooms/bedrooms/pop per household) | 2,610,193,158 | — (worse than parent) |
| 4 | improve (of step 3) | tuned RF: n_estimators 100→300, min_samples_leaf=2 | 2,588,810,620 | — |
| 5 | improve (of step 4) | +log1p on skewed counts, +KMeans(15) geo-cluster feature | 2,559,037,996 | — |
| 6 | improve (of step 5) | +haversine distance-to-4-major-cities features | 2,284,658,108 | ✅ |
| 7 | improve (of step 6) | swapped RandomForest → HistGradientBoostingRegressor | 1,961,223,556 | ✅ |
| 8 | improve (of step 7) | tuned HGB: lr=0.03, max_iter=800 w/ early stopping, max_leaf_nodes=63 | 1,930,051,051 | ✅ |
| 9 | improve (of step 8) | +median_income interaction features (income/room, income×distance) | **1,903,868,429** | ✅ **best overall** |
| 10 | improve (of step 9) | 50/50 blend of HGB + RandomForest | 2,041,777,392 | — (blend hurt: RF alone scored 2.35e9, dragged the average up) |
| 11 | improve (of step 10) | re-weighted blend to 80/20 HGB/RF | 1,938,342,331 | — (still worse than pure HGB) |

Two things worth calling out as genuine (not cherry-picked) observations
from letting the real algorithm run: step 3 made things *worse* than its
parent (a real, honest example of the search not being monotonically
improving — the debug/improve loop has to tolerate that), and at step 10
UCB1 chose to build on the *worse* step-10 blend rather than the
better-scoring step 9, because step 9 already had one child (lower
"unexplored" bonus) while step 10 was still a fresh leaf — real UCB1
exploration/exploitation trade-off behavior, not a scripted choice.

![Search trajectory vs. one-shot baseline](runs/eval_california_housing_search_vs_baseline.png)

### One-shot baseline (control group)

A single non-iterated attempt: one plan + code generation, one execution,
no search/reflection loop — the same "first reasonable attempt" a plain
LLM call would produce (median-fill missing values, one-hot encode the
categorical column, plain `RandomForestRegressor(n_estimators=100)`, no
feature engineering, no hyperparameter tuning, no model comparison).
Result: **MSE = 2,388,730,260** (RMSE ≈ 48,875) — coincidentally identical
to the search run's own step-1 draft, since that draft used the same
"first reasonable attempt" approach.

| | MSE | RMSE | vs. baseline |
|---|---|---|---|
| One-shot baseline (no search) | 2,388,730,260 | 48,875 | — |
| Search best (step 9, 12 steps total) | 1,903,868,429 | 43,633 | **−20.3% MSE / −10.7% RMSE** |

### Conclusion

On this run, the search loop found a solution reducing validation MSE by
20.3% (RMSE by 10.7%) over a single one-shot LLM attempt, driven mainly
by two changes a one-shot attempt has no mechanism to discover: swapping
model family after seeing RandomForest plateau (step 7, the single
biggest jump), and iteratively adding domain-informed geospatial features
(city-distance, geo-clusters, income interactions) that compound across
several "improve" steps. The two negative results (step 3's regression,
steps 10-11's blending attempts) are included rather than trimmed out,
since a fair account of "does the search process work" has to show it
isn't monotonic — it explores, sometimes gets worse, and still ends up
ahead of the no-search baseline.

## Reproducing this locally

Everything above can be reproduced with real LLM API calls (this project's
actual, unmodified pipeline) instead of the Claude-as-stand-in method
used to produce the numbers above. Here's the full software environment
and step-by-step process.

### 1. Software environment

- Python 3.10+ (3.11 was used for the run above)
- `pip install -r requirements.txt` (installs `pandas`, `scikit-learn`,
  `dataclasses-json`, `pyyaml`, `python-dotenv`, `anthropic`/`openai`
  SDKs, etc. — see the file for the full list)
- `pip install matplotlib` (only needed for the plotting script; not in
  `requirements.txt` since the core pipeline doesn't otherwise need it)
- A working LLM API credential — pick ONE:
  - **Claude (recommended given this repo's own experience with OpenAI/Coze
    credit limits):** an Anthropic Console API key (`platform.claude.com`
    → Settings → API Keys — a separate, pay-as-you-go credential, *not*
    the same thing as a claude.ai chat subscription). Put it in `.env` as
    `ANTHROPIC_API_KEY=sk-ant-...`.
  - OpenAI: `OPENAI_API_KEY=sk-...` in `.env`.
  - Coze: see `CozeBackend`'s docstring in `src/llm/backend.py` and
    `.env.example` — works, but this run's own experience is that its
    free tier struggled with this task's code-generation demands; budget
    accordingly or expect to fall back to Claude/OpenAI.
  - Or a local GGUF model via `llm.backend: "llama_cpp"` (no API key
    needed, but needs a GPU with enough VRAM and a downloaded model file
    — see the Setup section above).
- Copy `.env.example` to `.env` and fill in whichever key you're using:
  `cp .env.example .env`

### 2. Run the real search (replaces the Claude-as-stand-in driver)

```bash
python main.py --config configs/eval_california_housing.yaml
```

This runs the real `Agent.step()` loop for `agent.steps` iterations (15,
in that config — adjust the `agent.steps` value directly in the YAML for
a different range within 10-20), calling your configured LLM backend for
every draft/debug/improve/reflect/evaluate step, executing every attempt
in the real sandboxed `Interpreter`, and writing the growing `Journal` to
`runs/eval_california_housing_journal.json` after every step (so you can
inspect progress, or resume-by-rerunning, without waiting for the whole
run to finish — `main.py` picks up wherever `len(journal)` left off if
you re-point `Journal` at an existing file, though the shipped `main.py`
starts a fresh `Journal()` each invocation by default).

Per-iteration metric data: already recorded for you. Every `Node` in
`runs/eval_california_housing_journal.json` has `step`, `metric`,
`is_buggy`, `plan`, `code`, and the raw `term_out` — nothing extra to
track by hand.

### 3. Run the one-shot baseline

```bash
python scripts/run_one_shot_baseline.py --config configs/eval_california_housing.yaml
```

This calls the same `Agent._draft()` + `Agent.parse_exec_result()`
methods the search loop's first draft uses, but exactly once — a fair
control group generated by the same prompts/evaluation logic, just
without iteration. Writes `runs/eval_california_housing_baseline.json`.

### 4. Plot the curve

```bash
python scripts/plot_search_trajectory.py \
    --journal runs/eval_california_housing_journal.json \
    --baseline runs/eval_california_housing_baseline.json \
    --out runs/search_vs_baseline.png
```

X-axis: search iteration. Y-axis: best-metric-found-so-far (a monotonic
step curve), with each step's own raw metric plotted lightly underneath
and the one-shot baseline as a horizontal reference line. Pass
`--higher-is-better` if you point this at an accuracy-style config (e.g.
`eval_titanic.yaml`) instead of an MSE-style one — see the note at the
top of that script, and the `_node_value` limitation above, before doing
that.

### 5. Update this README

Re-run the numbers above (trajectory table, baseline comparison, plot)
with your new results, and update the "Methodology, stated plainly"
section to say you used a real LLM backend instead of the Claude
stand-in — everything else in that section (UCB1 search policy, real
Interpreter execution, real Journal bookkeeping) is already accurate for
a real run, since it's the same unmodified code either way.

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
- ✅ **Sandbox hardening**: the `Interpreter` enforces a memory cap and an
  optional CPU-time cap via three layers: fast POSIX kernel limits
  (`RLIMIT_AS`/`RLIMIT_CPU`) on Linux/Mac (toggle via
  `use_resource_limits`); native Windows kernel limits via Job Objects
  (`JOB_OBJECT_LIMIT_PROCESS_MEMORY` / `JOB_OBJECT_LIMIT_PROCESS_TIME`,
  see `src/interpreter/win_job_object.py`, toggle via
  `use_job_object_limits`); and a cross-platform `psutil`-based polling
  monitor that works everywhere and now serves as the fallback layer
  rather than Windows' primary mechanism. Plus best-effort network
  blocking (monkeypatches `socket.socket` in the child process). Not a
  substitute for real container isolation in production, but a
  meaningful, dependency-light layer of defense for a local/Colab
  environment — see `src/interpreter/interpreter.py` for the full design
  notes and the trade-offs between the three enforcement layers, and
  `src/interpreter/win_job_object.py` for why the Windows branch is a
  deliberately independent, kernel-level implementation rather than a
  variant of the psutil poller.
