"""
Typed, validated configuration (replaces the old dict-wrapper `Config`).

WHY THIS CHANGED FROM A PLAIN DOT-ACCESS WRAPPER: the previous `Config`
just recursively turned a dict into an object with attribute access -- it
never checked that a key existed, was spelled right, or held a value of
the right type. A typo like `agent.serach.debug_prob` in a YAML file
would silently produce an object with no `debug_prob` at all, and you
wouldn't find out until `AttributeError` surfaced deep inside a run --
potentially after several real (paid) LLM calls had already happened.

Using `pydantic-settings.BaseSettings` instead gets three things for
free, in increasing order of how much they'd have caught in this
project's own history:
  1. Type coercion + validation at config-LOAD time, before a single LLM
     call is made (a bad `steps: "fifteen"` fails immediately, not mid-run).
  2. `extra="forbid"` on every nested model, so a typo'd or misplaced key
     anywhere in the YAML -- not just at the top level -- is a hard error
     instead of a silently-ignored no-op.
  3. Environment-variable overrides for free (see `Config`'s docstring
     below) without hand-rolling any override-merging logic.

`OmegaConf` was the other option considered. It's a better fit when you
need to merge multiple config files/CLI overrides at composition time;
this project loads exactly one YAML file per run, so pydantic's schema
validation + env var support is the more direct fit for what's actually
needed here, and it's already a dependency (used elsewhere for LLM
structured-output validation, see src/agent/schemas.py) rather than a
new one.

Dot-access still works exactly as before (`cfg.agent.search.debug_prob`)
since pydantic models support attribute access natively -- no call site
outside this file needed to change.
"""

import random
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Every nested model shares extra="forbid" so a typo'd/misplaced key is a
# validation error no matter how deep it is, not just at the top level.
_StrictModel = ConfigDict(extra="forbid")


class ReflectionConfig(BaseModel):
    model_config = _StrictModel

    enabled: bool = True
    max_revisions: int = Field(default=1, ge=0, description="How many critic-revision rounds to allow.")


class SearchConfig(BaseModel):
    model_config = _StrictModel

    debug_prob: float = Field(default=0.5, ge=0.0, le=1.0, description="P(debug a buggy leaf) vs. improve.")
    num_drafts: int = Field(default=1, ge=1, description="Number of initial drafts before improving/debugging.")
    exploration_constant: float = Field(default=1.0, ge=0.0, description="UCB1 exploration weight.")


class AgentConfig(BaseModel):
    model_config = _StrictModel

    steps: int = Field(default=1, ge=1, description="Number of search iterations to run.")
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


class LLMConfig(BaseModel):
    model_config = _StrictModel

    backend: Literal["openai", "anthropic", "coze", "llama_cpp"] = "openai"
    model: str | None = Field(default=None, description="Model name (openai/anthropic) or Coze bot_id.")
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- llama_cpp-only settings ---
    model_path: str | None = None
    n_gpu_layers: int = -1
    n_ctx: int = Field(default=8192, ge=1)

    # Deliberately NOT a @model_validator on this class: unit tests build
    # partial Configs (see tests/test_search_policy.py, test_reflection.py)
    # that never touch `llm` at all and never intend to build a real
    # backend from it (they use a ScriptedBackend or no backend). Making
    # "model is required" a property of every LLMConfig instance would
    # force every such test fixture to also invent an irrelevant model
    # name just to pass validation. Instead this check lives in
    # require_llm_config() below, called from main.py's
    # build_llm_backend() at the one point where it actually matters:
    # right before constructing a real backend, still well before any
    # API call is made.


class InterpreterConfig(BaseModel):
    model_config = _StrictModel

    timeout: int = Field(default=3600, gt=0, description="Max wall-clock seconds for one code execution.")
    max_memory_mb: int = Field(default=4096, gt=0, description="Hard memory cap for the child process.")
    max_cpu_seconds: int | None = Field(default=None, gt=0, description="Hard CPU-time cap; None = no separate cap.")
    block_network: bool = True


def require_llm_config(llm_cfg: LLMConfig) -> None:
    """
    Cross-field check that a plain dict wrapper (or a lenient schema)
    wouldn't give you for free: "does this LLMConfig actually have what
    its own backend needs to run." Called from main.py's
    build_llm_backend() right before constructing a real backend --
    still well before any API call is made, so a missing model name
    fails fast instead of surfacing as a confusing error from inside the
    OpenAI/Anthropic/Coze client library.

    Not a pydantic @model_validator on LLMConfig itself on purpose: unit
    tests build partial Configs that never touch `llm` and never intend
    to build a real backend from it (see tests/test_search_policy.py,
    test_reflection.py) -- making this a property of every LLMConfig
    instance would force those fixtures to also invent an irrelevant
    model name just to pass validation.
    """
    if llm_cfg.backend == "llama_cpp" and not llm_cfg.model_path:
        raise ValueError('llm.backend is "llama_cpp" but llm.model_path is not set.')
    if llm_cfg.backend != "llama_cpp" and not llm_cfg.model:
        raise ValueError(
            f'llm.backend is "{llm_cfg.backend}" but llm.model is not set '
            '(it also doubles as the bot_id when backend is "coze").'
        )


class Config(BaseSettings):
    """
    Top-level run configuration, loaded from a YAML file and validated
    against this schema.

    Environment-variable overrides: any field can be overridden without
    touching the YAML file, using the prefix `AIAGENT_` and `__` as the
    nesting delimiter -- e.g.:

        AIAGENT_AGENT__STEPS=5 python main.py --config configs/config.yaml
        AIAGENT_LLM__BACKEND=anthropic AIAGENT_LLM__MODEL=claude-haiku-4-5 python main.py ...
        AIAGENT_INTERPRETER__MAX_MEMORY_MB=8192 python main.py ...

    Env vars win over the YAML file (see settings_customise_sources
    below) -- useful for one-off overrides (CI smoke tests, quick
    experiments) without editing/duplicating a config file. This is
    unrelated to API keys (OPENAI_API_KEY/ANTHROPIC_API_KEY/etc.), which
    are still loaded separately via `.env` + `python-dotenv` in main.py
    and read directly by each LLMBackend -- keeping "which model/task to
    run" config separate from "how to authenticate" secrets on purpose.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIAGENT_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    exp_name: str = "default_run"
    data_dir: str = "data"
    task_goal: str
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    interpreter: InterpreterConfig = Field(default_factory=InterpreterConfig)

    def __init__(self, dictionary: dict | None = None, **kwargs):
        # Preserves the existing call convention used throughout this
        # repo (`Config(raw_cfg)` with a single dict positional arg, e.g.
        # main.py's `Config(yaml.safe_load(f))`) instead of forcing every
        # call site to switch to Config(**raw_cfg) / Config.model_validate.
        if dictionary is not None:
            kwargs = {**dictionary, **kwargs}
        super().__init__(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # pydantic-settings' own default priority puts init kwargs (our
        # YAML file, passed as `dictionary` above) ABOVE env vars -- the
        # opposite of normal "env var overrides config file" semantics.
        # Reordering so env_settings comes first is what actually makes
        # the AIAGENT_* overrides documented above take effect.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def set_seed(seed: int = 531) -> None:
    """Set random seeds across python/numpy/torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
