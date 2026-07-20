"""
Unit tests for the pydantic-based Config schema (src/utils/config.py):
type/schema validation, typo detection, env-var overrides, and the
backend-specific requirement check used by main.py's build_llm_backend().
"""

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config, require_llm_config

VALID_DICT = {
    "exp_name": "unit_test",
    "data_dir": "data/whatever",
    "task_goal": "predict something",
    "llm": {"backend": "anthropic", "model": "claude-haiku-4-5"},
    "agent": {
        "steps": 5,
        "reflection": {"enabled": True, "max_revisions": 1},
        "search": {"debug_prob": 0.5, "num_drafts": 3, "exploration_constant": 1.0},
    },
    "interpreter": {"timeout": 180, "max_memory_mb": 4096, "max_cpu_seconds": None, "block_network": True},
}


def test_valid_full_config_loads_and_supports_dot_access():
    cfg = Config(VALID_DICT)
    assert cfg.exp_name == "unit_test"
    assert cfg.agent.steps == 5
    assert cfg.agent.search.debug_prob == 0.5
    assert cfg.llm.backend == "anthropic"
    assert cfg.interpreter.max_memory_mb == 4096


def test_partial_config_gets_sensible_defaults():
    """
    Mirrors what tests/test_search_policy.py and tests/test_reflection.py
    actually pass: only task_goal + agent.search, nothing else. Must keep
    working unchanged after the pydantic migration -- those tests only
    care about search_policy()/_reflect_and_revise(), not the full schema.
    """
    cfg = Config({"task_goal": "dummy task", "agent": {"search": {"debug_prob": 0.0, "num_drafts": 1}}})
    assert cfg.exp_name == "default_run"
    assert cfg.data_dir == "data"
    assert cfg.agent.steps == 1
    assert cfg.agent.search.debug_prob == 0.0
    assert cfg.llm.backend == "openai"
    assert cfg.llm.model is None


def test_missing_task_goal_is_rejected():
    with pytest.raises(ValidationError):
        Config({"exp_name": "x"})


@pytest.mark.parametrize(
    "bad_path,bad_value",
    [
        (("agent", "steps"), "fifteen"),  # wrong type, not coercible
        (("agent", "search", "debug_prob"), 1.5),  # out of [0, 1] range
        (("agent", "search", "num_drafts"), 0),  # must be >= 1
        (("interpreter", "timeout"), -1),  # must be > 0
        (("llm", "backend"), "chatgpt5"),  # not one of the known Literal values
    ],
)
def test_invalid_values_are_rejected(bad_path, bad_value):
    bad = _deep_copy_and_set(VALID_DICT, bad_path, bad_value)
    with pytest.raises(ValidationError):
        Config(bad)


def test_typo_in_nested_key_is_rejected_not_silently_ignored():
    """
    This is the exact failure mode the old dict-wrapper Config couldn't
    catch: a misspelled nested key used to silently produce an object
    with no `debug_prob` attribute at all, only surfacing as an
    AttributeError deep inside a run. extra="forbid" on every nested
    model makes this a load-time error instead.
    """
    bad = _deep_copy_and_set(VALID_DICT, ("agent", "search"), None)
    bad["agent"]["serach"] = {"debug_prob": 0.5, "num_drafts": 1}  # typo: serach
    with pytest.raises(ValidationError):
        Config(bad)


def test_unknown_top_level_key_is_rejected():
    bad = dict(VALID_DICT)
    bad["oops_typo_field"] = 123
    with pytest.raises(ValidationError):
        Config(bad)


def test_env_var_overrides_win_over_the_yaml_dict():
    os.environ["AIAGENT_AGENT__STEPS"] = "7"
    os.environ["AIAGENT_LLM__BACKEND"] = "coze"
    try:
        cfg = Config(VALID_DICT)
        assert cfg.agent.steps == 7  # overridden, VALID_DICT says 5
        assert cfg.llm.backend == "coze"  # overridden, VALID_DICT says anthropic
        assert cfg.llm.model == "claude-haiku-4-5"  # untouched field keeps the yaml value
    finally:
        del os.environ["AIAGENT_AGENT__STEPS"]
        del os.environ["AIAGENT_LLM__BACKEND"]


def test_require_llm_config_passes_for_a_complete_backend_config():
    cfg = Config(VALID_DICT)
    require_llm_config(cfg.llm)  # must not raise


def test_require_llm_config_rejects_missing_model_for_non_llama_cpp_backend():
    cfg = Config({"task_goal": "x", "llm": {"backend": "openai"}})  # model omitted
    with pytest.raises(ValueError, match="llm.model is not set"):
        require_llm_config(cfg.llm)


def test_require_llm_config_rejects_missing_model_path_for_llama_cpp():
    cfg = Config({"task_goal": "x", "llm": {"backend": "llama_cpp"}})  # model_path omitted
    with pytest.raises(ValueError, match="llm.model_path is not set"):
        require_llm_config(cfg.llm)


def test_partial_config_without_llm_never_raises_at_construction_time():
    """
    require_llm_config() is intentionally NOT a @model_validator on
    LLMConfig (see its docstring): constructing a Config that never
    touches `llm` at all -- like the two test files' make_cfg() helpers,
    which only exercise search_policy()/_reflect_and_revise() and never
    build a real backend -- must not fail just because a model name was
    never specified.
    """
    Config({"task_goal": "dummy task"})  # must not raise


def _deep_copy_and_set(base: dict, path: tuple, value) -> dict:
    import copy

    out = copy.deepcopy(base)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out
