"""
Unit tests for main.py's checkpoint/resume support: save_run() and
load_or_create_journal(). These don't require any real LLM/interpreter
call -- they test the save/load round trip and the config-drift warning
directly against the filesystem (using pytest's tmp_path).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import load_or_create_journal, save_run
from src.agent.journal import Journal
from src.agent.node import Node
from src.utils.config import Config


def make_cfg(exp_name="unit_test_run", data_dir="data/foo", task_goal="predict foo"):
    return Config({"exp_name": exp_name, "data_dir": data_dir, "task_goal": task_goal})


def _build_two_step_journal() -> Journal:
    journal = Journal()
    draft = Node(code="raise ValueError()", plan="p0")
    journal.append(draft)
    draft.is_buggy = True
    draft.metric = None

    fix = Node(code="print('ok')", plan="p1", parent=draft)
    journal.append(fix)
    fix.is_buggy = False
    fix.metric = 0.42
    return journal


def test_load_returns_empty_fresh_journal_when_no_checkpoint_exists(tmp_path):
    cfg = make_cfg()
    journal, resumed = load_or_create_journal(cfg, out_dir=str(tmp_path))
    assert resumed is False
    assert len(journal) == 0


def test_save_then_load_round_trips_the_full_tree(tmp_path):
    cfg = make_cfg()
    original = _build_two_step_journal()
    save_run(cfg, original, out_dir=str(tmp_path))

    assert (tmp_path / f"{cfg.exp_name}_journal.json").exists()
    assert (tmp_path / f"{cfg.exp_name}_meta.json").exists()

    loaded, resumed = load_or_create_journal(cfg, out_dir=str(tmp_path))
    assert resumed is True
    assert len(loaded) == 2
    draft, fix = loaded.nodes
    # Object-identity / tree-structure round trip (the Journal.from_dict
    # fix this relies on -- see journal.py) must still hold after going
    # through main.py's actual save_run()/load_or_create_journal() path,
    # not just the lower-level Journal.from_dict() test that already
    # covers this in isolation.
    assert fix.parent is draft
    assert draft.subtree_size == 2
    assert loaded.get_best_node().metric == 0.42


def test_resuming_with_matching_config_prints_no_mismatch_warning(tmp_path, capsys):
    cfg = make_cfg(data_dir="data/foo", task_goal="predict foo")
    save_run(cfg, _build_two_step_journal(), out_dir=str(tmp_path))

    capsys.readouterr()  # discard anything printed by save_run itself (none currently, but don't assume)
    load_or_create_journal(cfg, out_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_resuming_with_a_different_task_goal_prints_a_mismatch_warning(tmp_path, capsys):
    original_cfg = make_cfg(exp_name="shared_name", data_dir="data/foo", task_goal="predict foo")
    save_run(original_cfg, _build_two_step_journal(), out_dir=str(tmp_path))

    different_cfg = make_cfg(exp_name="shared_name", data_dir="data/foo", task_goal="predict something else entirely")
    _, resumed = load_or_create_journal(different_cfg, out_dir=str(tmp_path))
    out = capsys.readouterr().out

    assert resumed is True  # still resumes -- this is a warning, not a hard block
    assert "WARNING" in out
    assert "different exp_name" in out or "different task" in out or "don't match" in out


def test_resuming_without_a_meta_sidecar_does_not_false_positive_warn(tmp_path, capsys):
    """
    A journal file written before this feature existed (e.g. the one
    already committed at runs/eval_california_housing_journal.json)
    has no *_meta.json sidecar. Resuming it should work silently, not
    raise a spurious mismatch warning just because there's nothing to
    compare against.
    """
    cfg = make_cfg()
    journal_path = tmp_path / f"{cfg.exp_name}_journal.json"
    journal_path.write_text(json.dumps(_build_two_step_journal().to_dict(), default=str))
    # deliberately NOT writing the _meta.json sidecar

    _, resumed = load_or_create_journal(cfg, out_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert resumed is True
    assert "WARNING" not in out
