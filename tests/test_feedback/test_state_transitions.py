"""Tests for the session state machine and stage transitions."""
from __future__ import annotations

import pandas as pd
import pytest

from core.feedback.session import Session, SessionStore, STAGES
from core.feedback.state import (
    can_transition, valid_next_stages, TRANSITIONS,
)
from core.feedback.workflow_dag import (
    is_valid_action, valid_transitions, ACTION_PRECONDITIONS,
)
from core.feedback.decision_log import DecisionLog


def test_all_stages_defined():
    assert STAGES == ["empty", "data", "preflight", "optimize", "fit",
                       "compare", "risk", "backtest", "report"]


def test_can_transition_data_to_preflight():
    assert can_transition("data", "preflight")


def test_can_transition_preflight_to_optimize():
    assert can_transition("preflight", "optimize")


def test_can_transition_preflight_to_fit():
    assert can_transition("preflight", "fit")


def test_cannot_transition_empty_to_fit():
    assert not can_transition("empty", "fit")


def test_cannot_transition_data_to_compare():
    assert not can_transition("data", "compare")


def test_valid_next_stages_for_empty():
    assert valid_next_stages("empty") == ["data"]


def test_valid_next_stages_for_preflight():
    stages = valid_next_stages("preflight")
    assert "optimize" in stages
    assert "fit" in stages


def test_valid_next_stages_for_fit():
    stages = valid_next_stages("fit")
    assert "compare" in stages
    assert "risk" in stages
    assert "backtest" in stages
    # Fit can also go back to optimize (feedback loop)
    assert "optimize" in stages


def test_valid_next_stages_for_report():
    assert valid_next_stages("report") == []


# ── Session ────────────────────────────────────────────────────────

def test_session_initial_stage():
    s = Session(id="test-1")
    assert s.current_stage == "empty"
    assert s.returns is None
    assert s.fitted_models == {}
    assert len(s.decision_log) == 0


def test_session_advance_stage():
    s = Session(id="test-2")
    s.advance_stage("data")
    assert s.current_stage == "data"
    s.advance_stage("preflight")
    assert s.current_stage == "preflight"


def test_session_advance_invalid_stage_ignored():
    s = Session(id="test-3")
    s.advance_stage("invalid_stage")
    assert s.current_stage == "empty"  # unchanged


def test_session_reset_clears_state():
    s = Session(id="test-4")
    s.returns = pd.Series([1, 2, 3])
    s.advance_stage("data")
    s.log_decision("test.action", "target", outcome="ok")
    s.reset(keep_data=False)
    assert s.current_stage == "empty"
    assert s.returns is None
    assert len(s.decision_log) == 0


def test_session_reset_keeps_data():
    s = Session(id="test-5")
    s.returns = pd.Series([1, 2, 3])
    s.metadata = {"ticker": "TEST"}
    s.advance_stage("data")
    s.reset(keep_data=True)
    assert s.returns is not None
    assert s.metadata == {"ticker": "TEST"}
    assert s.current_stage == "data"


def test_session_log_decision():
    s = Session(id="test-6")
    s.log_decision("preflight.run", "n/a", outcome="pass",
                   detail="12/12 checks passed")
    assert len(s.decision_log) == 1
    entry = s.decision_log.entries[0]
    assert entry.action == "preflight.run"
    assert entry.outcome == "pass"


def test_session_to_dict_has_required_keys():
    s = Session(id="test-7")
    d = s.to_dict()
    assert "session_id" in d
    assert "stage" in d
    assert "next_actions" in d
    assert "decision_count" in d


def test_session_summary_compact():
    s = Session(id="test-8")
    s.advance_stage("fit")
    summary = s.summary()
    assert summary["session_id"] == "test-8"
    assert summary["stage"] == "fit"


# ── SessionStore ───────────────────────────────────────────────────

def test_session_store_open_and_get():
    store = SessionStore()
    sid = store.open()
    s = store.get(sid)
    assert s.id == sid
    assert s.current_stage == "empty"


def test_session_store_close():
    store = SessionStore()
    sid = store.open()
    store.close(sid)
    # Getting a closed session should raise
    import pytest
    with pytest.raises(KeyError):
        store.get(sid)


def test_session_store_list_ids():
    store = SessionStore()
    sid1 = store.open()
    sid2 = store.open()
    ids = store.list_ids()
    assert sid1 in ids
    assert sid2 in ids


# ── Workflow DAG ───────────────────────────────────────────────────

def test_is_valid_action_data_load_when_empty():
    s = Session(id="test-dag-1")
    assert is_valid_action(s, "data.load_market")


def test_is_valid_action_preflight_when_data():
    s = Session(id="test-dag-2")
    s.advance_stage("data")
    assert is_valid_action(s, "preflight.run")


def test_is_invalid_action_fit_when_empty():
    s = Session(id="test-dag-3")
    assert not is_valid_action(s, "models.fit")


def test_valid_transitions_returns_dict():
    s = Session(id="test-dag-4")
    transitions = valid_transitions(s)
    assert isinstance(transitions, dict)
    assert "data.load_market" in transitions


def test_all_actions_have_preconditions():
    """Every action in the DAG should have at least one prerequisite stage."""
    for action, prereqs in ACTION_PRECONDITIONS.items():
        assert len(prereqs) > 0, f"{action} has no prerequisites"
