"""§4 weight formula and §7 the shipped artifact's stub/real-artifact behaviour."""
from __future__ import annotations

import json

import pytest

from cheapy.capability.priors import (
    BASE_CAPABILITY,
    CANDIDATES,
    NEAR_UNMEASURED,
    PROBES,
    prior_for,
    weights,
)
from cheapy.capability.capability_model import score_for_trajectory, score_models
from cheapy.preprocessing.trajectory_analyzer import analyze


class TestPriors:
    def test_beta_zero_gives_equal_weights(self):
        w = weights(0.0)
        values = set(round(v, 9) for v in w.values())
        assert values == {1.0}

    def test_beta_one_equals_prior_directly(self):
        w = weights(1.0)
        for name in CANDIDATES:
            assert w[name] == pytest.approx(BASE_CAPABILITY[name])

    def test_larger_beta_spreads_weights_further_apart(self):
        w_small = weights(0.5)
        w_large = weights(4.0)
        spread_small = max(w_small.values()) - min(w_small.values())
        spread_large = max(w_large.values()) - min(w_large.values())
        assert spread_large > spread_small

    def test_negative_beta_rejected(self):
        with pytest.raises(ValueError):
            weights(-1.0)

    def test_unknown_model_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError):
            prior_for("not-a-real-model")

    def test_candidate_set_covers_all_nine_scored_models(self):
        # Spec §1.0 dropped the two rare models because each Anthropic candidate cost a
        # query per step. Under the OpenAI-only pivot no Anthropic model is queried, so
        # carrying them is free — but they stay flagged as near-unmeasured.
        assert len(CANDIDATES) == 9
        assert "claude-opus-4-6" in CANDIDATES
        assert "claude-sonnet-4-6" in CANDIDATES
        assert NEAR_UNMEASURED == {"claude-opus-4-6", "claude-sonnet-4-6"}

    def test_probes_are_the_three_queryable_models(self):
        assert PROBES == ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
        assert all(p in CANDIDATES for p in PROBES)
        assert all(not p.startswith("claude") for p in PROBES)


@pytest.fixture
def trajectory(claude_trajectory_line):
    return analyze(claude_trajectory_line, id=0)


class TestScoreModelsStubMode:
    """No fitted artifact is present in this checkout, so score_models runs in stub mode
    (§9.3) — this pins that the stub is well-formed, not that it's a real prediction."""

    def test_returns_one_score_per_candidate_in_zero_one(self, trajectory):
        scores = score_models(trajectory, beta=1.0)
        assert set(scores) == set(CANDIDATES)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_beta_zero_ranks_flat_under_the_symmetric_stub(self, trajectory, monkeypatch):
        # Force stub mode explicitly: a fitted artifact may or may not be present in the
        # working tree, and this test is about the stub's shape, not the fitted model's.
        import cheapy.capability.capability_model as cm

        monkeypatch.setattr(cm, "_load", lambda: None)
        scores = cm.score_models(trajectory, beta=0.0)
        values = set(round(v, 9) for v in scores.values())
        # the stub pair value is identical for every pair, so with equal weights every
        # model must land on the same score.
        assert len(values) == 1


class TestScoreForTrajectoryAdapter:
    def test_sets_performance_score_in_place_and_returns_the_list(self, trajectory):
        class FakeModel:
            def __init__(self, name):
                self.name = name
                self.performance_score = None

        # "gpt-4o-mini" is deliberately NOT in BASE_CAPABILITY — the adapter must leave
        # unknown models at None rather than inventing a score for them.
        models = [FakeModel(name) for name in CANDIDATES] + [FakeModel("gpt-4o-mini")]
        result = score_for_trajectory(trajectory, models, beta=1.0)
        assert result is models
        for model in models:
            if model.name in CANDIDATES:
                assert model.performance_score is not None
                assert 0.0 <= model.performance_score <= 1.0
            else:
                # outside the scored set — must not get an invented score.
                assert model.performance_score is None
