"""src/cheapy/routing/router.py: the scoreboard `--verbose` prints.

`decide()` snapshots every scored candidate into `RoutingDecision.scoreboard` at the moment
the verdict is made, because the `ModelLLM` objects it reads are rebuilt and re-scored for
the next trajectory — a later reader holding the objects would see the wrong numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cheapy.models.llm import ModelLLM
from cheapy.models.trajectory import Trajectory, ViktorEnvironment
from cheapy.routing.router import aggregate_scores, decide


def make_model(name: str, price: float | None, performance: float | None) -> ModelLLM:
    return ModelLLM(
        name=name,
        family=name,
        price_score=price,
        performance_score=performance,
        context_window_size=1_000_000,
        input_price_per_1m=2.0,
        cached_input_price_per_1m=0.2,
        output_price_per_1m=10.0,
        cached_output_price_per_1m=0.0,
    )


def make_trajectory(served: str = "model-b") -> Trajectory:
    return Trajectory(
        id=7,
        served_model=served,
        normalized_items=[],
        is_subagent=False,
        viktor_environment=ViktorEnvironment.SLACK,
        toolset=[],
        toolset_size=12,
        avg_tools_per_call=1.0,
        avg_images_received_per_call=0.0,
        avg_output_tokens_per_call=0.0,
        avg_input_tokens_per_call=0.0,
        total_calls=5,
        total_tokens=10_000,
        total_cached_tokens=0,
        last_call_input_tokens=2_000,
        total_tool_calls=0,
        total_images_received=0,
        total_ai_messages=0,
        total_user_messages=0,
        total_draft_submit_calls=0,
    )


@pytest.fixture
def decision():
    models = [
        make_model("model-a", price=0.9, performance=0.5),   # final 0.70
        make_model("model-b", price=0.1, performance=0.9),   # final 0.50, the incumbent
        make_model("model-c", price=0.4, performance=0.4),   # final 0.40
        make_model("model-d", price=None, performance=0.9),  # unscored: no price
    ]
    aggregate_scores(models, w_cost=0.5, w_performance=0.5)
    return decide(make_trajectory("model-b"), models)


class TestScoreboard:
    def test_ranked_best_first(self, decision):
        assert [row.name for row in decision.scoreboard] == ["model-a", "model-b", "model-c"]
        finals = [row.final_score for row in decision.scoreboard]
        assert finals == sorted(finals, reverse=True)

    def test_carries_both_inputs_behind_each_final_score(self, decision):
        top = decision.scoreboard[0]
        assert (top.price_score, top.performance_score) == (0.9, 0.5)
        assert top.final_score == pytest.approx(0.7)

    def test_marks_the_incumbent(self, decision):
        served = [row.name for row in decision.scoreboard if row.is_served]
        assert served == ["model-b"]

    def test_excludes_unscored_candidates(self, decision):
        # A model missing an input is reported as unscored, never ranked last on a score
        # that was never computed.
        assert "model-d" not in [row.name for row in decision.scoreboard]
        assert decision.unscored == ("model-d",)

    def test_ranking_is_the_scoreboard_by_name(self, decision):
        assert decision.ranking == tuple(row.name for row in decision.scoreboard)

    def test_snapshot_survives_rescoring_the_pool(self):
        # The same ModelLLM objects get new scores for the next trajectory; the decision
        # already made must not follow them.
        models = [make_model("model-a", 0.9, 0.5), make_model("model-b", 0.1, 0.9)]
        aggregate_scores(models, w_cost=0.5, w_performance=0.5)
        taken = decide(make_trajectory("model-b"), models)

        for model in models:
            model.price_score, model.performance_score = 0.0, 0.0
        aggregate_scores(models, w_cost=0.5, w_performance=0.5)

        assert taken.scoreboard[0].final_score == pytest.approx(0.7)
