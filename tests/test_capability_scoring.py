"""§5.1's three required behaviours, plus Stage A asymmetry and Stage B Jaccard."""
from __future__ import annotations

import pytest

from research.capability_fitting.scoring import (
    Action,
    ActionType,
    all_malformed,
    build_matrix,
    jaccard,
    pair,
    score,
    score_step,
)

W = {"a": 0.9, "b": 0.6, "c": 0.3}


def tool_call(*names: str) -> Action:
    return Action(type=ActionType.TOOL_CALL, tool_names=frozenset(names))


MESSAGE = Action(type=ActionType.MESSAGE)
MALFORMED = Action(type=ActionType.MALFORMED)


class TestLevel2Cases:
    """The three cases the spec says to verify directly against the formula."""

    def test_unanimity_scores_exactly_one_regardless_of_weights(self):
        actions = {m: tool_call("bash") for m in W}
        result = score_step(actions, W)
        assert result == pytest.approx({m: 1.0 for m in W})

    def test_total_disagreement_ranks_by_prior_share(self):
        actions = {"a": tool_call("x"), "b": tool_call("y"), "c": tool_call("z")}
        total = sum(W.values())
        result = score_step(actions, W)
        for m in W:
            assert result[m] == pytest.approx(W[m] / total)

    def test_lone_dissenter_floors_at_its_own_prior_share(self):
        # b and c agree with each other; a disagrees with both.
        actions = {"a": tool_call("x"), "b": tool_call("y"), "c": tool_call("y")}
        total = sum(W.values())
        result = score_step(actions, W)
        assert result["a"] == pytest.approx(W["a"] / total)
        # b, c inside a cluster score above their own floor.
        assert result["b"] > W["b"] / total
        assert result["c"] > W["c"] / total


class TestStageA:
    def test_both_message_scores_one(self):
        assert pair(MESSAGE, MESSAGE) == 1.0

    def test_message_vs_tool_is_asymmetric(self):
        # row = model being scored. Stopping while the other kept working scores 0;
        # working while the other stopped gets partial credit.
        assert pair(MESSAGE, tool_call("bash")) == 0.0
        assert pair(tool_call("bash"), MESSAGE) == 0.15

    def test_malformed_scores_zero_against_anything(self):
        assert pair(MALFORMED, MALFORMED) == 0.0
        assert pair(MALFORMED, MESSAGE) == 0.0
        assert pair(MESSAGE, MALFORMED) == 0.0
        assert pair(MALFORMED, tool_call("bash")) == 0.0
        assert pair(tool_call("bash"), MALFORMED) == 0.0

    def test_directions_are_computed_independently(self):
        a, b = tool_call("bash"), MESSAGE
        first = pair(a, b)
        second = pair(b, a)
        assert first != second


class TestStageB:
    def test_identical_tool_sets_score_one(self):
        assert jaccard({"bash"}, {"bash"}) == 1.0

    def test_disjoint_tool_sets_score_zero(self):
        assert jaccard({"bash"}, {"file_read"}) == 0.0

    def test_partial_overlap_is_proportional(self):
        assert jaccard({"bash", "file_read"}, {"bash"}) == pytest.approx(0.5)

    def test_duplicate_calls_collapse_to_the_same_set(self):
        assert pair(tool_call("bash", "bash"), tool_call("bash")) == 1.0

    def test_is_symmetric_even_though_stage_a_is_not(self):
        a, b = tool_call("bash", "file_read"), tool_call("bash")
        assert jaccard({"bash", "file_read"}, {"bash"}) == jaccard({"bash"}, {"bash", "file_read"})
        assert pair(a, b) == pair(b, a)


class TestAllMalformedDrop:
    def test_all_malformed_step_is_flagged(self):
        assert all_malformed({"a": MALFORMED, "b": MALFORMED})

    def test_mixed_step_is_not_flagged(self):
        assert not all_malformed({"a": MALFORMED, "b": MESSAGE})

    def test_empty_step_is_not_flagged(self):
        assert not all_malformed({})


class TestBuildMatrix:
    def test_incomplete_step_is_dropped_and_logged(self):
        steps = {"s1": {"a": tool_call("bash"), "b": tool_call("bash")}}  # missing "c"
        matrix = build_matrix(steps, W, candidates=("a", "b", "c"))
        assert matrix.steps_scored == 0
        assert matrix.steps_dropped_incomplete == 1

    def test_all_malformed_step_is_dropped_and_logged(self):
        steps = {"s1": {m: MALFORMED for m in W}}
        matrix = build_matrix(steps, W, candidates=tuple(W))
        assert matrix.steps_scored == 0
        assert matrix.steps_dropped_all_malformed == 1

    def test_complete_step_is_scored_and_malformed_counted(self):
        steps = {"s1": {"a": tool_call("bash"), "b": MALFORMED, "c": tool_call("bash")}}
        matrix = build_matrix(steps, W, candidates=("a", "b", "c"))
        assert matrix.steps_scored == 1
        assert matrix.malformed_counts["b"] == 1
        assert "a" not in matrix.malformed_counts
        rate = matrix.malformed_rate()
        assert rate["b"] == pytest.approx(1.0)
