"""Unit tests for router_models/price_model.py.

Plain `unittest` (stdlib) rather than pytest, to keep the pipeline runnable offline
with no extra dependency — consistent with the README's "offline on a laptop" contract.

Run with: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory, ViktorEnvironment
from router_models.price_model import _next_call_cost, score_price


def make_model(name: str, **overrides) -> ModelLLM:
    defaults = dict(
        name=name,
        family=name,
        context_window_size=1_000_000,
        input_price_per_1m=2.0,
        cached_input_price_per_1m=0.2,
        output_price_per_1m=10.0,
        cached_output_price_per_1m=0.0,
    )
    defaults.update(overrides)
    return ModelLLM(**defaults)


def make_trajectory(**overrides) -> Trajectory:
    defaults = dict(
        id=0,
        served_model="model-a",
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
        total_tool_calls=0,
        total_images_received=0,
        total_ai_messages=0,
        total_user_messages=0,
        total_draft_submit_calls=0,
    )
    defaults.update(overrides)
    return Trajectory(**defaults)


class TestNextCallCost(unittest.TestCase):
    """Exercises `_next_call_cost` directly so the arithmetic is pinned exactly."""

    def test_partial_cache_hold_splits_context_correctly(self):
        # total_tokens=10,000, total_cached_tokens=7,000, total_calls=5 -> increment=2,000.
        # HOLD should bill 7,000 at the cached rate and (10,000-7,000)+2,000=5,000 at the
        # normal rate -- not the whole 10,000+2,000 at the normal-then-fully-cached split
        # a naive "assume it's all cached" formula would produce.
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=7_000, total_calls=5
        )
        model = make_model("model-a", output_price_per_1m=0.0)  # isolate input-side math

        cost = _next_call_cost(trajectory, model)

        expected = (5_000 * 2.0 + 7_000 * 0.2) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_full_cache_hold_bills_only_the_increment_at_full_price(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=100_000, total_cached_tokens=100_000, total_calls=10
        )
        model = make_model("model-a", output_price_per_1m=0.0)

        cost = _next_call_cost(trajectory, model)

        # increment = 100,000 / 10 = 10,000; all of it uncached, the rest fully cached.
        expected = (10_000 * 2.0 + 100_000 * 0.2) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_no_cache_hold_equals_switch_cost_for_the_same_model(self):
        # With total_cached_tokens=0, holding gets no cache discount at all, so it must
        # cost exactly what switching to an identically-priced model would cost.
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=0, total_calls=5
        )
        hold_model = make_model("model-a")
        switch_model = make_model("model-b")  # identical prices, different name

        hold_cost = _next_call_cost(trajectory, hold_model)
        switch_cost = _next_call_cost(trajectory, switch_model)

        self.assertAlmostEqual(hold_cost, switch_cost)

    def test_switch_bills_the_entire_next_input_uncached(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=7_000, total_calls=5
        )
        switch_model = make_model("model-b", output_price_per_1m=0.0)

        cost = _next_call_cost(trajectory, switch_model)

        # next_input_tokens = 10,000 + increment(2,000) = 12,000, all at the normal rate.
        expected = (12_000 * 2.0) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_cache_makes_holding_cheaper_than_an_equally_priced_switch(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=7_000, total_calls=5
        )
        hold_model = make_model("model-a")
        switch_model = make_model("model-b")  # same prices as model-a

        hold_cost = _next_call_cost(trajectory, hold_model)
        switch_cost = _next_call_cost(trajectory, switch_model)

        self.assertLess(hold_cost, switch_cost)

    def test_much_cheaper_switch_can_still_beat_holding(self):
        # The cache trap penalizes switching, but a big enough base-rate discount must
        # still be able to overcome it.
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=1_000_000, total_cached_tokens=950_000, total_calls=20,
            avg_output_tokens_per_call=500,
        )
        hold_model = make_model("model-a", context_window_size=2_000_000)
        cheap_switch_model = make_model(
            "model-cheap",
            context_window_size=2_000_000,
            input_price_per_1m=0.2,
            cached_input_price_per_1m=0.02,
            output_price_per_1m=1.2,
        )

        hold_cost = _next_call_cost(trajectory, hold_model)
        switch_cost = _next_call_cost(trajectory, cheap_switch_model)

        self.assertLess(switch_cost, hold_cost)

    def test_infeasible_context_window_returns_none(self):
        trajectory = make_trajectory(total_tokens=10_000, total_cached_tokens=0, total_calls=5)
        tiny_model = make_model("tiny", context_window_size=100)

        self.assertIsNone(_next_call_cost(trajectory, tiny_model))


class TestScorePrice(unittest.TestCase):
    """Exercises `score_price`'s normalization across a candidate pool."""

    def test_hold_outscores_equally_priced_switch(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=7_000, total_calls=5
        )
        models = [make_model("model-a"), make_model("model-b")]

        score_price(trajectory, models)

        by_name = {m.name: m.price_score for m in models}
        self.assertEqual(by_name["model-a"], 1.0)
        self.assertEqual(by_name["model-b"], 0.0)

    def test_no_cache_ties_hold_and_switch_at_full_score(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=0, total_calls=5
        )
        models = [make_model("model-a"), make_model("model-b")]

        score_price(trajectory, models)

        self.assertEqual(models[0].price_score, 1.0)
        self.assertEqual(models[1].price_score, 1.0)

    def test_infeasible_candidate_scores_zero_and_does_not_compress_others(self):
        trajectory = make_trajectory(
            served_model="model-a", total_tokens=10_000, total_cached_tokens=7_000, total_calls=5
        )
        models = [
            make_model("model-a"),
            make_model("model-b"),
            make_model("tiny", context_window_size=100),
        ]

        score_price(trajectory, models)

        by_name = {m.name: m.price_score for m in models}
        self.assertEqual(by_name["tiny"], 0.0)
        self.assertEqual(by_name["model-a"], 1.0)
        self.assertEqual(by_name["model-b"], 0.0)


if __name__ == "__main__":
    unittest.main()
