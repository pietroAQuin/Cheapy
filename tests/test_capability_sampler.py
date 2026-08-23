"""Sampling — §1: interior cut selection and the §1.1 context-window filter."""
from __future__ import annotations

import random

from analysis.complexity_model.sampler import MAX_PREFIX_TOKENS, _call_starts, sample_cut


def _record(n_calls: int) -> dict:
    """A trajectory with `n_calls` trivial one-item calls, alternating user/assistant."""
    items = [{"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]}]
    for i in range(n_calls):
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"turn {i}"}]})
        items.append(
            {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": f"reply {i}"}]}
        )
    return {"model": "claude-opus-5", "input": items, "tools": []}


class TestCallStarts:
    def test_counts_one_start_per_call(self):
        record = _record(3)
        starts = _call_starts(record["input"])
        assert len(starts) == 3


class TestSampleCut:
    def test_needs_at_least_three_calls_for_an_interior_boundary(self):
        rng = random.Random(0)
        assert sample_cut(0, _record(2), rng) is None
        assert sample_cut(0, _record(3), rng) is not None

    def test_cut_never_includes_the_first_or_last_call(self):
        rng = random.Random(0)
        for trial in range(20):
            record = _record(6)
            sample = sample_cut(trial, record, random.Random(trial))
            assert sample is not None
            assert 1 <= sample.step_index <= 4  # 6 calls -> starts has 6 entries, interior is [1, 4]

    def test_served_model_is_recorded_but_not_derived_from_anything_else(self):
        record = _record(3)
        sample = sample_cut(0, record, random.Random(0))
        assert sample.served_model == "claude-opus-5"

    def test_step_id_is_a_pure_function_of_trajectory_and_cut(self):
        record = _record(4)
        a = sample_cut(7, record, random.Random(1))
        b = sample_cut(7, record, random.Random(1))
        assert a.step_id == b.step_id  # same trajectory id + same rng draw -> same key

    def test_oversized_prefix_is_excluded(self, monkeypatch):
        import analysis.complexity_model.sampler as sampler_module

        monkeypatch.setattr(sampler_module, "MAX_PREFIX_TOKENS", 1)  # force the filter to trip
        record = _record(5)
        assert sample_cut(0, record, random.Random(0)) is None
