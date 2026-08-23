"""The OpenAI-only pivot: logged-action recovery, the encoding trap, Nystrom completion,
and the logged-vs-elicited calibration."""
from __future__ import annotations

import numpy as np
import pytest

from research.capability_fitting.calibration import estimate_delta
from research.capability_fitting.canonical import ToolDef
from cheapy.capability.completion import nystrom_complete
from research.capability_fitting.logged_action import action_run_at, logged_action_for
from research.capability_fitting.parser import classify_run
from research.capability_fitting.scoring import ActionType

BASH = ToolDef(name="bash", description="", parameters={
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})
TOOLS = {"bash": BASH}


def claude_msg(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": text}]}


def gpt_msg(text):
    return {"role": "assistant", "content": text}


def call(name="bash", args='{"command": "ls"}'):
    return {"type": "function_call", "name": name, "call_id": "c1", "arguments": args}


class TestEncodingTrap:
    """The two encodings must not be readable as `served_model`. Measured on the corpus, a
    naive reading gives Claude a 56-72% message-rate against 8-14% for GPT."""

    def test_claude_preamble_beside_a_tool_call_is_a_tool_call(self):
        # 74-83% of Claude assistant-message items are preamble accompanying a call.
        run = [claude_msg("I'll check the directory."), call()]
        action = classify_run(run, TOOLS)
        assert action.type is ActionType.TOOL_CALL
        assert action.tool_names == frozenset({"bash"})

    def test_gpt_reasoning_beside_a_tool_call_is_a_tool_call(self):
        run = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]}, call()]
        assert classify_run(run, TOOLS).type is ActionType.TOOL_CALL

    def test_gpt_empty_message_stop_matches_a_claude_text_stop(self):
        # All 65 such turns in the corpus are followed by a user message: the model
        # genuinely stopped. Both encodings must land on the same class.
        assert classify_run([gpt_msg("")], TOOLS).type is ActionType.MESSAGE
        assert classify_run([claude_msg("Done, two files.")], TOOLS).type is ActionType.MESSAGE

    def test_empty_run_is_malformed(self):
        assert classify_run([], TOOLS).type is ActionType.MALFORMED

    def test_call_naming_an_undeclared_tool_is_malformed(self):
        assert classify_run([call(name="shell_command")], TOOLS).type is ActionType.MALFORMED


class TestLoggedActionRecovery:
    def test_recovers_the_run_at_the_cut(self):
        items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            claude_msg("working"), call(),
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]
        assert action_run_at(items, 1) == [items[1], items[2]]

    def test_run_stops_at_the_first_non_generated_item(self):
        items = [call(), {"type": "function_call_output", "call_id": "c1", "output": "ok"}, call()]
        assert action_run_at(items, 0) == [items[0]]

    def test_cut_on_a_non_generated_item_yields_nothing(self):
        items = [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]
        assert action_run_at(items, 0) == []

    def test_classifies_the_served_models_turn(self):
        items = [claude_msg("preamble"), call()]
        logged = logged_action_for("claude-opus-5", items, 0, TOOLS)
        assert logged.model == "claude-opus-5"
        assert logged.action.type is ActionType.TOOL_CALL


class TestNystromCompletion:
    def _truth(self, rank, seed=0):
        rng = np.random.default_rng(seed)
        F = rng.random((9, rank)); G = F @ F.T
        d = np.sqrt(np.diag(G))
        return G / np.outer(d, d)

    def test_exact_when_agreement_is_rank_three(self):
        M = self._truth(3)
        out = nystrom_complete(M[:3, :3], M[3:, :3], ridge=0.0)
        assert np.abs(out.matrix - M[3:, 3:]).max() < 1e-8

    def test_degrades_but_stays_bounded_beyond_rank_three(self):
        M = self._truth(6)
        out = nystrom_complete(M[:3, :3], M[3:, :3])
        assert out.matrix.min() >= 0.0 and out.matrix.max() <= 1.0
        assert np.abs(out.matrix - M[3:, 3:])[np.triu_indices(6, 1)].mean() < 0.30

    def test_reports_the_rank_cap_it_is_limited_by(self):
        M = self._truth(4)
        assert nystrom_complete(M[:3, :3], M[3:, :3]).rank_cap == 3

    def test_warns_when_probes_are_too_alike_to_span(self):
        probe = np.full((3, 3), 0.99); np.fill_diagonal(probe, 1.0)
        out = nystrom_complete(probe, np.full((6, 3), 0.9), ridge=1e-6)
        assert any("ill-conditioned" in w for w in out.warnings())

    def test_output_is_always_clamped_into_range(self):
        probe = np.eye(3) * 0.5
        out = nystrom_complete(probe, np.full((6, 3), 5.0))
        assert out.matrix.min() >= 0.0 and out.matrix.max() <= 1.0
        assert out.clamp_rate > 0


class TestDeltaCalibration:
    def test_consistent_estimates_licence_a_correction(self):
        d = estimate_delta({m: [0.60] * 10 for m in ("a", "b", "c")},
                           {m: [0.74] * 10 for m in ("a", "b", "c")})
        assert d.is_consistent
        assert d.correction() == pytest.approx(0.14)

    def test_scattered_estimates_refuse_to_correct(self):
        # The three probes disagree, so "delta is a property of the regime" is contradicted.
        d = estimate_delta({"a": [0.30] * 10, "b": [0.70] * 10, "c": [0.55] * 10},
                           {m: [0.74] * 10 for m in ("a", "b", "c")})
        assert not d.is_consistent
        assert d.correction() == 0.0

    def test_a_model_with_no_self_pair_is_skipped(self):
        d = estimate_delta({"a": [0.6] * 5, "b": [0.6] * 5}, {"a": [0.74] * 5})
        assert set(d.per_model) == {"a"}

    def test_noise_floor_is_reported_per_model(self):
        d = estimate_delta({"a": [0.6] * 5}, {"a": [0.8, 0.7, 0.9, 0.8, 0.8]})
        assert d.noise_floor["a"] == pytest.approx(0.8)
