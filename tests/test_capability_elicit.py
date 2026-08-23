"""Elicitation driver — §2.3/§2.4: step-major completeness, and the crash-safe/resumable
store contract. A true SIGKILL drill against a subprocess lives outside pytest (run
manually; see the plan) — this covers the store/driver logic pytest can exercise cheaply:
absent vs failed distinguishability, and that a restart never re-issues a completed call.
"""
from __future__ import annotations

from analysis.complexity_model.clients import QueryResult
from analysis.complexity_model.elicit import load_store, run_elicitation
from analysis.complexity_model.priors import PROBES
from analysis.complexity_model.sampler import Sample


def _sample(step_id: str) -> Sample:
    return Sample(
        step_id=step_id,
        trajectory_id=0,
        prefix_items=[
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
        tools=[],
        prefix_token_count=10,
        step_index=1,
        served_model="claude-opus-5",  # a NON-probe: the logged action is the point
    )


class _ScriptedClient:
    """Returns `ok` for the first `succeed_after` calls's worth, then raises — used to
    simulate a model whose calls fail after retries are exhausted (§2.4's "failed" case).
    """

    def __init__(self, always_fail: bool = False):
        self.calls = 0
        self.always_fail = always_fail

    def query(self, model, request, temperature):
        self.calls += 1
        if self.always_fail:
            raise RuntimeError("simulated exhausted-retries failure")
        return QueryResult(raw={"output": []}, effective_temperature=temperature)

    def probe_temperature(self, model):
        return True


class TestStoreDistinguishesAbsentFromFailed:
    def test_a_failed_call_is_recorded_not_left_absent(self, tmp_path):
        store_path = tmp_path / "store.jsonl"
        clients = {m: _ScriptedClient(always_fail=(m == "gpt-5.6-luna")) for m in PROBES}
        run_elicitation([_sample("s1")], store_path, clients=clients, temperatures={m: 0.0 for m in PROBES})

        records = load_store(store_path)
        assert ("gpt-5.6-luna", "s1") in records
        assert records[("gpt-5.6-luna", "s1")]["status"] == "error"
        assert records[("gpt-5.6-sol", "s1")]["status"] == "ok"


class TestResumability:
    def test_restart_does_not_reissue_calls_for_completed_keys(self, tmp_path):
        store_path = tmp_path / "store.jsonl"
        first_clients = {m: _ScriptedClient() for m in PROBES}
        run_elicitation(
            [_sample("s1"), _sample("s2")],
            store_path,
            clients=first_clients,
            temperatures={m: 0.0 for m in PROBES},
        )
        assert len(load_store(store_path)) == 2 * len(PROBES)

        second_clients = {m: _ScriptedClient() for m in PROBES}
        run_elicitation(
            [_sample("s1"), _sample("s2")],
            store_path,
            clients=second_clients,
            temperatures={m: 0.0 for m in PROBES},
        )
        # every key was already present -> the second run should have issued zero calls.
        assert all(client.calls == 0 for client in second_clients.values())
        assert len(load_store(store_path)) == 2 * len(PROBES)  # no duplicate records

    def test_partial_step_is_completed_on_restart_without_touching_the_other(self, tmp_path):
        store_path = tmp_path / "store.jsonl"
        # First run: only gpt-5.6-sol succeeds on s1 (simulate a mid-step crash by
        # supplying a client dict missing the rest — run_elicitation only queries the
        # models present in `clients`, so this models "3 of 7 done, process died").
        partial_clients = {"gpt-5.6-sol": _ScriptedClient()}
        run_elicitation(
            [_sample("s1")], store_path, clients=partial_clients,
            candidates=("gpt-5.6-sol",), temperatures={"gpt-5.6-sol": None},
        )
        assert len(load_store(store_path)) == 1

        full_clients = {m: _ScriptedClient() for m in PROBES}
        run_elicitation([_sample("s1")], store_path, clients=full_clients, temperatures={m: 0.0 for m in PROBES})

        records = load_store(store_path)
        assert len(records) == len(PROBES)
        # the model that already had a record was not re-queried.
        assert full_clients["gpt-5.6-sol"].calls == 0
        # every other model was queried exactly once to fill the gap.
        assert all(full_clients[m].calls == 1 for m in PROBES if m != "gpt-5.6-sol")


class TestSelfPair:
    def test_flagged_step_gets_a_second_draw_per_probe(self, tmp_path):
        store_path = tmp_path / "store.jsonl"
        clients = {m: _ScriptedClient() for m in PROBES}
        run_elicitation(
            [_sample("s1"), _sample("s2")],
            store_path,
            clients=clients,
            temperatures={m: None for m in PROBES},
            self_pair_steps={"s1"},
        )
        records = load_store(store_path)
        # s1 gets both draws, s2 only the first.
        assert ("gpt-5.6-luna", "s1") in records
        assert ("gpt-5.6-luna#2", "s1") in records
        assert ("gpt-5.6-luna#2", "s2") not in records
        assert len(records) == 2 * len(PROBES) + len(PROBES)

    def test_self_pair_key_is_queried_as_the_base_model(self, tmp_path):
        """The `#2` suffix is a store key, not a model id — the API must never see it."""
        store_path = tmp_path / "store.jsonl"

        class _RecordingClient:
            def __init__(self):
                self.queried = []

            def query(self, model, request, temperature):
                self.queried.append(model)
                return QueryResult(raw={"output": []}, effective_temperature=temperature)

            def probe_temperature(self, model):
                return True

        clients = {m: _RecordingClient() for m in PROBES}
        run_elicitation(
            [_sample("s1")], store_path, clients=clients,
            temperatures={m: None for m in PROBES}, self_pair_steps={"s1"},
        )
        for model, client in clients.items():
            assert client.queried == [model, model]  # two draws, real id both times
