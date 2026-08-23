"""Elicitation driver — spec §2.3, §2.4.

Step-major, **required**: complete every probe on step 1, persist, then move to step 2.
Scoring needs a step's full set of responses to be worth anything, so a partially-filled
step is worth nothing. Model-major interrupted at 50% yields zero usable rows; step-major
interrupted at 50% yields half the steps, fully usable — see the spec's worked comparison.

Only the three OpenAI `PROBES` are elicited: OpenAI credits are the only ones available,
so the six Anthropic candidates are positioned instead by recovering their logged actions
from the corpus (`logged_action.py`) — no query, and no synthesised agreement.

The response store is crash-safe and fully resumable by construction: every response is
appended to disk **the moment it returns**, before the next call is issued, and a run
restarted after any failure re-reads the store and skips whatever key is already present.
Nothing is ever buffered in memory for a batch write.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from research.capability_fitting.canonical import ToolDef, to_canonical
from research.capability_fitting.clients import Client, RetryableError, client_for
from cheapy.capability.priors import PROBES
from research.capability_fitting.sampler import Sample


#: Suffix marking the second draw of a self-pair. Keeping it in the `model` slot means the
#: store's `(model, step_id)` key stays unique and resume logic needs no special case.
SELF_PAIR_SUFFIX = "#2"


def self_pair_id(model: str) -> str:
    return f"{model}{SELF_PAIR_SUFFIX}"


def base_model_of(key_model: str) -> str:
    """Strip the self-pair suffix — the model to actually query, and to parse the reply as."""
    return key_model.removesuffix(SELF_PAIR_SUFFIX)


def _record_key(model: str, step_id: str) -> tuple[str, str]:
    return (model, step_id)


def load_store(store_path: str | Path) -> dict[tuple[str, str], dict]:
    """Read every record already on disk. Missing file -> empty store (first run)."""
    store_path = Path(store_path)
    if not store_path.exists():
        return {}
    records: dict[tuple[str, str], dict] = {}
    with open(store_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[_record_key(record["model"], record["step_id"])] = record
    return records


#: Serializes appends. Several steps may be in flight at once (`step_workers`), and two
#: threads writing the store concurrently could interleave a partial line and corrupt the
#: JSONL. The lock is held only for the write, never across an API call.
_WRITE_LOCK = threading.Lock()


def _append_record(store_path: Path, record: dict) -> None:
    """Persist one response the instant it returns. Append-only, flushed and fsynced
    immediately — a crash on call N+1 must never threaten calls 1..N already on disk."""
    with _WRITE_LOCK, open(store_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def _query_one(
    client: Client,
    model: str,
    sample: Sample,
    tools: tuple[ToolDef, ...],
    temperature: float | None,
) -> dict:
    """One model's response to one step, as a store record — success or failure, never
    absent. Absent means 'not yet attempted' and would be retried forever; a stored
    failure means 'attempted, gave up' (§2.4)."""
    canonical = to_canonical(sample.prefix_items, sample.tools)
    query_model = base_model_of(model)
    started = time.time()
    try:
        result = client.query(query_model, canonical, temperature)
        return {
            "model": model,
            "step_id": sample.step_id,
            "status": "ok",
            "raw": result.raw,
            "effective_temperature": result.effective_temperature,
            "elapsed_s": time.time() - started,
        }
    except Exception as exc:  # noqa: BLE001 — a failed call is a *recorded* outcome
        # Terminal vs transient. §2.4 wants failures stored so "gave up" is distinguishable
        # from "not yet attempted" — but that reasoning assumes the failure is a property of
        # the request. A network outage is not: banking it terminally would permanently lose
        # that (model, step_id) to an event unrelated to the model. Transient failures are
        # recorded for the audit trail and retried on the next run (see `run_elicitation`).
        transient = isinstance(exc, RetryableError) or type(exc).__name__ in (
            "APITimeoutError", "APIConnectionError", "ConnectionError", "TimeoutError",
        )
        return {
            "model": model,
            "step_id": sample.step_id,
            "status": "error",
            "terminal": not transient,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.time() - started,
        }


def probe_temperatures(
    clients: dict[str, Client], candidates: Iterable[str] = PROBES
) -> dict[str, float | None]:
    """One test call per model (§2.1) — run once before the batch, never assumed. Returns
    the temperature to send: `0.0` where accepted, `None` (omit the parameter) where the
    model 400s on it."""
    effective: dict[str, float | None] = {}
    for model in candidates:
        accepts = clients[model].probe_temperature(model)
        effective[model] = 0.0 if accepts else None
        print(f"[elicit] temperature probe: {model} -> {'0.0' if accepts else 'omitted (rejects)'}")
    return effective


def run_elicitation(
    samples: Iterable[Sample],
    store_path: str | Path,
    clients: dict[str, Client] | None = None,
    candidates: Iterable[str] = PROBES,
    temperatures: dict[str, float | None] | None = None,
    max_workers: int = 6,
    self_pair_steps: set[str] | None = None,
    step_workers: int = 1,
) -> None:
    """The step-major driver. `clients` is injectable so tests (and the resumability
    drill) can run against a recorded-fixture client with no network or API keys.

    A step is marked complete only once every candidate has a stored record — success or
    terminal failure. Steps already fully present in the store are skipped without
    issuing any call, which is what makes a restart free.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = tuple(candidates)
    if clients is None:
        from research.capability_fitting.env import load_keys

        anthropic_key, openai_keys = load_keys()
        clients = {
            m: client_for(m, anthropic_key=anthropic_key, openai_keys=openai_keys) for m in candidates
        }
    temperatures = temperatures or probe_temperatures(clients, candidates)

    done = load_store(store_path)
    # A transient failure is not "done" — drop it so this run retries it. Terminal failures
    # (a content-policy 400, say) stay, because retrying them just re-buys the same refusal.
    done = {
        key: rec for key, rec in done.items()
        if rec.get("status") == "ok" or rec.get("terminal", True)
    }

    self_pair_steps = self_pair_steps or set()

    def _wanted_for(sample: Sample) -> list[str]:
        wanted = list(candidates)
        if sample.step_id in self_pair_steps:
            # Second independent draw from each probe on the same prefix. This is the only
            # way to see the sampling-noise floor: none of the three models accepts
            # `temperature=0` (they 400 on it), so every measurement carries real
            # stochasticity, and measured self-agreement is the ceiling against which any
            # cross-model agreement has to be read.
            wanted += [self_pair_id(m) for m in candidates]
        return wanted

    def _run_step(sample: Sample) -> None:
        wanted = _wanted_for(sample)
        pending = [m for m in wanted if _record_key(m, sample.step_id) not in done]
        if not pending:
            return  # step already complete — the free part of resuming

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _query_one,
                    clients[base_model_of(model)],
                    model,
                    sample,
                    (),
                    temperatures.get(base_model_of(model)),
                ): model
                for model in pending
            }
            for future in as_completed(futures):
                record = future.result()
                _append_record(store_path, record)  # before the next call, per §2.4
                done[_record_key(record["model"], record["step_id"])] = record

        ok = sum(
            1 for m in wanted if done.get(_record_key(m, sample.step_id), {}).get("status") == "ok"
        )
        print(f"[elicit] step {sample.step_id}: {ok}/{len(wanted)} ok", flush=True)

    if step_workers <= 1:
        for sample in samples:
            _run_step(sample)
        return

    # Several steps in flight. Still step-major in the sense §2.3 actually cares about: a
    # step is only ever completed as a unit, so an interruption leaves at most
    # `step_workers` partial steps and every other finished step is fully usable.
    with ThreadPoolExecutor(max_workers=step_workers) as step_pool:
        list(step_pool.map(_run_step, samples))
