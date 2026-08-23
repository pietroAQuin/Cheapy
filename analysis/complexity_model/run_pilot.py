#!/usr/bin/env python3
"""End-to-end driver for the offline pipeline — spec §9, as revised by the pivot.

    python -m analysis.complexity_model.run_pilot --n 957 --fit

Requires `.env` (see `.env.example`) and `pip install -r requirements-elicit.txt`. Costs
real money: only the 3 OpenAI probes are queried, at roughly **$0.16 per step** (~$154 for
the full 957 eligible trajectories, against a $225 budget).

Step-major and fully resumable (§2.3/§2.4) — re-running costs nothing for steps already in
the store, and **Ctrl-C is safe**: the current step finishes, everything is flushed, and
whatever completed is a valid dataset to train on.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from analysis.complexity_model.elicit import load_store, probe_temperatures, run_elicitation
from analysis.complexity_model.priors import PROBES
from analysis.complexity_model.sampler import Sample, write_samples
from pre_processing.model_list import price_for

STORE_DIR = Path(__file__).parent / "store"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

#: Steps that additionally get a second draw from every probe, to measure the
#: sampling-noise floor. Drawn from probe-served trajectories, where the same model is also
#: visible in the log — that pairing is what identifies the logged-vs-elicited offset
#: (`calibration.py`). Small on purpose: it is a control, not a training signal.
DEFAULT_SELF_PAIR_STEPS = 50

#: Sum of the probes' per-1M input prices — input dominates ~130:1 here, so this is the
#: whole cost model to within a rounding error.
_COST_PER_1M = sum(price_for(m)["input_price_per_1m"] for m in PROBES)


def _load_samples(path: Path) -> list[Sample]:
    with open(path, encoding="utf-8") as handle:
        return [Sample(**json.loads(line)) for line in handle if line.strip()]


def _install_graceful_stop() -> None:
    """Ctrl-C finishes the in-flight step rather than tearing the process down mid-write.

    The store is append-and-fsync per response, so a hard kill is survivable anyway — but
    an interrupted step leaves a partial row that scoring then has to discard, and that is
    wasted spend.
    """
    def _handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/trajectories_v1_01.jsonl"))
    p.add_argument("--n", type=int, default=957, help="steps to sample (default: all eligible)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=Path, default=STORE_DIR / "samples.jsonl")
    p.add_argument("--responses", type=Path, default=STORE_DIR / "responses.jsonl")
    p.add_argument("--self-pair-steps", type=int, default=DEFAULT_SELF_PAIR_STEPS)
    p.add_argument("--step-workers", type=int, default=6,
                   help="steps in flight at once; an interrupt leaves at most this many partial")
    p.add_argument("--fit", action="store_true", help="fit the model once elicitation finishes")
    args = p.parse_args()

    from analysis.complexity_model.clients import client_for
    from analysis.complexity_model.env import load_keys

    _, openai_keys = load_keys()
    if not openai_keys:
        raise SystemExit("no OPENAI_API_KEY[_2/_3] set — copy .env.example to .env")
    print(f"[run_pilot] {len(openai_keys)} OpenAI key(s) available for rotation")

    if not args.samples.exists():
        write_samples(args.data, args.n, args.samples, seed=args.seed)
    else:
        print(f"[run_pilot] reusing existing sample draw at {args.samples}")
    samples = _load_samples(args.samples)

    # Self-pair control on probe-served steps — see DEFAULT_SELF_PAIR_STEPS.
    probe_served = [s.step_id for s in samples if s.served_model in PROBES]
    self_pair_steps = set(probe_served[: args.self_pair_steps])

    # Each probe sees the whole prefix once, so the *billed* quantity is prefix tokens
    # times the SUM of the probes' rates — not prefix x n_probes x that sum, which would
    # count the fan-out twice.
    prefix_tokens = sum(s.prefix_token_count for s in samples)
    prefix_tokens += sum(s.prefix_token_count for s in samples if s.step_id in self_pair_steps)
    est_cost = prefix_tokens / 1e6 * _COST_PER_1M
    print(f"[run_pilot] {len(samples)} steps, {len(self_pair_steps)} with self-pair control")
    print(f"[run_pilot] ~{prefix_tokens * len(PROBES) / 1e6:.1f}M input tokens across {len(PROBES)} probes "
          f"-> ~${est_cost:.2f} (${est_cost/max(len(samples),1):.3f}/step)")

    clients = {m: client_for(m, openai_keys=openai_keys) for m in PROBES}
    temperatures = probe_temperatures(clients, PROBES)

    _install_graceful_stop()
    try:
        run_elicitation(samples, args.responses, clients=clients,
                        temperatures=temperatures, self_pair_steps=self_pair_steps,
                        step_workers=args.step_workers)
    except KeyboardInterrupt:
        done = load_store(args.responses)
        complete = sum(1 for s in samples if all((m, s.step_id) in done for m in PROBES))
        print(f"\n[run_pilot] stopped early — {complete} complete steps on disk, all usable. "
              f"Re-run to continue; finished steps cost nothing.")
        return 0

    if args.fit:
        from analysis.complexity_model.train import build_dataset, fit, save_artifact
        from analysis.complexity_model.calibration import estimate_delta

        dataset, diag, lve, sp, cells = build_dataset(args.samples, args.responses)
        print(f"[run_pilot] {diag}")
        delta = estimate_delta(lve, sp)
        print(delta.report())
        model, report = fit(dataset)
        print(f"[run_pilot] {report}")
        if model is not None:
            print(f"[run_pilot] artifact -> {save_artifact(model, dataset, report, delta, cells, ARTIFACT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
