#!/usr/bin/env python3
"""Diagnostics for the fitted pipeline — run after `train.py`.

The single most important number here is the **variance decomposition**. The model predicts
per-step agreement from prefix features, so it can only work if agreement actually varies
*between* steps. If nearly all the variance is *within* a step (i.e. which pair of models is
being compared, plus noise), then no feature set computed from the prefix can beat a
constant predictor, and the baseline gate (§9.6) is failing for a structural reason rather
than a fixable one.

    python -m analysis.complexity_model.diagnostics \
        --samples analysis/complexity_model/store/samples.jsonl \
        --responses analysis/complexity_model/store/responses.jsonl
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis.complexity_model.parser import parse_response
from analysis.complexity_model.priors import PROBES
from analysis.complexity_model.scoring import ActionType
from analysis.complexity_model.train import (
    ALL_FEATURE_NAMES,
    actions_for_step,
    build_dataset,
    load_samples,
    tools_of,
)
from analysis.complexity_model.elicit import load_store


def target_distribution(y: np.ndarray) -> str:
    counts = Counter(np.round(y, 3).tolist())
    lines = ["target distribution (pair values):"]
    for value, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"    {value:<6} {n:>6}  ({n/len(y):>5.1%})")
    lines.append(f"    mean={y.mean():.4f}  var={y.var():.4f}  at 1.0={np.mean(y == 1.0):.1%}  at 0.0={np.mean(y == 0.0):.1%}")
    return "\n".join(lines)


def variance_decomposition(y: np.ndarray, groups: np.ndarray) -> str:
    """Split target variance into between-step and within-step components.

    `between / total` is the ceiling on any R^2 a per-step feature model can reach: features
    are constant within a step, so within-step variance is unreachable by construction.
    """
    by_step: dict[str, list[float]] = defaultdict(list)
    for value, step in zip(y, groups):
        by_step[step].append(float(value))
    step_means = np.array([statistics.fmean(v) for v in by_step.values()])
    grand = y.mean()
    n_per = np.array([len(v) for v in by_step.values()])
    between = float(np.sum(n_per * (step_means - grand) ** 2) / len(y))
    total = float(y.var())
    within = total - between
    return (
        "variance decomposition:\n"
        f"    total            {total:.5f}\n"
        f"    between steps    {between:.5f}   ({between/total:.1%} of total)\n"
        f"    within  steps    {within:.5f}   ({within/total:.1%} of total)\n"
        f"    -> CEILING on any per-step feature model's R^2 is ~{between/total:.1%};\n"
        f"       the rest is which-pair plus single-observation noise and is unreachable."
    )


def feature_signal(X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> str:
    """Correlation of each feature with the target — a cheap read on which carry anything."""
    lines = ["feature/target correlation (|r| descending):"]
    scored = []
    for i, name in enumerate(names):
        col = X[:, i]
        if col.std() == 0:
            scored.append((0.0, name, "CONSTANT"))
            continue
        scored.append((float(np.corrcoef(col, y)[0, 1]), name, ""))
    for r, name, note in sorted(scored, key=lambda t: -abs(t[0]))[:12]:
        lines.append(f"    {name:<24} r={r:+.4f} {note}")
    return "\n".join(lines)


def per_model_stats(samples_path: Path, responses_path: Path) -> str:
    """§5.2 malformed rate per probe, plus measured self-agreement (the ceiling)."""
    samples = load_samples(samples_path)
    store = load_store(responses_path)
    types: dict[str, Counter] = defaultdict(Counter)
    for step_id, sample in samples.items():
        tools = tools_of(sample)
        for probe in PROBES:
            rec = store.get((probe, step_id))
            if rec and rec.get("status") == "ok":
                types[probe][parse_response(probe, rec["raw"], tools).type] += 1
    lines = ["per-probe action mix (§5.2):"]
    for m in PROBES:
        c = types[m]; n = sum(c.values()) or 1
        lines.append(f"    {m:<16} tool_call={c[ActionType.TOOL_CALL]/n:>6.1%}  "
                     f"message={c[ActionType.MESSAGE]/n:>6.1%}  malformed={c[ActionType.MALFORMED]/n:>6.1%}  n={n}")
    return "\n".join(lines)


def same_model_vs_cross_model(samples_path: Path, responses_path: Path) -> str:
    """Probe-vs-log agreement, split by whether the probe *is* the model that served.

    NOT a home-field measurement, despite the obvious reading. When the probe served the
    trajectory, "probe vs log" is the same model compared against itself across the two
    regimes, so the contrast is dominated by model identity, not by who wrote the prefix.
    §8's home-field effect is confounded with that here and cannot be separated: no probe
    is ever observed continuing its own prefix *and* a stranger's on the same step.

    What it does measure cleanly is the same-model/cross-model gap, and read against
    `calibration.py`'s delta (same model, both regimes) it bounds how much of the gap is
    regime rather than identity.
    """
    samples = load_samples(samples_path)
    store = load_store(responses_path)
    from analysis.complexity_model.scoring import pair as pair_score

    own: dict[str, list[float]] = defaultdict(list)
    other: dict[str, list[float]] = defaultdict(list)
    for step_id, sample in samples.items():
        tools = tools_of(sample)
        acts = actions_for_step(sample, store, tools)
        if acts.logged is None or set(acts.elicited) != set(PROBES):
            continue
        for probe in PROBES:
            v = pair_score(acts.elicited[probe], acts.logged)
            (own if sample.served_model == probe else other)[probe].append(v)
    lines = ["probe vs logged action, split by whether the probe served the trajectory",
             "(same-model vs cross-model — see the docstring, this is NOT home-field):"]
    for m in PROBES:
        o, x = own.get(m, []), other.get(m, [])
        if o and x:
            lines.append(f"    {m:<16} same-model={statistics.fmean(o):.4f} (n={len(o)})  "
                         f"cross-model={statistics.fmean(x):.4f} (n={len(x)})  gap={statistics.fmean(o)-statistics.fmean(x):+.4f}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--responses", required=True, type=Path)
    args = p.parse_args()

    dataset, diag, _, self_pairs, cells = build_dataset(args.samples, args.responses)
    print(f"[diag] {diag}\n")
    print(target_distribution(dataset.y), "\n")
    print(variance_decomposition(dataset.y, dataset.groups), "\n")
    print(feature_signal(dataset.X, dataset.y, ALL_FEATURE_NAMES), "\n")
    print(per_model_stats(args.samples, args.responses), "\n")
    print(same_model_vs_cross_model(args.samples, args.responses), "\n")
    if self_pairs:
        print("measured self-agreement (the ceiling on any pair value):")
        for m, vals in sorted(self_pairs.items()):
            print(f"    {m:<16} {statistics.fmean(vals):.4f}  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
