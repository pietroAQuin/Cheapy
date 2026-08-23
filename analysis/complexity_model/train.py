"""Fitting the pairwise-agreement model — spec §6, as revised by the OpenAI-only pivot.

Reads the response store and the samples it came from, recovers each step's logged action,
scores every measured ordered pair, and fits one pooled model.

    python -m analysis.complexity_model.train \
        --samples analysis/complexity_model/store/samples.jsonl \
        --responses analysis/complexity_model/store/responses.jsonl

Everything here is offline and re-runnable from the store at zero API cost — the point of
persisting raw responses (§2.4). Rescoring after a rule change costs nothing.

### What is measured

Per step: the 3 probes are elicited, and the served model's real next action is recovered
from the log. That gives **12 ordered rows per step** — 6 probe-vs-probe and 6
logged-vs-probe — against 6 under the abandoned probe-only design. Pooled over the corpus
it fills all three probe *columns* of the 9x9 agreement matrix for every model, i.e. 21 of
36 unordered pairs. The Claude-vs-Claude block is completed separately (`completion.py`).

### The regime indicators

`a_is_logged` / `b_is_logged` mark whether each side of a pair came from the log rather
than from a live query. They matter because the two regimes are not interchangeable: a
logged action was produced by the model that wrote the whole prefix, at the harness's
sampling settings. Carrying them as features lets the model learn that offset **jointly**,
identified from probe-served steps where the same model is visible both ways — instead of
bolting a scalar correction on afterwards. At inference nothing is logged, so both are 0.
`calibration.py` independently checks that this offset really is a property of the regime
and not of the model, which is the assumption the indicators encode.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from analysis.complexity_model.calibration import estimate_delta
from analysis.complexity_model.canonical import ToolDef, convert_tool
from analysis.complexity_model.elicit import SELF_PAIR_SUFFIX, base_model_of, load_store
from analysis.complexity_model.features import (
    FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    extract_step_features,
    pair_descriptor,
)
from analysis.complexity_model.logged_action import logged_action_for
from analysis.complexity_model.parser import parse_response
from analysis.complexity_model.priors import BASE_CAPABILITY, CANDIDATES, PROBES
from analysis.complexity_model.sampler import Sample
from analysis.complexity_model.scoring import Action, pair as pair_score
from pre_processing.trajectory_analyzer import analyze

FAMILY_OF = {n: ("gpt-5.6" if n.startswith("gpt") else n.rsplit("-", 1)[0]) for n in CANDIDATES}
PROVIDER_OF = {n: ("openai" if n.startswith("gpt") else "anthropic") for n in CANDIDATES}

#: Regime indicators appended to every row — see the module docstring.
REGIME_FEATURE_NAMES: tuple[str, ...] = ("a_is_logged", "b_is_logged")

ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + PAIR_FEATURE_NAMES + REGIME_FEATURE_NAMES


def load_samples(path: str | Path) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                samples[record["step_id"]] = Sample(**record)
    return samples


def tools_of(sample: Sample) -> dict[str, ToolDef]:
    return {t.get("name"): convert_tool(t) for t in sample.tools}


def trajectory_for(sample: Sample):
    """The prefix as a `Trajectory` — the object `features.py` reads at inference too."""
    return analyze(
        {"model": sample.served_model, "input": sample.prefix_items, "tools": sample.tools},
        id=sample.trajectory_id,
    )


@dataclass
class StepActions:
    """Everything observable at one step: the elicited probes, the logged action, and any
    self-pair second draws."""

    elicited: dict[str, Action]
    logged_model: str
    logged: Action | None
    self_pairs: dict[str, Action]


def actions_for_step(sample: Sample, store: dict, tools: dict[str, ToolDef]) -> StepActions:
    elicited: dict[str, Action] = {}
    self_pairs: dict[str, Action] = {}
    for key_model, record in ((m, store.get((m, sample.step_id))) for m in _keys_for(store, sample.step_id)):
        if record is None or record.get("status") != "ok":
            continue
        base = base_model_of(key_model)
        action = parse_response(base, record["raw"], tools)
        if key_model.endswith(SELF_PAIR_SUFFIX):
            self_pairs[base] = action
        else:
            elicited[base] = action

    logged = logged_action_for(sample.served_model, sample.prefix_items + sample.logged_action_items,
                               len(sample.prefix_items), tools).action if sample.logged_action_items else None
    return StepActions(elicited, sample.served_model, logged, self_pairs)


def _keys_for(store: dict, step_id: str) -> list[str]:
    return [m for (m, s) in store if s == step_id]


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]
    row_pairs: list[tuple[str, str]] | None = None
    """The ordered (a, b) each row describes — needed for leave-one-probe-out."""


def _row(step_features: dict, a: str, b: str, a_logged: bool, b_logged: bool) -> list[float]:
    desc = pair_descriptor(a, b, BASE_CAPABILITY, FAMILY_OF, PROVIDER_OF)
    return (
        [step_features[n] for n in FEATURE_NAMES]
        + [desc[n] for n in PAIR_FEATURE_NAMES]
        + [float(a_logged), float(b_logged)]
    )


def build_dataset(samples_path, responses_path, probes: tuple[str, ...] = PROBES):
    """One row per (step, ordered measured pair). Returns the dataset, diagnostics, the
    raw observations needed for calibration, and the pooled agreement cells."""
    samples = load_samples(samples_path)
    store = load_store(responses_path)

    diag = {"steps_total": len(samples), "steps_incomplete": 0, "steps_used": 0,
            "rows_probe_probe": 0, "rows_logged_probe": 0}
    rows: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    row_pairs: list[tuple[str, str]] = []

    logged_vs_elicited: dict[str, list[float]] = defaultdict(list)
    self_pair_vals: dict[str, list[float]] = defaultdict(list)
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)

    for step_id, sample in samples.items():
        tools = tools_of(sample)
        acts = actions_for_step(sample, store, tools)
        if set(acts.elicited) != set(probes):
            diag["steps_incomplete"] += 1
            continue

        feats = extract_step_features(trajectory_for(sample))

        for a in probes:                                    # probe vs probe
            for b in probes:
                if a == b:
                    continue
                t = pair_score(acts.elicited[a], acts.elicited[b])
                rows.append(_row(feats, a, b, False, False)); targets.append(t); groups.append(step_id)
                row_pairs.append((a, b))
                cells[(a, b)].append(t); diag["rows_probe_probe"] += 1

        if acts.logged is not None:                          # logged vs probe
            m = acts.logged_model
            for b in probes:
                if m == b:
                    # same model both ways: this is the regime observation delta needs.
                    logged_vs_elicited[m].append(pair_score(acts.logged, acts.elicited[b]))
                    continue
                t_ab = pair_score(acts.logged, acts.elicited[b])
                t_ba = pair_score(acts.elicited[b], acts.logged)
                rows.append(_row(feats, m, b, True, False)); targets.append(t_ab); groups.append(step_id)
                rows.append(_row(feats, b, m, False, True)); targets.append(t_ba); groups.append(step_id)
                row_pairs.append((m, b)); row_pairs.append((b, m))
                cells[(m, b)].append(t_ab); cells[(b, m)].append(t_ba)
                diag["rows_logged_probe"] += 2

        for m, action in acts.self_pairs.items():            # noise floor
            if m in acts.elicited:
                self_pair_vals[m].append(pair_score(acts.elicited[m], action))

        diag["steps_used"] += 1

    dataset = Dataset(np.array(rows, dtype=float), np.array(targets, dtype=float),
                      np.array(groups), ALL_FEATURE_NAMES, row_pairs)
    return dataset, diag, dict(logged_vs_elicited), dict(self_pair_vals), dict(cells)


def fit(dataset: Dataset, n_splits: int = 5):
    """Logit-link L2 fit on the [0,1] target.

    Not plain ridge, for two reasons that both bite here. The link **guarantees**
    predictions land in [0,1] — an unconstrained fit puts a few percent of steps outside
    the range, and an out-of-range agreement inverts the ordering downstream. And the
    target is nearly Bernoulli: ~87% of its variance is the single bit "did the two models
    pick the same action", with fractional Jaccard values carrying the rest. Still a
    regularized linear model, so §6.3's capacity argument holds.

    Cross-validated **by step**, never by row: the 12 rows from one step share every step
    feature, so a random split leaks the step across folds and reports an optimistic score.
    """
    n_groups = len(set(dataset.groups))
    if n_groups < 2 or len(dataset.y) == 0:
        return None, {"note": "not enough steps to fit"}

    # sample_weight lets a continuous [0,1] target train a binary-link model: each row
    # contributes mass y to the "agree" class and 1-y to "disagree".
    X = np.vstack([dataset.X, dataset.X])
    y = np.concatenate([np.ones(len(dataset.y)), np.zeros(len(dataset.y))])
    w = np.concatenate([dataset.y, 1.0 - dataset.y])
    g = np.concatenate([dataset.groups, dataset.groups])

    # Standardize first: the raw features span log-token counts (~9) to tool-output
    # lengths (~1e3), and lbfgs will not converge on that scale spread. The scaler is part
    # of the fitted object, so inference applies the identical transform.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=5000, C=1.0)),
    ])
    splits = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=splits)
    proba = cross_val_predict(model, X, y, cv=cv, groups=g, method="predict_proba",
                              params={"logit__sample_weight": w})[: len(dataset.y), 1]
    model.fit(X, y, logit__sample_weight=w)

    baseline = np.full_like(dataset.y, dataset.y.mean())
    report = {
        "n_rows": int(len(dataset.y)), "n_steps": int(n_groups), "n_splits": int(splits),
        "model_mse": float(mean_squared_error(dataset.y, proba)),
        "model_r2": float(r2_score(dataset.y, proba)),
        "baseline_mse": float(mean_squared_error(dataset.y, baseline)),
        "target_mean": float(dataset.y.mean()),
    }
    report["beats_baseline"] = report["model_mse"] < report["baseline_mse"]
    return model, report


def leave_one_probe_out(dataset: Dataset, held_out: str) -> dict:
    """Genuine held-out validation across the provider boundary.

    Fit using only the pairs that never involve `held_out`, then predict the pairs that do —
    including its **cross-provider** cells (logged Anthropic action vs the held-out probe),
    which the model has then never seen for that probe. This is the check the abandoned
    prior-distance design could not construct at all: with agreement a function of prior
    distance, every cell is determined by the same two parameters, so nothing is ever
    genuinely held out.

    It does **not** validate the Claude-vs-Claude completion — nothing available does.
    """
    if dataset.row_pairs is None:
        return {"note": "row pairs unavailable"}
    involves = np.array([held_out in pair for pair in dataset.row_pairs])
    if involves.all() or not involves.any():
        return {"note": f"cannot hold out {held_out}"}

    train = Dataset(dataset.X[~involves], dataset.y[~involves], dataset.groups[~involves],
                    dataset.feature_names)
    model, _ = fit(train)
    if model is None:
        return {"note": "training split too small"}

    X_test, y_test = dataset.X[involves], dataset.y[involves]
    pred = model.predict_proba(X_test)[:, 1]
    baseline = np.full_like(y_test, train.y.mean())
    return {
        "held_out_probe": held_out,
        "n_train_rows": int((~involves).sum()),
        "n_test_rows": int(involves.sum()),
        "test_mse": float(mean_squared_error(y_test, pred)),
        "baseline_mse": float(mean_squared_error(y_test, baseline)),
        "beats_baseline": bool(mean_squared_error(y_test, pred) < mean_squared_error(y_test, baseline)),
    }


def save_artifact(model, dataset, report, delta, cells, out_dir) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scaler = model.named_steps["scale"]
    logit = model.named_steps["logit"]
    artifact = {
        "feature_names": list(dataset.feature_names),
        "coef": logit.coef_[0].tolist(),
        "intercept": float(logit.intercept_[0]),
        # The scaler is part of the model: inference must apply the identical transform
        # before the linear term, so its parameters ship with the coefficients.
        "scale_mean": scaler.mean_.tolist(),
        "scale_std": scaler.scale_.tolist(),
        "link": "logit",
        "probes": list(PROBES),
        "training_report": report,
        "delta": {"per_model": delta.per_model, "noise_floor": delta.noise_floor,
                  "mean": delta.mean, "spread": delta.spread,
                  "is_consistent": delta.is_consistent},
        "measured_cells": {f"{a}|{b}": float(np.mean(v)) for (a, b), v in cells.items()},
        "measured_cell_counts": {f"{a}|{b}": len(v) for (a, b), v in cells.items()},
    }
    path = out_dir / "pair_model.json"
    path.write_text(json.dumps(artifact, indent=2))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--responses", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "artifacts")
    args = p.parse_args()

    dataset, diag, lve, sp, cells = build_dataset(args.samples, args.responses)
    print(f"[train] {diag}")
    if len(dataset.y) == 0:
        print("[train] no usable rows — is the response store populated?")
        return 1

    delta = estimate_delta(lve, sp)
    print(delta.report())

    model, report = fit(dataset)
    print(f"[train] {report}")
    if model is None:
        return 1
    if not report.get("beats_baseline", True):
        print("[train] WARNING: did not beat the constant predictor (§9.6). "
              "Stop and revisit features before building on top.")

    for probe in PROBES:
        print(f"[train] leave-one-probe-out: {leave_one_probe_out(dataset, probe)}")

    path = save_artifact(model, dataset, report, delta, cells, args.out_dir)
    print(f"[train] artifact -> {path}")
    print(f"[train] measured cells: {len(cells)} of {len(CANDIDATES)*(len(CANDIDATES)-1)} ordered pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
