"""`capability_model.py` — the shipped artifact, spec §7 as revised by the pivot.

    trajectory (the router's own, or a prefix-Trajectory built offline)
      -> step features (features.py — the same function used to train)
      -> predict pair(i, probe) for all 9 models x 3 probes, plus probe x probe
      -> Nystrom-complete the Claude-vs-Claude block (completion.py)
      -> score(i) = ( w_i + sum_j w_j * pair(i,j) ) / sum_all_j w_j
      -> { model_name: score }, plus a per-model measured/extrapolated breakdown

No API calls, no network, milliseconds. `router_models/performance_model.py` is the
intended caller.

**Two tiers of evidence, and the caller is told which is which.** The three probe columns
are predicted by a model fitted on measured data — including for the six Anthropic
candidates, whose real actions were recovered from the corpus rather than queried. The 15
Claude-vs-Claude cells are *completed*, never observed, and capped at rank 3 by having only
three probes. `ModelScore.measured_fraction` reports the split so nothing downstream
presents nine equally-earned numbers.

**Stub mode**: with no fitted artifact under `artifacts/`, `score_models` returns documented
constants so `router_models/model.py` integration is unblocked (§9.3). Once `train.py`
writes `artifacts/pair_model.json` this module picks it up with no call-site change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from analysis.complexity_model.completion import nystrom_complete
from analysis.complexity_model.features import extract_step_features, pair_descriptor
from analysis.complexity_model.priors import BASE_CAPABILITY, CANDIDATES, NEAR_UNMEASURED, PROBES, weights
from data_models.Trajectory import Trajectory

_ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "pair_model.json"

FAMILY_OF = {n: ("gpt-5.6" if n.startswith("gpt") else n.rsplit("-", 1)[0]) for n in CANDIDATES}
PROVIDER_OF = {n: ("openai" if n.startswith("gpt") else "anthropic") for n in CANDIDATES}

#: Stub pair value — the middle of the type table, not 1.0 or 0.0. A stub at 1.0 would make
#: every step look uncontested; at 0.0, maximally contested. Replaced wholesale by a fitted
#: artifact, never blended with one.
_STUB_PAIR_VALUE = 0.7

#: Default conformity weighting. Lives here, next to the scoring it parameterises, and is
#: re-exported by `router_models/performance_model.py` as `DEFAULT_BETA` — see that module
#: for the sweep evidence behind the value.
DEFAULT_BETA = 3.0


@dataclass(frozen=True)
class ModelScore:
    score: float
    measured_fraction: float
    """Share of this model's pairs that come from the fitted (measured-data) path rather
    than from the completion. 1.0 for probes; 3/8 for the Anthropic candidates."""
    near_unmeasured: bool
    """True for models that served almost no trajectories (see `priors.NEAR_UNMEASURED`) —
    their measured cells rest on ~1 task, so the flag travels with the score."""


@dataclass(frozen=True)
class Artifact:
    feature_names: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    scale_mean: np.ndarray
    scale_std: np.ndarray


@lru_cache(maxsize=1)
def _load() -> Artifact | None:
    if not _ARTIFACT_PATH.exists():
        return None
    d = json.loads(_ARTIFACT_PATH.read_text())
    return Artifact(
        feature_names=tuple(d["feature_names"]),
        coef=np.array(d["coef"], dtype=float),
        intercept=float(d["intercept"]),
        scale_mean=np.array(d["scale_mean"], dtype=float),
        scale_std=np.array(d["scale_std"], dtype=float),
    )


def _predict(art: Artifact, step_features: dict[str, float], a: str, b: str) -> float:
    """One predicted ordered pair value. Regime indicators are 0: at inference nothing is
    logged, so the model is asked for the elicited-regime value even for candidates whose
    training rows were all logged."""
    desc = pair_descriptor(a, b, BASE_CAPABILITY, FAMILY_OF, PROVIDER_OF)
    row = np.array(
        [{**step_features, **desc, "a_is_logged": 0.0, "b_is_logged": 0.0}[n] for n in art.feature_names],
        dtype=float,
    )
    z = (row - art.scale_mean) / np.where(art.scale_std == 0, 1.0, art.scale_std)
    return 1.0 / (1.0 + math.exp(-float(np.dot(z, art.coef) + art.intercept)))


def pair_matrix(
    trajectory: Trajectory, candidates: tuple[str, ...] = CANDIDATES
) -> tuple[dict[tuple[str, str], float], set[tuple[str, str]]]:
    """Full ordered pair matrix over `candidates`, plus the set of *completed* (never
    observed) cells."""
    art = _load()
    probes = tuple(p for p in PROBES if p in candidates)

    if art is None or not probes:
        pairs = {(a, b): _STUB_PAIR_VALUE for a in candidates for b in candidates if a != b}
        return pairs, set()

    feats = extract_step_features(trajectory)
    pairs: dict[tuple[str, str], float] = {}

    # Tier 1 — every model against every probe. Fitted on measured data.
    for a in candidates:
        for b in probes:
            if a != b:
                pairs[(a, b)] = _predict(art, feats, a, b)
        if a not in probes:
            for b in probes:
                pairs[(b, a)] = _predict(art, feats, b, a)

    # Tier 2 — the block with no probe on either side, completed from the probe columns.
    others = [m for m in candidates if m not in probes]
    completed: set[tuple[str, str]] = set()
    if others:
        # Diagonal is the model's agreement with itself. The probes do not accept
        # temperature=0, so this is a measured noise floor well below 1.0 — using 1.0 would
        # overstate how much structure the landmarks pin down.
        probe_block = np.array(
            [[1.0 if p == q else pairs[(p, q)] for q in probes] for p in probes], dtype=float
        )
        cross_block = np.array([[pairs[(m, p)] for p in probes] for m in others], dtype=float)
        result = nystrom_complete(probe_block, cross_block)
        for i, a in enumerate(others):
            for j, b in enumerate(others):
                if a != b:
                    pairs[(a, b)] = float(result.matrix[i, j])
                    completed.add((a, b))
    return pairs, completed


def score_models(
    trajectory: Trajectory, beta: float, candidates: tuple[str, ...] = CANDIDATES
) -> dict[str, float]:
    """Capability score per model — spec §5.1, unchanged. `beta` stays a runtime argument
    so the conformity/prior tradeoff is swept without retraining."""
    return {name: s.score for name, s in score_models_detailed(trajectory, beta, candidates).items()}


def score_models_detailed(
    trajectory: Trajectory, beta: float, candidates: tuple[str, ...] = CANDIDATES
) -> dict[str, ModelScore]:
    """As `score_models`, but each score carries how much of it rests on measurement."""
    # De-duplicate while preserving order. A repeated name would be counted once in the
    # denominator (a dict) but twice in the numerator (the loop), pushing the score above
    # 1.0 — the §5.1 bound holds only over a *set* of models.
    candidates = tuple(dict.fromkeys(candidates))
    w = weights(beta, candidates)
    pairs, completed = pair_matrix(trajectory, candidates)
    total = sum(w.values())

    out: dict[str, ModelScore] = {}
    for model in candidates:
        others = [o for o in candidates if o != model]
        agreement = sum(w[o] * pairs[(model, o)] for o in others)
        n_completed = sum(1 for o in others if (model, o) in completed)
        out[model] = ModelScore(
            score=(w[model] + agreement) / total,
            measured_fraction=(len(others) - n_completed) / len(others) if others else 1.0,
            near_unmeasured=model in NEAR_UNMEASURED,
        )
    return out


def score_for_trajectory(
    trajectory: Trajectory, models: list, beta: float = DEFAULT_BETA
) -> list:
    """Adapter for `router_models/performance_model.py`: sets `performance_score` in place
    on each `ModelLLM` and returns the list (README §2's stage contract). Models outside the
    scored set keep `performance_score = None` — never 0."""
    candidates = tuple(m.name for m in models if m.name in BASE_CAPABILITY)
    scores = score_models(trajectory, beta, candidates) if candidates else {}
    for model in models:
        if model.name in scores:
            model.performance_score = scores[model.name]
    return models
