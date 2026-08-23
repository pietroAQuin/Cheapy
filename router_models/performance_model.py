"""`performance_model.py` — sets `ModelLLM.performance_score`, per README §2/§5.

Thin adapter only. All the scoring logic — pairwise divergence, prior-weighted
conformity, the ridge regressor — lives in `analysis/complexity_model/`, whose design is
recorded in `analysis/complexity_model/complexity_scoring_spec.md`. This module exists
solely to satisfy the stage contract every `router_models/` file follows: a function of
`(Trajectory, list[ModelLLM]) -> list[ModelLLM]` that enriches the given `ModelLLM`
objects in place (README §2, "The scoring chain").

Never reads `trajectory.served_model` (README §5's "Shared contract") — neither this
module nor `capability_model.score_for_trajectory` touches it for scoring; it is read
only inside `analysis/complexity_model/` as an explicit, logged diagnostic (spec §1.0,
§8), never as a feature.
"""

from __future__ import annotations

from analysis.complexity_model.capability_model import DEFAULT_BETA, score_for_trajectory
from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory

#: Default conformity weighting (spec §4) — `w_i = prior_i ** BETA`, so BETA sets how much
#: more a high-prior model's agreement counts than a low-prior model's.
#:
#: **3.0, not 1.0.** The design has two intents, and only one of them holds at BETA = 1.
#: Measured over 200 trajectories (`analysis/complexity_model/` sweep):
#:
#:   (1) unanimity compresses scores toward 1 — holds at every BETA (spread on settled
#:       steps is ~half that on contested ones, and nothing falls below ~0.80).
#:   (2) divergence rewards conformity with the *most intelligent* model — fails at
#:       BETA = 1: rank correlation between the score ordering and `BASE_CAPABILITY` is
#:       only 0.52 on contested steps, and the model rewarded is `gpt-5.6-sol`, a
#:       mid-prior GPT that wins by sitting at the centre of an all-GPT probe panel.
#:
#: rho(score, prior) on contested steps by BETA: 0.20 (0.0), 0.52 (1.0), 0.70 (2.0),
#: 0.73 (3.0), 0.75 (5.0). BETA = 3 is the lowest value at which the highest-prior model
#: actually tops the ranking on contested steps, so it is the cheapest point that
#: satisfies intent (2) without flattening the score into the prior — rho keeps creeping
#: up past 3, but only by letting the prior dominate the measurement.
#:
#: Callers sweeping the cost/capability trade-off pass their own value straight through.
#: The value itself lives in `capability_model` and is imported, not restated, so the
#: two call sites cannot drift apart.
__all__ = ["DEFAULT_BETA", "score_performance"]


def score_performance(
    trajectory: Trajectory, models: list[ModelLLM], beta: float = DEFAULT_BETA
) -> list[ModelLLM]:
    """Set `performance_score` on every `ModelLLM` in `models` that is in the trained
    candidate set (spec §1.0's seven), leaving the rest at `None` — never `0` (see
    `ModelLLM.performance_score`'s docstring: unset must not be treated as worst-possible
    in any downstream trade-off).
    """
    return score_for_trajectory(trajectory, models, beta=beta)
