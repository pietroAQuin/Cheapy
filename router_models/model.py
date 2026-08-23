"""Combine `price_score` and `performance_score` into a routing decision.

Per the README's architecture (§2), this is the last stage of the scoring chain: it
reads the `price_score` and `performance_score` that `router_models/price_model.py` and
`router_models/performance_model.py` already set on each `ModelLLM`, combines them into
`final_score`, ranks the candidate pool, and turns the top pick into a HOLD /
CHANGE TO decision relative to `trajectory.served_model`.

The weights are a parameter threaded through every call here, not a module-level
constant (README §5 "Aggregation") -- sweeping them across calls is how a cost-quality
frontier gets produced instead of a single operating point.
"""

from __future__ import annotations

from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory


def score_final(
    models: list[ModelLLM], w_price: float = 0.5, w_perf: float = 0.5
) -> list[ModelLLM]:
    """Set `final_score = w_price * price_score + w_perf * performance_score`.

    Both inputs must already be scored (not `None`) -- per the shared contract in
    README §5, an unset score must never be treated as 0, so this raises rather than
    silently defaulting one in if a caller skipped a scoring stage.
    """
    for model in models:
        if model.price_score is None or model.performance_score is None:
            raise ValueError(
                f"{model.name!r} is missing price_score or performance_score -- run "
                "both scoring stages before score_final()"
            )
        model.final_score = w_price * model.price_score + w_perf * model.performance_score
    return models


def rank(models: list[ModelLLM]) -> list[ModelLLM]:
    """Sort `models` by `final_score`, best first. Requires `score_final` to have run."""
    return sorted(models, key=lambda model: model.final_score, reverse=True)


def decide(trajectory: Trajectory, ranked_models: list[ModelLLM]) -> str:
    """HOLD if the top-ranked model is already serving the trajectory, else CHANGE TO."""
    top = ranked_models[0]
    if top.name == trajectory.served_model:
        return "HOLD"
    return f"CHANGE TO {top.name}"


def route(
    trajectory: Trajectory,
    models: list[ModelLLM],
    w_price: float = 0.5,
    w_perf: float = 0.5,
) -> tuple[list[ModelLLM], str]:
    """Score, rank, and decide in one call. Returns (ranked_models, decision).

    `models` must already carry `price_score` and `performance_score` from the earlier
    stages in the chain (README §2) -- this stage only combines and ranks them.
    """
    score_final(models, w_price=w_price, w_perf=w_perf)
    ranked_models = rank(models)
    decision = decide(trajectory, ranked_models)
    return ranked_models, decision
