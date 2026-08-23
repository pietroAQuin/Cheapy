"""`router.py` — the last stage: weighted aggregation, ranking, HOLD / CHANGE TO.

Per docs/FULL_REPORT.md §2 this stage takes the `ModelLLM` objects the two scoring stages already
enriched and combines their scores into one number::

    final_score = (W_COST · price_score + W_PERFORMANCE · performance_score)
                  / (W_COST + W_PERFORMANCE)

The division is what keeps `final_score` inside `[0, 1]` alongside its two inputs, so
weights can be passed as any pair of non-negative numbers (`0.7 / 0.3`, `7 / 3`, `1 / 0`)
without changing the scale the ranking is read on.

**Weights are arguments, never module constants** (docs/FULL_REPORT.md §5, "Aggregation"): sweeping
them is how the cost/quality frontier gets produced, so `aggregate_scores` and `route`
require them explicitly rather than defaulting to one operating point. The only default
here is `min_gain`, which is a switching threshold, not an operating point.

`served_model` is read exactly once, in `decide`, to compare the winner against the
incumbent — that is the decision itself, not a scoring feature. The scores it ranks were
produced without it (`performance_model`), or with the one switching-cost carve-out
docs/FULL_REPORT.md §5 grants (`price_model`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cheapy.models.llm import ModelLLM
from cheapy.models.trajectory import Trajectory
from cheapy.routing.performance_model import DEFAULT_BETA, score_performance
from cheapy.routing.price_model import DEFAULT_PRICE_EXPONENT, score_price

__all__ = ["ScoreRow", "RoutingDecision", "aggregate_scores", "rank", "decide", "route"]

Action = Literal["HOLD", "CHANGE"]


@dataclass(frozen=True)
class ScoreRow:
    """One candidate's line on a trajectory's scoreboard.

    A flat snapshot taken at `decide` time, because the `ModelLLM` objects it comes from
    are rebuilt and re-scored for the next trajectory — holding on to them would hand a
    later reader the wrong trajectory's numbers.
    """

    name: str
    final_score: float
    price_score: float
    performance_score: float
    is_served: bool
    """True for the model that actually served this trajectory (the incumbent)."""


@dataclass(frozen=True)
class RoutingDecision:
    """What the router concluded for one trajectory's next call."""

    trajectory_id: int
    served_model: str
    top_model: str
    action: Action
    label: str
    """`"HOLD"` or `"CHANGE TO <model>"` — the docs/FULL_REPORT.md §1 verdict string."""

    top_final_score: float
    top_price_score: float
    top_performance_score: float
    served_final_score: float | None
    """`None` when the incumbent was not in the candidate pool, or went unscored."""
    score_gap: float
    """`top_final_score - served_final_score`; `0.0` when the incumbent is the winner.

    Also the "how much does this decision matter here" signal: a gap near zero means any
    candidate would do and the choice is nearly free either way."""

    scoreboard: tuple[ScoreRow, ...]
    """Every scored candidate, best first, with the two inputs behind each `final_score`.

    This is what `--verbose` prints. It is kept on the decision rather than recomputed by
    the caller so the printed board is, by construction, the board the verdict came from.
    """
    unscored: tuple[str, ...]
    """Candidates left out of the ranking because a stage did not score them."""

    @property
    def ranking(self) -> tuple[str, ...]:
        """Every scored candidate by name, best first."""
        return tuple(row.name for row in self.scoreboard)


def aggregate_scores(
    models: list[ModelLLM], *, w_cost: float, w_performance: float
) -> list[ModelLLM]:
    """Set `final_score` in place on every fully-scored `ModelLLM` and return `models`.

    A model missing either input keeps `final_score = None`. It is **not** scored on the
    half it has: an unset score means "this stage did not score this model", and both
    `ModelLLM` docstrings are explicit that it must never be read as 0 — doing so would
    rank an unscored model as the worst possible one and quietly drop it out of
    contention on evidence that was never gathered.
    """
    if w_cost < 0 or w_performance < 0:
        raise ValueError(f"weights must be >= 0, got {w_cost} / {w_performance}")
    total = w_cost + w_performance
    if total == 0:
        raise ValueError("w_cost + w_performance must be > 0 — no ranking is defined at 0/0")

    for model in models:
        if model.price_score is None or model.performance_score is None:
            model.final_score = None
            continue
        model.final_score = (
            w_cost * model.price_score + w_performance * model.performance_score
        ) / total
    return models


def rank(models: list[ModelLLM], served_model: str | None = None) -> list[ModelLLM]:
    """Scored candidates, best `final_score` first.

    Ties break toward `served_model` and then by name. Both tie-breaks exist to make the
    ranking deterministic, and the first also keeps a dead-even score from reading as a
    reason to switch: switching has costs the score does not fully capture (the reset
    prefix cache is only the measurable one), so an exact tie resolves to the incumbent.
    """
    scored = [model for model in models if model.final_score is not None]
    return sorted(
        scored,
        key=lambda m: (-m.final_score, m.name != served_model, m.name),  # type: ignore[operator]
    )


def decide(
    trajectory: Trajectory, models: list[ModelLLM], *, min_gain: float = 0.0
) -> RoutingDecision:
    """Rank the already-aggregated `models` and emit the HOLD / CHANGE TO verdict.

    `min_gain` is the `final_score` improvement a challenger must show over the incumbent
    before the switch is worth making — a deadband, not a scoring weight. At the default
    `0.0` any improvement at all flips the decision to CHANGE; raise it to suppress
    switches whose predicted gain is inside the noise of an estimate built on estimated
    tokens and predicted agreement.
    """
    if min_gain < 0:
        raise ValueError(f"min_gain must be >= 0, got {min_gain}")

    ranked = rank(models, trajectory.served_model)
    if not ranked:
        raise ValueError(
            f"trajectory {trajectory.id}: no candidate has both a price_score and a "
            "performance_score — run score_price and score_performance first"
        )

    top = ranked[0]
    served = next(
        (m for m in ranked if m.name == trajectory.served_model), None
    )
    served_final = served.final_score if served is not None else None

    # An unranked incumbent leaves no baseline to beat, so the ranking's own winner
    # stands: gain is undefined, not zero, and min_gain cannot be applied to it.
    gain = top.final_score - served_final if served_final is not None else None  # type: ignore[operator]
    holds = top.name == trajectory.served_model or (gain is not None and gain <= min_gain)

    if holds and served is not None:
        top = served

    return RoutingDecision(
        trajectory_id=trajectory.id,
        served_model=trajectory.served_model,
        top_model=top.name,
        action="HOLD" if holds else "CHANGE",
        label="HOLD" if holds else f"CHANGE TO {top.name}",
        top_final_score=top.final_score,  # type: ignore[arg-type]
        top_price_score=top.price_score,  # type: ignore[arg-type]
        top_performance_score=top.performance_score,  # type: ignore[arg-type]
        served_final_score=served_final,
        score_gap=0.0 if holds else gain if gain is not None else 0.0,
        scoreboard=tuple(
            ScoreRow(
                name=m.name,
                final_score=m.final_score,  # type: ignore[arg-type]
                price_score=m.price_score,  # type: ignore[arg-type]
                performance_score=m.performance_score,  # type: ignore[arg-type]
                is_served=m.name == trajectory.served_model,
            )
            for m in ranked
        ),
        unscored=tuple(m.name for m in models if m.final_score is None),
    )


def route(
    trajectory: Trajectory,
    models: list[ModelLLM],
    *,
    w_cost: float,
    w_performance: float,
    beta: float = DEFAULT_BETA,
    price_exponent: float = DEFAULT_PRICE_EXPONENT,
    min_gain: float = 0.0,
) -> RoutingDecision:
    """The whole scoring chain for one trajectory, in docs/FULL_REPORT.md §2's order.

    price -> performance -> aggregate -> decide, each stage enriching the same `models`
    in place. `models` must be this trajectory's own list: the stages mutate the objects,
    so a list shared between trajectories would carry the previous one's scores.
    """
    score_price(trajectory, models, price_exponent=price_exponent)
    score_performance(trajectory, models, beta=beta)
    aggregate_scores(models, w_cost=w_cost, w_performance=w_performance)
    return decide(trajectory, models, min_gain=min_gain)
