"""Assign `price_score` to each candidate `ModelLLM` for one `Trajectory`.

Per the README's scoring contract (§5): `price_score` is `[0, 1]`, higher = cheaper,
normalized *within this trajectory's candidate pool* — not against some global price
scale. This module is a pure function of `(Trajectory, List[ModelLLM])`; the only place
it reads `Trajectory.served_model` is the switching-cost check below, which the README
explicitly carves out as legitimate (it is not a task-quality proxy).

The crux, per README §5: providers cache the shared input prefix across a trajectory's
calls, and a model switch resets that cache. So "cheaper per token" and "cheaper for
the next call" are different questions once a trajectory has accumulated context — a
model switch pays full uncached price for the whole accumulated prefix, which in long
trajectories can dwarf any per-token saving. That's the entire reason this module needs
`served_model` at all: to tell a HOLD's mostly-cached next call apart from a CHANGE's
fully-uncached one.

`total_cached_tokens` is used directly as the already-cached share of that accumulated
context — see `_next_call_tokens` for why, and why an earlier version of this module
was wrong to instead assume the entire accumulated context was cached under HOLD.
"""

from __future__ import annotations

from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory


def _next_call_tokens(trajectory: Trajectory) -> tuple[float, float, float]:
    """ESTIMATE the shape of the trajectory's next call.

    Returns (accumulated_context_tokens, increment_tokens, next_output_tokens).

    `accumulated_context_tokens` is `Trajectory.total_tokens` taken at face value: the
    total input tokens billed across the trajectory's calls so far, before any cache
    discount (see that field's docstring). This is a *cumulative* total across every
    past call, not the size of any one prompt — a trajectory's calls each resend the
    whole history, so `total_tokens` is already the whole-trajectory-to-date footprint
    a switch has to re-pay. Treated this way, the "next call" estimate below reads as
    "the switch penalty at stake right now", which is what the cache trap is actually
    about, at the cost of overstating the literal byte size of one upcoming API call —
    see the README's "Known simplifications" for the size of that overstatement on long
    trajectories.

    `total_cached_tokens` is a share of that same `total_tokens` (again, by its own
    docstring) — the part of it already known to be cache-eligible. An earlier version
    of this function computed `total_tokens - total_cached_tokens` and called *that* the
    "current context", then separately assumed the result was 100% cached under HOLD.
    That double-used `total_cached_tokens` in contradictory ways: first to define a
    quantity as "the not-yet-cached remainder", then to treat that same remainder as
    fully cached. This version keeps `total_tokens` and `total_cached_tokens` as the two
    numbers they actually are — a whole and its already-cached share — and never
    re-derives one from the other.

    `increment_tokens` approximates how much *new* content (tool outputs, replies, ...)
    a typical call in this trajectory appends, as the trajectory's own average per-call
    growth (accumulated context / calls so far). There is no better signal available:
    the next call's actual new content is by definition not in the log yet.
    """
    accumulated_context_tokens = float(trajectory.total_tokens)
    increment_tokens = (
        accumulated_context_tokens / trajectory.total_calls if trajectory.total_calls else 0.0
    )
    next_output_tokens = trajectory.avg_output_tokens_per_call
    return accumulated_context_tokens, increment_tokens, next_output_tokens


def _next_call_cost(trajectory: Trajectory, model: ModelLLM) -> float | None:
    """ESTIMATE the USD cost of `model` serving this trajectory's next call.

    Returns None if `model.context_window_size` can't hold the next call at all —
    that candidate is infeasible, not merely expensive, and is handled separately.
    Note the feasibility check compares against `accumulated_context_tokens` (the whole
    trajectory's cumulative footprint, per `_next_call_tokens`), which is stricter than
    the real constraint a context window enforces (the size of one actual prompt) — see
    the README's "Known simplifications".
    """
    accumulated_context_tokens, increment_tokens, next_output_tokens = _next_call_tokens(
        trajectory
    )
    next_input_tokens = accumulated_context_tokens + increment_tokens

    if next_input_tokens + next_output_tokens > model.context_window_size:
        return None

    holding = model.name == trajectory.served_model
    if holding:
        # Same model as the log: bill the trajectory's own already-cached share at the
        # cached rate, and everything else — the never-cached remainder of the
        # accumulated context, plus the new increment — at the normal rate.
        cached_tokens = min(float(trajectory.total_cached_tokens), accumulated_context_tokens)
        uncached_tokens = (accumulated_context_tokens - cached_tokens) + increment_tokens
    else:
        # A switch resets the provider's prefix cache: the whole next prompt is
        # uncached, not just the increment. This is the cache trap from README §5.
        cached_tokens = 0.0
        uncached_tokens = next_input_tokens

    input_cost = (
        uncached_tokens * model.input_price_per_1m
        + cached_tokens * model.cached_input_price_per_1m
    ) / 1_000_000
    output_cost = next_output_tokens * model.output_price_per_1m / 1_000_000
    return input_cost + output_cost


def score_price(trajectory: Trajectory, models: list[ModelLLM]) -> list[ModelLLM]:
    """Enrich each `ModelLLM` in `models` with a `price_score` in `[0, 1]`.

    Higher is cheaper. Costs are estimated per `_next_call_cost` and min-max
    normalized across `models` — this trajectory's candidate pool only, per README
    §5's "normalize within the candidate pool, not globally". Infeasible candidates
    (next call wouldn't fit their context window) are scored 0.0 and excluded from the
    min-max range so one unusable model doesn't compress everyone else's spread.
    """
    costs: dict[str, float | None] = {
        model.name: _next_call_cost(trajectory, model) for model in models
    }

    feasible_costs = [cost for cost in costs.values() if cost is not None]
    if not feasible_costs:
        for model in models:
            model.price_score = 0.0
        return models

    cheapest, priciest = min(feasible_costs), max(feasible_costs)
    spread = priciest - cheapest

    for model in models:
        cost = costs[model.name]
        if cost is None:
            model.price_score = 0.0
        elif spread == 0:
            model.price_score = 1.0
        else:
            model.price_score = (priciest - cost) / spread

    return models
