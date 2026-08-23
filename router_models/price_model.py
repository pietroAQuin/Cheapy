"""Assign `price_score` to each candidate `ModelLLM` for one `Trajectory`.

Per the README's scoring contract (§5): `price_score` is `[0, 1]`, higher = cheaper,
normalized *within this trajectory's candidate pool* — not against some global price
scale. This module is a pure function of `(Trajectory, List[ModelLLM])`; the only place
it reads `Trajectory.served_model` is the switching-cost check below, which the README
explicitly carves out as legitimate (it is not a task-quality proxy).

The crux, per README §5: providers cache the shared input prefix across a trajectory's
calls, and a model switch resets that cache. So "cheaper per token" and "cheaper for
the next call" are different questions once a trajectory has accumulated context — a
model switch pays full uncached price for the whole next prompt, which in long
trajectories can dwarf any per-token saving. That's the entire reason this module needs
`served_model` at all: to tell a HOLD's mostly-cached next call apart from a CHANGE's
fully-uncached one.

The base for "how big is the next prompt" is `Trajectory.last_call_input_tokens`, not
`total_tokens` -- see `_next_call_tokens` for why: every call resends the whole history
up to that point, so the next call's size tracks the size of the *most recent* call, not
the sum of every past call's own (already-superseded) full-history snapshot.
"""

from __future__ import annotations

from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory


def _next_call_tokens(trajectory: Trajectory) -> tuple[float, float, float]:
    """ESTIMATE the shape of the trajectory's next call.

    Returns (last_call_input_tokens, increment_tokens, next_output_tokens).

    `last_call_input_tokens` is `Trajectory.last_call_input_tokens` taken at face value:
    the size of the most recently recovered call's full prompt (tool schemas + history
    up to that point). Because every call resends the whole history, this is the best
    available proxy for the size of the *next* call's prompt.

    This deliberately does NOT use `total_tokens`. `total_tokens` sums every past call's
    prompt -- each of which was itself a full-history resend at the time -- so it adds up
    several superseded snapshots of a growing thing instead of reporting the size of the
    latest one. On a trajectory whose prompt grows roughly linearly over `n` calls,
    `total_tokens` overstates the real next-call size by very roughly a factor of `n/2`,
    and can trip the context-window feasibility gate below for a model that would
    actually serve the real next call fine. `last_call_input_tokens` doesn't have that
    problem: it's one call's real size, not an accumulating sum.

    `increment_tokens` approximates how much *new* content (tool outputs, replies, ...)
    a typical call in this trajectory appends, as the trajectory's own average per-call
    growth (`last_call_input_tokens / total_calls`) -- not `total_tokens / total_calls`,
    which is the same superseded-snapshots mistake `last_call_input_tokens` exists to
    avoid, just applied to the increment instead of the prompt size: it averages in
    every past call's own (smaller, since-outgrown) prompt size, not how much this
    trajectory actually grows per step. There is no better signal available for the
    increment itself: the next call's actual new content is by definition not in the
    log yet. This average is a separate, smaller known simplification (it can over- or
    under-estimate a single step depending on whether the trajectory's growth is
    front- or back-loaded) -- see the README's "Known simplifications".
    """
    last_call_input_tokens = float(trajectory.last_call_input_tokens)
    increment_tokens = (
        last_call_input_tokens / trajectory.total_calls if trajectory.total_calls else 0.0
    )
    next_output_tokens = trajectory.avg_output_tokens_per_call
    return last_call_input_tokens, increment_tokens, next_output_tokens


def _next_call_cost(trajectory: Trajectory, model: ModelLLM) -> float:
    """ESTIMATE the USD cost of `model` serving this trajectory's next call.

    No candidate is ever excluded for having too small a context window. If
    `last_call_input_tokens` is bigger than what `model` can hold, this assumes the
    context gets compacted down to the model's max `context_window_size` before the
    call is sent -- clamp the size, then price it with the exact same cache-aware
    formula used for every other call, holding or switching alike.
    """
    last_call_input_tokens, increment_tokens, next_output_tokens = _next_call_tokens(
        trajectory
    )
    last_call_input_tokens = min(last_call_input_tokens, float(model.context_window_size))
    next_input_tokens = last_call_input_tokens + increment_tokens

    holding = model.name == trajectory.served_model
    if holding:
        # Same model as the log: the last call's whole prompt was already sent once and
        # is cache-eligible; only the new increment is guaranteed uncached. Cap at
        # `last_call_input_tokens` -- `total_cached_tokens` is a whole-trajectory
        # aggregate and can exceed the size of any single call, but only that much of it
        # is actually part of *this* call's bill.
        cached_tokens = min(float(trajectory.total_cached_tokens), last_call_input_tokens)
        uncached_tokens = (last_call_input_tokens - cached_tokens) + increment_tokens
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


def estimate_costs(trajectory: Trajectory, models: list[ModelLLM]) -> dict[str, float]:
    """ESTIMATE the USD cost of each candidate in `models` serving the trajectory's next
    call, keyed by model name — see `_next_call_cost`.

    Public so callers other than `score_price` -- e.g. a benchmark comparing policies by
    actual dollar cost rather than the normalized `price_score` -- can get the same
    per-candidate cost figures `score_price` normalizes, without duplicating the
    cache-trap cost logic in `_next_call_cost`.
    """
    return {model.name: _next_call_cost(trajectory, model) for model in models}


def score_price(trajectory: Trajectory, models: list[ModelLLM]) -> list[ModelLLM]:
    """Enrich each `ModelLLM` in `models` with a `price_score` in `(0, 1]`.

    Higher is cheaper. Costs are estimated per `_next_call_cost` (via `estimate_costs`)
    and scored as a ratio to the *cheapest* candidate in this trajectory's candidate
    pool — per README §5's "normalize within the candidate pool, not globally" — rather
    than min-max stretched across the pool's full price range. A candidate priced at `k`
    times the cheapest one scores `1/k`: a candidate only marginally above the cheapest
    price lands close to `1.0`, not at the same `0.0` floor a wildly pricier candidate
    would get under min-max. It also depends only on the cheapest cost, not the
    priciest, so a candidate's score no longer shifts just because some unrelated,
    pricier (or cheaper) candidate joins the pool. Every candidate gets a real cost (a
    too-small context window is compacted to fit, never excluded — see
    `_next_call_cost`), so every model in the pool gets a ratio-based score.
    """
    costs = estimate_costs(trajectory, models)
    cheapest = min(costs.values())

    for model in models:
        cost = costs[model.name]
        # cost == 0 only when cheapest == 0 too (costs are never negative), so this is
        # the tie-at-the-top case, not a magic number -- 0/0 is otherwise undefined.
        model.price_score = 1.0 if cost == 0 else cheapest / cost

    return models
