"""Assign `performance_score` to each candidate `ModelLLM` for one `Trajectory`.

MOCK — this is a placeholder standing in for the real performance model described in
the README's §5 "Performance" section (observable-structure proxies, or judge-model
rescoring). It exists so `router_models/model.py`'s weighted aggregation has a second,
real (non-constant) input to combine with `price_score`, which in turn is what makes a
cost-quality frontier — swept over `w_price`/`w_perf` — a *curve* instead of a single
point collapsed onto the price axis.

The mock's capability signal is each candidate's own price: a model's per-token rate is
treated as the market's bet on its capability, so a pricier model gets a higher score.
This is a real, if crude, heuristic (providers generally do price newer/more-capable
tiers higher) and it has one deliberate, useful property for this router: it is a pure
function of `ModelLLM`'s static price fields, so it never reads `Trajectory` at all --
not even `served_model` -- and so cannot leak the served model into the score by
construction, unlike a structural proxy derived from trajectory content would risk doing
(see the README §4 leakage table).

Replace this module with a real proxy (or judge-model rescoring) per README §5 once one
is designed; nothing downstream needs to change when it's swapped, since the contract
--(Trajectory, List[ModelLLM]) -> List[ModelLLM], performance_score in [0, 1]-- stays
the same.
"""

from __future__ import annotations

from data_models.model_llm import ModelLLM
from data_models.Trajectory import Trajectory


def _capability_proxy(model: ModelLLM) -> float:
    """MOCK capability signal: a model's own blended per-token rate.

    Input tokens dominate this workload roughly 130:1 over output (README §5), so the
    blend is weighted accordingly rather than split evenly -- an evenly-split blend would
    let a model with a cheap input rate but a wildly marked-up output rate look more
    capable than its actual (mostly input-bound) usage pattern here would suggest.
    """
    return 0.9 * model.input_price_per_1m + 0.1 * model.output_price_per_1m


def score_performance(trajectory: Trajectory, models: list[ModelLLM]) -> list[ModelLLM]:
    """Enrich each `ModelLLM` in `models` with a MOCK `performance_score` in `[0, 1]`.

    Higher is more capable. `trajectory` is accepted only to satisfy the shared
    (Trajectory, List[ModelLLM]) -> List[ModelLLM] stage contract (README §5) -- it is
    deliberately unused, so this mock cannot read `served_model` even by accident.
    Min-max normalized across `models`, same as `price_model.score_price`, so both
    scores land on the same comparable scale for `model.py`'s weighted average.
    """
    proxies = {model.name: _capability_proxy(model) for model in models}
    cheapest, priciest = min(proxies.values()), max(proxies.values())
    spread = priciest - cheapest

    for model in models:
        if spread == 0:
            model.performance_score = 1.0
        else:
            model.performance_score = (proxies[model.name] - cheapest) / spread

    return models
