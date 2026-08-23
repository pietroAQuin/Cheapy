"""Prior weights — spec §4.

There is no reference model and no strong-model subset here. Every model is scored by
how much the others agree with it, and `prior_i` decides how much each of those others'
agreement is worth. Nothing is treated as ground truth.

The weight applied at scoring time is::

    w_i = prior_i ** BETA

`BETA` is a **runtime parameter**, never a constant baked into a fitted regressor: it is
applied in `capability_model.py` after the regressor has produced its pairwise
predictions, so changing it costs neither a retrain nor an API call.
"""

from __future__ import annotations

#: Curated 0-1 public-benchmark aggregate — the `base_capability` score of spec §4.
#: Supplied externally; this dict is the single swap point, nothing else in the package
#: hardcodes a prior.
#:
#: Under the OpenAI-only pivot the prior does **much less work** than §4 envisaged. It no
#: longer synthesizes unmeasured agreement (that approach was shown to collapse the score
#: into prior-centrality — see the module docstring of `completion.py`); it only sets the
#: conformity weights `w_i`. Agreement itself is now measured, including for the models
#: that cannot be queried, by recovering their logged actions from the corpus
#: (`logged_action.py`).
BASE_CAPABILITY: dict[str, float] = {
    "claude-opus-5": 0.9123,
    "claude-fable-5": 0.9559,
    "claude-opus-4-8": 0.6695,
    "gpt-5.6-sol": 0.7274,
    "claude-sonnet-5": 0.4192,
    "gpt-5.6-terra": 0.3491,
    "gpt-5.6-luna": 0.2326,
    "claude-opus-4-6": 0.4649,
    "claude-sonnet-4-6": 0.1228
}

PRIORS_ARE_MOCK = False
"""True while `BASE_CAPABILITY` holds placeholder values; flag any artifact built on them."""

#: Every model the router scores. All nine, including `claude-opus-4-6` and
#: `claude-sonnet-4-6`: spec §1.0 excluded those two because each Anthropic candidate cost
#: a query per step, but under the pivot no Anthropic model is queried at all, so they are
#: free to carry. See `NEAR_UNMEASURED` for the caveat that comes with them.
CANDIDATES: tuple[str, ...] = tuple(BASE_CAPABILITY)

#: The models actually queried. OpenAI credits are the only ones available, so these three
#: are the pipeline's "probes" — every other model is positioned by measuring its logged
#: actions against them (`logged_action.py`, `completion.py`).
PROBES: tuple[str, ...] = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

#: Scored, but on almost no evidence: these served 2 and 1 of the 1,000 exported
#: trajectories, giving 18 and 5 interior cut points — and cuts inside one trajectory are
#: near-duplicates (§1), so the effective sample is ~1 task each. Their scores must carry
#: this caveat and stay out of headline claims.
NEAR_UNMEASURED: frozenset[str] = frozenset({"claude-opus-4-6", "claude-sonnet-4-6"})


def is_probe(name: str) -> bool:
    """True if `name` is elicited directly rather than recovered from the log."""
    return name in PROBES


def prior_for(name: str) -> float:
    """`prior_i` for one candidate. Raises rather than defaulting: a silently-invented
    prior would propagate into every weight on every step."""
    try:
        return BASE_CAPABILITY[name]
    except KeyError:
        raise KeyError(
            f"no base_capability for {name!r} — add it to BASE_CAPABILITY, or drop the "
            f"model from CANDIDATES"
        ) from None


def weights(beta: float, models: tuple[str, ...] | list[str] | None = None) -> dict[str, float]:
    """`w_i = prior_i ** BETA` for each candidate.

    `BETA = 0` gives equal weights (pure conformity, prior ignored); `BETA = 1` makes
    agreement count in proportion to the prior; large `BETA` approaches "only the best
    model's opinion matters". Weights are deliberately *not* normalized — §5.1's
    denominator sums them, so any common factor cancels.
    """
    if beta < 0:
        raise ValueError(f"BETA must be >= 0, got {beta}")
    names = tuple(models) if models is not None else CANDIDATES
    return {name: prior_for(name) ** beta for name in names}
