"""Scoring — spec §5.

Three levels, bottom-up:

1. `pair(a, b)` — agreement between two responses, in [0, 1]. Stage A (type) then, when
   both sides called tools, Stage B (tool-set overlap).
2. `score(i, step)` — model `i`'s capability score on this step, from all its pairwise
   agreements plus its own prior weight.
3. `score_matrix` — the collection of `score(i, step)` over every model and step.

Comparison is coarse **by design** (§8): tool arguments are never compared, duplicate
calls to one tool collapse, and message content is never read. That buys a pipeline with
no LLM judge and no argument normalization, at the cost of a signal that resolves only at
the level of *which action type and which tools* — not *whether the action was right*.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    """Canonical class of one model response — see §3. `malformed` is an explicit class,
    never a dropped row: a model that cannot emit a valid call is failing the step."""

    MALFORMED = "malformed"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class Action:
    """One parsed response. `tool_names` is a set because a step may carry parallel calls.

    Arguments are deliberately absent: they are used during parsing to decide `tool_call`
    vs `malformed`, then discarded (§3).
    """

    type: ActionType
    tool_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.type is not ActionType.TOOL_CALL and self.tool_names:
            raise ValueError(f"{self.type.value} action must carry no tool names")
        if self.type is ActionType.TOOL_CALL and not self.tool_names:
            raise ValueError("tool_call action must name at least one tool")


# --- Stage A constants (§5.1). Tunable; rescoring from the response store is free. -----

SCORE_BOTH_MESSAGE = 1.00
SCORE_MESSAGE_WHEN_TOOL = 0.00  # this model stopped, the other kept working
SCORE_TOOL_WHEN_MESSAGE = 0.15  # this model kept working, the other stopped
SCORE_MALFORMED = 0.00


def pair(a: Action, b: Action) -> float:
    """Agreement of response `a` measured against response `b`, in [0, 1].

    **Not symmetric.** `a` is the model being scored, `b` the one it is compared against.
    A model that stops when others kept working is a silent task failure and scores 0; a
    model that keeps working when others stopped is recoverable — one redundant step — and
    gets partial credit. Compute both directions; never cache one and reuse it for the
    other.

    `malformed` scores 0 in every cell, in both directions. Two malformed responses are
    not two models concurring.
    """
    if a.type is ActionType.MALFORMED or b.type is ActionType.MALFORMED:
        return SCORE_MALFORMED
    if a.type is ActionType.MESSAGE:
        return SCORE_BOTH_MESSAGE if b.type is ActionType.MESSAGE else SCORE_MESSAGE_WHEN_TOOL
    if b.type is ActionType.MESSAGE:
        return SCORE_TOOL_WHEN_MESSAGE
    return jaccard(a.tool_names, b.tool_names)


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Stage B — plain unweighted overlap of the two tool-name *sets*.

    Symmetric, even though Stage A is not. Set semantics collapse duplicates: a model
    calling `file_read` three times on different paths contributes `{file_read}`,
    identical to a model calling it once. Direct consequence of dropping argument
    comparison. No per-tool weighting — down-weighting tools everyone calls is a
    refinement that can be applied later offline, since the raw responses are cached.
    """
    union = a | b
    if not union:  # unreachable via Action, which forbids an empty tool_call
        return 1.0
    return len(a & b) / len(union)


def score(model: str, actions: dict[str, Action], w: dict[str, float]) -> float:
    """`score(i, step) = ( w_i + Σ_{j≠i} w_j · pair(i, j) ) / Σ_all_j w_j`  — §5.1 level 2.

    Each model agrees with itself, weighted by its own prior. The denominator sums over
    **all** models including `i`, so the result lands in [0, 1] with no clamping.

    Three behaviours fall straight out of this one expression, with no branch or
    threshold — `tests/test_capability_scoring.py` pins all three:

    - **Unanimity** — every `pair` is 1, so every model scores exactly 1.0 whatever the
      weights. The prior cancels, the capability gap is zero, and `model.py` routes on
      cost alone. Correct: if every model does the same thing, buy the cheap one.
    - **Total disagreement** — every `pair` is 0, so `score(i) = w_i / Σw`. The ranking is
      the prior ranking: nobody knows what to do, so the strongest model has the best
      chance.
    - **Partial clustering** — a model inside a cluster of high-prior models scores well; a
      lone dissenter falls back toward `w_i / Σw`, which is the floor it can never drop
      below.

    Scores are deliberately **not normalized per step** (§8): at total disagreement the
    absolute values are small and the spread is narrow, so a cost-heavy weighting in
    `model.py` routes cheap even on contested steps. The cost/capability tradeoff belongs
    to the user's weights, not to this module.
    """
    total = sum(w[name] for name in actions)
    own = w[model]
    agreement = sum(
        w[other] * pair(actions[model], actions[other])
        for other in actions
        if other != model
    )
    return (own + agreement) / total


def score_step(actions: dict[str, Action], w: dict[str, float]) -> dict[str, float]:
    """`score(i, step)` for every model on one step."""
    return {name: score(name, actions, w) for name in actions}


def all_malformed(actions: dict[str, Action]) -> bool:
    """A step where every response is malformed must be dropped (§5.1 Stage A).

    Every score would come out at `w_i/Σw` — indistinguishable from total disagreement,
    but caused by a broken schema rather than a hard step. Log the count; nonzero means
    the tool schemas (§1.3) need revisiting.
    """
    return bool(actions) and all(a.type is ActionType.MALFORMED for a in actions.values())


@dataclass
class ScoreMatrix:
    """Output of §5: `step_id × model → score`, plus the diagnostics §5.2/§8 ask for."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    pairs: dict[str, dict[tuple[str, str], float]] = field(default_factory=dict)
    malformed_counts: Counter[str] = field(default_factory=Counter)
    steps_scored: int = 0
    steps_dropped_incomplete: int = 0
    steps_dropped_all_malformed: int = 0

    def malformed_rate(self) -> dict[str, float]:
        """Per-model malformed rate — §5.2.

        Tracked standalone **in addition to** counting malformed as disagreement: it is a
        direct capability fact and should not be laundered through a divergence score. It
        is also the most legible single number in the pipeline.
        """
        if not self.steps_scored:
            return {}
        return {m: c / self.steps_scored for m, c in self.malformed_counts.items()}


def build_matrix(
    steps: dict[str, dict[str, Action]],
    w: dict[str, float],
    candidates: tuple[str, ...] | list[str],
) -> ScoreMatrix:
    """Score every complete step. `steps` maps `step_id -> {model: Action}`.

    Two filters, both of which must be logged rather than silently applied:

    - **Completeness** (§2.4) — a step is used only if every candidate has a successful
      record. A partial row distorts conformity scoring, because an absent response is not
      a neutral one.
    - **All-malformed** (§5.1) — dropped, see `all_malformed`.
    """
    matrix = ScoreMatrix()
    wanted = set(candidates)
    for step_id, actions in steps.items():
        if set(actions) != wanted:
            matrix.steps_dropped_incomplete += 1
            continue
        if all_malformed(actions):
            matrix.steps_dropped_all_malformed += 1
            continue
        matrix.scores[step_id] = score_step(actions, w)
        matrix.pairs[step_id] = {
            (a, b): pair(actions[a], actions[b])
            for a in actions
            for b in actions
            if a != b
        }
        for name, action in actions.items():
            if action.type is ActionType.MALFORMED:
                matrix.malformed_counts[name] += 1
        matrix.steps_scored += 1
    return matrix
