"""Recover the served model's real next action from the corpus.

The pivot rests on this module. OpenAI credits are the only ones available, so the six
Anthropic candidates can never be queried — but every export line carries `model` plus the
full action history, so at any interior cut **the served model's actual next action is
already in the log**: it is precisely the run of generated items the sampler cuts away.

That turns 18 of the 36 model pairs from assumption into measurement, at zero API cost,
and it is what stops the capability score from degenerating into a benchmark lookup.

**This is a target, never a feature.** `features.extract_step_features` must never see it:
at inference the router has only the prefix, so a feature derived from the recorded action
is unavailable in production no matter how well it scores offline (spec §6.2). Using it as
the thing being predicted is a different matter and carries no leakage.

Two biases ride along with it, both stated rather than hidden (spec §8, and `calibration.py`
which measures the first):

- **Home-field advantage.** The logged action was produced by the model that wrote the
  entire prefix, in its own conventions; the elicited probes continue a stranger's history.
- **Temperature.** The logged action came from the Viktor harness at its own sampling
  settings; probes run at whatever `clients.probe_temperature` established.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.capability_fitting.canonical import ToolDef, is_generated_item
from research.capability_fitting.parser import classify_run
from research.capability_fitting.scoring import Action


@dataclass(frozen=True)
class LoggedAction:
    """The served model's next turn at a cut point, plus the raw items behind it."""

    model: str
    action: Action
    items: list[dict]


def action_run_at(items: list[dict], cut: int) -> list[dict]:
    """The maximal run of generated items starting at index `cut`.

    `cut` is a call boundary as produced by `sampler._call_starts`, so `items[cut]` is the
    first item the model generated for that call. The run ends at the first item the model
    did not generate (a tool result, or a user message).
    """
    if cut >= len(items) or not is_generated_item(items[cut]):
        return []
    end = cut
    while end < len(items) and is_generated_item(items[end]):
        end += 1
    return items[cut:end]


def logged_action_for(
    served_model: str, items: list[dict], cut: int, tools: dict[str, ToolDef]
) -> LoggedAction:
    """Classify the served model's turn at `cut` with the same rule used for elicited
    responses — see `parser.classify_run` for why that shared rule is non-negotiable."""
    run = action_run_at(items, cut)
    return LoggedAction(model=served_model, action=classify_run(run, tools), items=run)
