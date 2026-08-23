"""Sampling — spec §1.

Draws trajectories at random from the corpus, cuts each at a random interior call
boundary, and emits one sample record per trajectory. One sample per trajectory: steps
within a trajectory are near-duplicates (same task, same files, same tools), so sampling
across trajectories buys far more effective diversity than sampling within one (§1).

This module works on the **raw** export line, not `cheapy.models.trajectory` — the prefix
has to be re-rendered onto the wire (`canonical.py`), and `Trajectory` only keeps the
normalized, encoding-independent view that a wire format can't be reconstructed from.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from research.capability_fitting.canonical import is_generated_item, to_canonical
from research.capability_fitting.logged_action import action_run_at
from cheapy.preprocessing.trajectory_analyzer import count_tokens

#: Spec §1.1. Measured on this export the filter is a no-op — the largest prefix seen is
#: ~123K tokens — but it is cheap and the spec requires it, so it stays on and logs.
MAX_PREFIX_TOKENS = 1_000_000


@dataclass(frozen=True)
class Sample:
    """One sampled cut point — spec §1, the record shape at the top of that section."""

    step_id: str
    trajectory_id: int
    prefix_items: list[dict]  # raw export items, verbatim, up to the cut
    tools: list[dict]  # raw export tool defs, verbatim
    prefix_token_count: int
    step_index: int
    served_model: str
    """Which model produced `logged_action_items`. Under the pivot this is a **target
    label**, not a feature: `features.extract_step_features` must never read it (docs/FULL_REPORT.md §4
    leakage), but the pair it anchors is a genuine measurement."""

    logged_action_items: list[dict] = field(default_factory=list)
    """The served model's real next turn — the run of generated items starting at the cut.

    This is what makes the OpenAI-only pivot work: the six Anthropic candidates can never
    be queried, but their actions at this exact cut are already in the log. Carried on the
    sample so `store/samples.jsonl` is self-contained and scoring never re-reads the
    export. See `logged_action.py`.
    """

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _call_starts(items: list[dict]) -> list[int]:
    """Index of the first item of each model call — maximal runs of `is_generated_item`.

    Mirrors `trajectory_analyzer._assign_call_indices`'s segmentation exactly (see
    `is_generated_item`'s docstring on why the two must agree), repeated here because it
    operates on raw items rather than `NormalizedItem`.
    """
    starts: list[int] = []
    in_run = False
    for index, item in enumerate(items):
        if is_generated_item(item):
            if not in_run:
                starts.append(index)
                in_run = True
        else:
            in_run = False
    return starts


def _step_id(trajectory_id: int, cut_index: int) -> str:
    """Stable cache key — used to dedupe the elicitation store, so it must be a pure
    function of the cut, not of anything about the run that produced it."""
    digest = hashlib.sha1(f"{trajectory_id}:{cut_index}".encode()).hexdigest()
    return digest[:16]


def sample_cut(
    trajectory_id: int, record: dict, rng: random.Random
) -> Sample | None:
    """One random-interior-cut sample from one export line, or `None` if it has no
    interior boundary to cut at (fewer than 2 calls) or the cut prefix exceeds §1.1.

    "Interior" excludes both the very first call (an empty prefix teaches nothing about
    divergence) and the very last (that is the end of the trajectory, not mid-flight).
    """
    items: list[dict] = record.get("input") or []
    starts = _call_starts(items)
    if len(starts) < 3:
        return None  # no interior boundary strictly between the first and last call

    cut_index = rng.randrange(1, len(starts) - 1)
    cut = starts[cut_index]
    prefix_items = items[:cut]
    tools: list[dict] = record.get("tools") or []

    canonical = to_canonical(prefix_items, tools)
    token_count = count_tokens(canonical.system) + sum(
        count_tokens(json.dumps(asdict(t))) for t in canonical.tools
    )
    for item in canonical.items:
        text = getattr(item, "text", None) or getattr(item, "output", None) or ""
        token_count += count_tokens(text)
        if hasattr(item, "arguments"):
            token_count += count_tokens(json.dumps(item.arguments))

    if token_count > MAX_PREFIX_TOKENS:
        return None

    return Sample(
        logged_action_items=action_run_at(items, cut),
        step_id=_step_id(trajectory_id, cut_index),
        trajectory_id=trajectory_id,
        prefix_items=prefix_items,
        tools=tools,
        prefix_token_count=token_count,
        step_index=cut_index,
        served_model=str(record.get("model")),
    )


def draw_samples(
    jsonl_path: str | Path, n: int, seed: int = 0
) -> Iterator[Sample]:
    """Draw up to `n` samples at random from the corpus, one per trajectory.

    Trajectories with no interior boundary or whose prefix exceeds §1.1 are skipped and
    logged, then the draw continues to the next candidate trajectory — the caller gets `n`
    usable samples (or as many as the corpus can supply), not `n` attempts.
    """
    with open(jsonl_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)

    emitted = 0
    excluded_short = 0
    excluded_oversized = 0
    for trajectory_id in order:
        if emitted >= n:
            break
        record = json.loads(lines[trajectory_id])
        sample = sample_cut(trajectory_id, record, rng)
        if sample is None:
            starts = _call_starts(record.get("input") or [])
            if len(starts) < 3:
                excluded_short += 1
            else:
                excluded_oversized += 1
            continue
        emitted += 1
        yield sample

    print(
        f"[sampler] emitted {emitted}/{n} requested "
        f"(skipped {excluded_short} too-short, {excluded_oversized} over MAX_PREFIX_TOKENS, "
        f"out of {len(lines)} trajectories scanned)"
    )


def write_samples(jsonl_path: str | Path, n: int, out_path: str | Path, seed: int = 0) -> Path:
    """Draw and persist samples as JSONL — the input to `elicit.py`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for sample in draw_samples(jsonl_path, n, seed=seed):
            handle.write(json.dumps(sample.to_json()) + "\n")
    return out_path
