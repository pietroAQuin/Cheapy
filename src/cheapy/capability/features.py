"""Step features — spec §6.2. The one function called by both `train.py` (offline) and
`capability_model.py` (inference), so the two paths cannot drift apart.

Operates on `cheapy.models.trajectory` — not on raw export dicts. This is deliberate, not
just convenient: `Trajectory` already *is* "everything known about the prefix as it
exists right now" (docs/FULL_REPORT.md §2), which is exactly the object the router hands
`performance_model.py` at inference. Building a prefix-Trajectory offline (via
`cheapy.preprocessing.trajectory_analyzer.analyze()` on the truncated raw items — see
`train.py`) and reading the router's own live `Trajectory` at inference are therefore the
same operation on the same type, satisfying §6.2's "same extraction code offline and at
inference" literally rather than by convention. It also reuses `analyze()`'s tested,
export-cross-checked segmentation and token counting instead of re-deriving them.

Two hard constraints, both about *when* a feature is computable, not which features to
pick:

- **No leakage.** Every feature must be derivable from the prefix as it exists at the cut
  point. Nothing from the remainder of the trajectory, the recorded action, or the
  responses being predicted — at inference the router has only the prefix, and a
  `Trajectory` built from a prefix has no way to see past it in the first place.
- **No `served_model` proxy.** docs/FULL_REPORT.md §4's leakage table names five prefix properties that
  separate the model families at ~100%: coding dialect (`bash`/`file_*` vs
  `shell_command`/`apply_patch`), item encoding, `custom_tool_call` presence, `reasoning`
  presence, and an empty final assistant message. None of those are read here. Tool
  identity is folded into aggregate counts (distinct-tool ratio, repeat-tool runs) rather
  than exposed as which named tools appear — the thing that would let the regressor key
  off the toolset dialect instead of genuine step difficulty. `Trajectory.served_model`
  itself is never touched.
"""

from __future__ import annotations

import math

from cheapy.models.trajectory import ItemKind, Trajectory, ViktorEnvironment

#: Keyword scan for §1.3's "cheap extra signal": tool-result content that retroactively
#: marks a prior call as having failed. Coarse on purpose — this is a count feature, not a
#: judge; a false positive on the word "error" costs nothing since ridge just sees one more
#: noisy count among many.
_ERROR_MARKERS = ("error", "Error", "ERROR", "failed", "Failed", "traceback", "Traceback", "exception")

#: Fixed column order `extract_step_features` returns values in. `train.py` and
#: `capability_model.py` both import this rather than hardcoding the order, so a feature
#: added here appears in both call sites from a single edit.
FEATURE_NAMES: tuple[str, ...] = (
    "log_prefix_tokens",
    "item_count",
    "call_count",
    "tool_calls_per_call",
    "distinct_tool_ratio",
    "max_repeat_run",
    "recent_error_rate",
    "last_output_len_tokens",
    "last_is_user_message",
    "assistant_share",
    "tool_output_share",
    "images_so_far",
    "toolset_size",
    "is_teams",
    "is_subagent",
)


def extract_step_features(trajectory: Trajectory) -> dict[str, float]:
    """`Trajectory` (built from a prefix, whether a training-time cut or the router's own
    live state) -> the feature vector, as a `{name: value}` dict over `FEATURE_NAMES`.
    Pure function of its argument — no I/O, no model calls.
    """
    items = trajectory.normalized_items

    tool_calls = [item for item in items if item.kind is ItemKind.TOOL_CALL]
    tool_names = [item.tool_name for item in tool_calls if item.tool_name]
    distinct_ratio = (len(set(tool_names)) / len(tool_names)) if tool_names else 1.0
    max_repeat = _max_repeat_run(tool_names)

    tool_outputs = [item for item in items if item.kind is ItemKind.TOOL_OUTPUT]
    recent = tool_outputs[-5:]
    recent_error_rate = (
        sum(1 for item in recent if _looks_like_error(item.tool_output or "")) / len(recent)
        if recent
        else 0.0
    )
    last_output_len = tool_outputs[-1].est_tokens if tool_outputs else 0

    last_is_user = bool(items) and items[-1].kind is ItemKind.USER_MESSAGE

    item_token_total = sum(item.est_tokens for item in items) or 1
    assistant_tokens = sum(
        item.est_tokens for item in items if item.kind is ItemKind.ASSISTANT_MESSAGE
    )
    tool_output_tokens = sum(item.est_tokens for item in tool_outputs)

    return {
        "log_prefix_tokens": math.log1p(trajectory.total_tokens),
        "item_count": float(len(items)),
        "call_count": float(trajectory.total_calls),
        "tool_calls_per_call": trajectory.avg_tools_per_call,
        "distinct_tool_ratio": distinct_ratio,
        "max_repeat_run": float(max_repeat),
        "recent_error_rate": recent_error_rate,
        "last_output_len_tokens": float(last_output_len),
        "last_is_user_message": float(last_is_user),
        "assistant_share": assistant_tokens / item_token_total,
        "tool_output_share": tool_output_tokens / item_token_total,
        "images_so_far": float(trajectory.total_images_received),
        "toolset_size": float(trajectory.toolset_size),
        "is_teams": float(trajectory.viktor_environment is ViktorEnvironment.TEAMS),
        "is_subagent": float(trajectory.is_subagent),
    }


def _max_repeat_run(names: list[str]) -> int:
    """Longest run of the same tool called back-to-back — a retry/thrash signal."""
    if not names:
        return 0
    best = run = 1
    for prev, curr in zip(names, names[1:]):
        run = run + 1 if curr == prev else 1
        best = max(best, run)
    return best


def _looks_like_error(output: str) -> bool:
    return any(marker in output for marker in _ERROR_MARKERS)


def pair_descriptor(
    model_a: str, model_b: str, prior: dict[str, float], family_of: dict[str, str], provider_of: dict[str, str]
) -> dict[str, float]:
    """Pair-descriptor features — spec §6.1, **not optional**: without them the pooled
    rows are indistinguishable across pairs and the regressor fits one average divergence
    curve for all pairs instead of learning how prior-distance predicts divergence.
    """
    pa, pb = prior[model_a], prior[model_b]
    return {
        "prior_a": pa,
        "prior_b": pb,
        "prior_abs_diff": abs(pa - pb),
        "prior_min": min(pa, pb),
        "prior_max": max(pa, pb),
        "same_provider": float(provider_of[model_a] == provider_of[model_b]),
        "same_family": float(family_of[model_a] == family_of[model_b]),
    }


PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "prior_a",
    "prior_b",
    "prior_abs_diff",
    "prior_min",
    "prior_max",
    "same_provider",
    "same_family",
)
