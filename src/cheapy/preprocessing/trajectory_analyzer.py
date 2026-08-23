"""Parse one line of the export into one `Trajectory`.

Each JSONL line is a complete, self-contained trajectory: no line's input continues
another's, so there is no grouping or reconstruction step — iterate the file and call
`analyze` per line.

Within a line, the call structure is recovered by segmenting `input` into maximal runs
of model-generated items. Every run is one call's output, including the trailing
assistant message that closes the trajectory.

Two encodings appear in the export and both must be handled: `claude-*` lines use
`{"type": "message", "content": [{"type": "input_text", "text": ...}]}` while `gpt-*`
lines use `{"role": ..., "content": "..."}` with no `type` key at all. A parser that
only understands the first returns zero tokens for every GPT line — a silent
under-count perfectly correlated with model family.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from cheapy.models.trajectory import (
    CHARS_PER_TOKEN,
    GENERATED_KINDS,
    ItemKind,
    NormalizedItem,
    Tool,
    Trajectory,
    ViktorEnvironment,
)

#: Fraction of the shared prefix assumed to be served from the provider's cache.
#: 1.0 is the optimistic bound: every call after the first re-sends its predecessor's
#: entire prompt, and all of it hits. Real caches have TTLs and minimum block sizes,
#: and the export has no timestamps to model expiry with, so this is the one assumption
#: standing in for all of that.
#:
#: This is the *fallback* only. `analyze()` takes it as an argument, and the CLI passes
#: whatever `CACHE_HIT_RATE` in `cheapy.yaml` (or `--cache-hit-rate`) resolves to — so
#: sweep it there rather than hardcoding a discount anywhere else.
CACHE_HIT_RATE = 1.0

#: Uniform tokenizer for every line, both model families. The anonymized ids map to no
#: public tokenizer (`encoding_for_model("claude-opus-5")` raises), and using a
#: different tokenizer per family would make prices incomparable across the very
#: comparison the router exists to make. Measured against chars/4 on this export:
#: chars/4 understates Claude text by 4% and GPT text by 9%, because tool-call JSON
#: tokenizes at ~3.2 chars/token against ~4.15 for prose. Still an ESTIMATE, but a
#: family-neutral one.
ENCODING_NAME = "o200k_base"

#: Repo-local vocab cache so the pipeline runs offline after a one-time 3.6 MB fetch.
TOKENIZER_CACHE_DIR = Path(__file__).resolve().parent.parent / ".tokenizer_cache"

_encoder: Any = None
_encoder_ready = False


def _get_encoder() -> Any:
    """Load the tokenizer once. Returns None if unavailable, so callers fall back."""
    global _encoder, _encoder_ready
    if _encoder_ready:
        return _encoder
    _encoder_ready = True
    try:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TOKENIZER_CACHE_DIR))
        TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import tiktoken

        _encoder = tiktoken.get_encoding(ENCODING_NAME)
    except Exception:
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """ESTIMATE the tokens in `text`.

    Uses the real tokenizer when it is available and falls back to the chars/4
    convention when it is not, so a laptop with no network still runs the pipeline —
    at the cost of the family-correlated bias documented on ENCODING_NAME.
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is None:
        return len(text) // CHARS_PER_TOKEN
    return len(encoder.encode(text, disallowed_special=()))


#: Raw item type -> common kind. The GPT-encoded lines omit `type` on messages, so a
#: missing type means "message" and the role decides.
_ROLE_KINDS = {
    "system": ItemKind.SYSTEM_MESSAGE,
    "user": ItemKind.USER_MESSAGE,
    "assistant": ItemKind.ASSISTANT_MESSAGE,
}
_TYPE_KINDS = {
    "reasoning": ItemKind.REASONING,
    "function_call": ItemKind.TOOL_CALL,
    "custom_tool_call": ItemKind.TOOL_CALL,
    "function_call_output": ItemKind.TOOL_OUTPUT,
    "custom_tool_call_output": ItemKind.TOOL_OUTPUT,
}


def _text_of(item: dict) -> str:
    """Message text from either encoding, or a reasoning item's summary text."""
    parts: list[str] = []

    content = item.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_text":
                parts.append(part.get("text") or "")

    for part in item.get("summary") or []:
        if isinstance(part, dict):
            parts.append(part.get("text") or "")

    return "".join(parts)


def _count_images(item: dict) -> int:
    """`input_image` parts on one item. Only user messages ever carry them.

    Redacted to ~45-char placeholder URLs, so the count is kept but the real token
    cost of the original images is unrecoverable.
    """
    content = item.get("content")
    if not isinstance(content, list):
        return 0
    return sum(
        1 for part in content
        if isinstance(part, dict) and part.get("type") == "input_image"
    )


def normalize(items: list[dict]) -> list[NormalizedItem]:
    """Convert one line's raw `input` into the encoding-independent representation.

    This is the only place in the pipeline that has to know the export ships two
    encodings; everything downstream reads `NormalizedItem`.
    """
    normalized: list[NormalizedItem] = []

    for item in items:
        raw_type = item.get("type") or "message"
        if raw_type == "message":
            kind = _ROLE_KINDS.get(item.get("role") or "", ItemKind.USER_MESSAGE)
        else:
            kind = _TYPE_KINDS.get(raw_type, ItemKind.TOOL_OUTPUT)

        is_custom = raw_type.startswith("custom_tool_call")
        # apply_patch carries its patch under `input`; function_call uses `arguments`.
        arguments = item.get("arguments")
        if arguments is None and is_custom:
            arguments = item.get("input")

        output = item.get("output")
        text = _text_of(item)
        name = item.get("name")

        est_tokens = (
            count_tokens(text)
            + count_tokens(arguments if isinstance(arguments, str) else "")
            + count_tokens(output if isinstance(output, str) else "")
            + count_tokens(name if isinstance(name, str) else "")
        )

        normalized.append(
            NormalizedItem(
                kind=kind,
                text=text,
                tool_name=name if isinstance(name, str) else None,
                tool_arguments=arguments if isinstance(arguments, str) else None,
                tool_output=output if isinstance(output, str) else None,
                tool_call_id=item.get("call_id"),
                is_custom_tool=is_custom,
                images=_count_images(item),
                est_tokens=est_tokens,
                item_id=item.get("id"),
            )
        )

    _assign_call_indices(normalized)
    return normalized


def _assign_call_indices(items: list[NormalizedItem]) -> None:
    """Stamp each generated item with the index of the call that produced it.

    A call is a maximal consecutive run of generated kinds, closed by any other kind.
    Runs are 1-2 items in 96% of cases; longer ones are parallel tool calls. The
    trailing run is the trajectory's final call, so every call's output is present.
    """
    call_index = -1
    in_run = False
    for item in items:
        if item.kind in GENERATED_KINDS:
            if not in_run:
                call_index += 1
                in_run = True
            item.call_index = call_index
        else:
            in_run = False


def analyze(
    line: str | dict, id: int, *, cache_hit_rate: float = CACHE_HIT_RATE
) -> Trajectory:
    """Build one `Trajectory` from one line of the export.

    `id` is supplied by the caller because a line carries no position of its own.
    `cache_hit_rate` is the assumed prefix-cache hit fraction; it feeds
    `total_cached_tokens` and therefore every downstream cost estimate.
    """
    if not 0.0 <= cache_hit_rate <= 1.0:
        raise ValueError(f"cache_hit_rate must be in [0, 1], got {cache_hit_rate}")

    record = json.loads(line) if isinstance(line, str) else line
    tool_defs: list[dict] = record.get("tools") or []
    items = normalize(record.get("input") or [])

    tool_names = [str(tool.get("name")) for tool in tool_defs]
    toolset = [Tool(name) for name in tool_names]

    environment = (
        ViktorEnvironment.TEAMS
        if any("msteams" in name for name in tool_names)
        else ViktorEnvironment.SLACK
    )

    # The tool schemas are re-sent on every call — a median 4,203 est. tokens, and the
    # most cacheable segment of the prefix since they sit at its very front. Dropping
    # them would understate every prompt by a large constant.
    tools_tokens = sum(count_tokens(json.dumps(tool)) for tool in tool_defs)

    # Prefix sums let each call's prompt be read off in O(1).
    prefix_tokens = [0] * (len(items) + 1)
    for index, item in enumerate(items):
        prefix_tokens[index + 1] = prefix_tokens[index] + item.est_tokens

    # Each call's prompt is everything preceding the run that produced it, plus the
    # schemas; its output is that run.
    call_starts: list[int] = []
    output_tokens: dict[int, int] = {}
    for index, item in enumerate(items):
        if item.call_index is None:
            continue
        if item.call_index == len(call_starts):
            call_starts.append(index)
        output_tokens[item.call_index] = (
            output_tokens.get(item.call_index, 0) + item.est_tokens
        )

    total_calls = len(call_starts)
    prompt_tokens = [tools_tokens + prefix_tokens[start] for start in call_starts]
    total_tokens = sum(prompt_tokens)

    # Each call's prompt extends its predecessor's, so under a perfect prefix cache the
    # cached share of call i is exactly call i-1's whole prompt. Summed, that collapses
    # to an identity rather than a measurement — worth naming plainly instead of
    # dressing a subtraction up as an overlap search. CACHE_HIT_RATE is the assumption.
    total_cached_tokens = (
        int((total_tokens - prompt_tokens[-1]) * cache_hit_rate) if prompt_tokens else 0
    )

    # custom_tool_call counts as a tool call: it is how apply_patch is invoked, and
    # excluding it would undercount tool activity on exactly the GPT-encoded lines.
    total_tool_calls = sum(1 for item in items if item.kind is ItemKind.TOOL_CALL)
    total_ai_messages = sum(
        1 for item in items if item.kind is ItemKind.ASSISTANT_MESSAGE
    )
    total_user_messages = sum(1 for item in items if item.kind is ItemKind.USER_MESSAGE)
    total_images_received = sum(item.images for item in items)
    total_draft_submit_calls = sum(
        1 for item in items
        if item.kind is ItemKind.TOOL_CALL and item.tool_name == Tool.SUBMIT_DRAFT.value
    )

    def per_call(total: float) -> float:
        return total / total_calls if total_calls else 0.0

    return Trajectory(
        id=id,
        served_model=record["model"],
        normalized_items=items,
        is_subagent=Tool.SUBMIT_SUBAGENT_RESULT in toolset,
        viktor_environment=environment,
        toolset=toolset,
        toolset_size=len(tool_defs),
        avg_tools_per_call=per_call(total_tool_calls),
        avg_images_received_per_call=per_call(total_images_received),
        avg_output_tokens_per_call=per_call(sum(output_tokens.values())),
        avg_input_tokens_per_call=per_call(total_tokens),
        total_calls=total_calls,
        total_tokens=total_tokens,
        total_cached_tokens=total_cached_tokens,
        last_call_input_tokens=prompt_tokens[-1] if prompt_tokens else 0,
        total_tool_calls=total_tool_calls,
        total_images_received=total_images_received,
        total_ai_messages=total_ai_messages,
        total_user_messages=total_user_messages,
        total_draft_submit_calls=total_draft_submit_calls,
    )


def analyze_file(
    path: str | Path, *, cache_hit_rate: float = CACHE_HIT_RATE
) -> Iterator[Trajectory]:
    """Yield one `Trajectory` per line. Lines are independent — no grouping needed."""
    with open(path, encoding="utf-8") as handle:
        for id, line in enumerate(handle):
            line = line.strip()
            if line:
                yield analyze(line, id, cache_hit_rate=cache_hit_rate)
