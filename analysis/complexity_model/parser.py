"""Normalization — spec §3.

Parses one raw provider response into a canonical action. Arguments are used only to
decide `tool_call` vs `malformed` (schema validation), then discarded — no argument
normalization or comparison happens anywhere in the pipeline (§5.1 compares tool names,
not calls).
"""

from __future__ import annotations

from typing import Any

import jsonschema

from analysis.complexity_model.canonical import ToolDef, _parse_arguments
from analysis.complexity_model.scoring import Action, ActionType

#: Raw export item types that carry a tool invocation.
_TOOL_CALL_TYPES = ("function_call", "custom_tool_call")


def _validates(arguments: dict, tool: ToolDef) -> bool:
    try:
        jsonschema.validate(arguments, tool.parameters)
        return True
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        # A malformed *schema* (from convert_tool or the export) must not fail every call
        # against it — that would misclassify every model identically, which is a bug in
        # the pipeline, not evidence about any model's capability.
        return True


def classify_calls(
    calls: list[tuple[str, dict[str, Any]]], tools: dict[str, ToolDef]
) -> Action:
    """One or more `(name, arguments)` pairs from a single response -> `Action`.

    Any call naming a tool absent from the toolset, or failing schema validation against
    its definition, makes the **whole response** `malformed` (§3: "If a response mixes
    valid and invalid calls, classify the whole response malformed").
    """
    if not calls:
        return Action(type=ActionType.MALFORMED)
    for name, arguments in calls:
        tool = tools.get(name)
        if tool is None or not isinstance(arguments, dict) or not _validates(arguments, tool):
            return Action(type=ActionType.MALFORMED)
    return Action(type=ActionType.TOOL_CALL, tool_names=frozenset(name for name, _ in calls))


def classify_run(items: list[dict], tools: dict[str, ToolDef]) -> Action:
    """Classify one *logged* model turn — a maximal run of generated export items.

    This is the logged-action counterpart of `parse_openai` / `parse_anthropic`, and all
    three apply the **identical** rule, because their outputs are compared against each
    other. The rule is *did the model act, or did it stop*:

    1. any tool call  -> `tool_call` (validated; an invalid call is `malformed`)
    2. otherwise, any output at all -> `message` (the model ended its turn without acting)
    3. nothing at all -> `malformed`

    **This is the most dangerous place in the pipeline to get wrong**, because both the
    export's encodings represent the same behaviour differently and every naive reading is
    ~100% correlated with the provider — i.e. with `served_model`, which the README forbids
    any score from reading. Two traps, both measured on the corpus and both handled above:

    - Anthropic logs put preamble text in assistant `message` items sitting *next to* tool
      calls; OpenAI puts the equivalent in `reasoning` items. Reading "any assistant
      message means the turn was a message" gives Claude a 56-72% message-rate against
      8-14% for GPT — a spurious 5x gap that `SCORE_MESSAGE_WHEN_TOOL = 0.00` would turn
      into catastrophic cross-provider divergence. Rule 1 collapses it: 74-83% of those
      Claude message items are preamble accompanying a call.
    - A GPT turn that stops is logged as `{"content": "", "role": "assistant"}`, where a
      Claude turn that stops carries real text. Treating the empty one as `malformed`
      penalised GPT for an encoding quirk; treating it as "no action recorded" and
      dropping it silently deleted GPT's stop-turns while keeping Claude's. Both were
      wrong: all 65 such turns in the corpus are followed by a user message, so the model
      genuinely stopped and the user replied. Rule 2 classifies them as `message`, which
      is what they are.
    """
    calls: list[tuple[str, dict]] = []
    for item in items:
        kind = item.get("type") or "message"
        if kind in _TOOL_CALL_TYPES:
            # apply_patch (custom_tool_call) carries its payload under `input`.
            arguments = item.get("arguments")
            if arguments is None:
                arguments = {"patch": item.get("input") or ""}
            calls.append((str(item.get("name")), _parse_arguments(arguments)))

    if calls:
        return classify_calls(calls, tools)
    if items:
        return Action(type=ActionType.MESSAGE)
    return Action(type=ActionType.MALFORMED)


def parse_openai(response: dict, tools: dict[str, ToolDef]) -> Action:
    """Classify one OpenAI Responses-API response.

    `output` is a list of items; `function_call` items are the tool calls, `message`
    items with text content are the human-facing response. A response containing neither
    (e.g. empty output, or a refusal item this pipeline doesn't otherwise recognize) is
    `malformed` — an unparseable response is explicitly in that class (§3).
    """
    output = response.get("output")
    if not isinstance(output, list):
        return Action(type=ActionType.MALFORMED)

    calls: list[tuple[str, dict]] = []
    has_message = False
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            calls.append((str(item.get("name")), _parse_arguments(item.get("arguments"))))
        elif item_type == "message":
            # Presence, not non-emptiness: an empty message is how a stop is encoded, and
            # the logged path must classify it identically. See `classify_run`.
            has_message = True

    if calls:
        return classify_calls(calls, tools)
    if has_message:
        return Action(type=ActionType.MESSAGE)
    return Action(type=ActionType.MALFORMED)


def parse_anthropic(response: dict, tools: dict[str, ToolDef]) -> Action:
    """Classify one Anthropic Messages-API response.

    `content` is a list of blocks; `tool_use` blocks are the tool calls, `text` blocks
    with non-empty text are the human-facing response.
    """
    content = response.get("content")
    if not isinstance(content, list):
        return Action(type=ActionType.MALFORMED)

    calls: list[tuple[str, dict]] = []
    has_message = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            arguments = block.get("input")
            calls.append((str(block.get("name")), arguments if isinstance(arguments, dict) else {}))
        elif block_type == "text":
            has_message = True  # presence, not non-emptiness — see `classify_run`

    if calls:
        return classify_calls(calls, tools)
    if has_message:
        return Action(type=ActionType.MESSAGE)
    return Action(type=ActionType.MALFORMED)


#: Anthropic ids in the candidate set (spec §1.0) — decides which parser a stored raw
#: response needs, since the store keys on `(model, step_id)` and not on provider.
ANTHROPIC_MODELS = frozenset(
    {"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-8"}
)


def parse_response(model: str, response: dict, tools: dict[str, ToolDef]) -> Action:
    """Dispatch to the right parser by model id."""
    if model in ANTHROPIC_MODELS:
        return parse_anthropic(response, tools)
    return parse_openai(response, tools)
