"""Canonical wire format — one provider-neutral request, two renderers.

Spec §2 asks for the logged prompt and tool schemas "verbatim", because any deviation is
measured as divergence and that is the largest confound in the pipeline. **Verbatim is
not achievable for this export**, for three reasons found in the data:

1. `reasoning` items (1,792, gpt-family only) carry summary text but no encrypted
   content, so no model can be asked to continue from them — not even the one that
   produced them.
2. `apply_patch` is an OpenAI `custom` freeform-grammar tool (245 lines). Anthropic has no
   equivalent tool type.
3. Every message uses `input_text` content parts, **including the 5,530 assistant turns**,
   which is not a valid Responses *output* shape and cannot be replayed as one.
4. `input_image` parts are redacted to `data:image/jpeg;base64,[base64 image redacted]` —
   not decodable base64, so any provider rejects them.

The response is to normalize **once, uniformly**, and render the same canonical object for
every provider. Every loss below is therefore identical across all seven candidates and
adds no per-model confound — which is the property §2 actually needs. What it costs is
stated plainly in the writeup rather than hidden:

- reasoning items are dropped;
- message content is flattened to plain text and images become a text marker;
- `apply_patch` becomes an ordinary function taking one string argument, with its grammar
  moved into the description so the model still knows the patch syntax;
- empty assistant messages are dropped (308 in the export) — several providers reject an
  empty text block outright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

#: Stands in for a redacted `input_image` part. The export replaced every image with a
#: placeholder data URL that is not decodable, so the image itself is unrecoverable; what
#: is preserved is the *fact* that an image was present at that point in the history.
IMAGE_PLACEHOLDER = "[image redacted]"

#: Item *types* the model generates whenever type alone decides it. `message` is
#: deliberately absent — an assistant `message` is also generated, but a *user* or
#: *system* `message` is not, so type alone can't decide; `is_generated_item` below
#: is the real predicate and reads the role too. Kept for callers that only need the
#: type-only subset.
GENERATED_TYPES = frozenset({"reasoning", "function_call", "custom_tool_call"})


def is_generated_item(item: dict) -> bool:
    """True if `item` was produced by the model, not the user/system/tool runtime.

    Mirrors `cheapy.models.trajectory.GENERATED_KINDS` (`ASSISTANT_MESSAGE`, `REASONING`,
    `TOOL_CALL`) exactly, on raw export items instead of `NormalizedItem` — sampling and
    feature extraction work on raw items that still have to go back on the wire, so this
    is repeated here rather than reused directly. A maximal consecutive run of items this
    returns `True` for is one call. **The two must stay in lock-step**: `sampler.py`'s cut
    point is meaningless unless it lands on the same call boundary `analyze()` will later
    recompute for the very same prefix (§6.2's shared-extraction-code requirement starts
    at this segmentation, one level below `extract_step_features`).
    """
    kind = item.get("type") or "message"
    if kind in GENERATED_TYPES:
        return True
    return kind == "message" and item.get("role") == "assistant"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    text: str


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    output: str


Item = Message | ToolCall | ToolResult


@dataclass(frozen=True)
class ToolDef:
    """One tool, in JSON-Schema terms both providers accept."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class CanonicalRequest:
    """A prefix ready to be rendered for any provider."""

    system: str
    items: tuple[Item, ...]
    tools: tuple[ToolDef, ...]


# --- export -> canonical -----------------------------------------------------------


def _text_of(item: dict) -> str:
    """Flatten either encoding's message content to plain text.

    `claude-*` lines use `content: [{"type": "input_text", "text": ...}]`; `gpt-*` lines
    use `content: "..."` with no `type` key at all. Both land here. An `input_image` part
    becomes `IMAGE_PLACEHOLDER` in position, so the history still records that an image
    arrived even though the pixels are gone.
    """
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "input_image":
            parts.append(IMAGE_PLACEHOLDER)
        else:
            parts.append(part.get("text") or "")
    return "".join(parts)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments as an object. All 10,123 calls in the export parse cleanly to
    one; the fallback keeps a malformed string visible rather than dropping the call."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"_unparsed_arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"_unparsed_arguments": str(raw)}


def convert_tool(tool: dict) -> ToolDef:
    """One exported tool definition -> `ToolDef`.

    `function` tools pass through. The one `custom` tool — `apply_patch` — is rewritten as
    a function taking a single `patch` string, with its lark grammar appended to the
    description. Anthropic has no freeform-grammar tool type, so the alternative was
    dropping 245 trajectories (every one of them gpt-served, i.e. a family-correlated
    cut). Applied identically for all seven candidates, so it costs comparability nothing.
    """
    name = str(tool.get("name"))
    description = tool.get("description") or ""
    if tool.get("type") == "custom":
        grammar = (tool.get("format") or {}).get("definition") or ""
        return ToolDef(
            name=name,
            description=(
                f"{description}\n\nSupply the complete patch text as the `patch` argument. "
                f"It must follow this grammar:\n\n{grammar}"
            ).strip(),
            parameters={
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "The full patch text."}
                },
                "required": ["patch"],
            },
        )
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return ToolDef(name=name, description=description, parameters=parameters)


def to_canonical(raw_items: list[dict], raw_tools: list[dict]) -> CanonicalRequest:
    """Build one `CanonicalRequest` from a prefix of an export line's `input` and `tools`.

    Leading system messages become the top-level `system` string (Anthropic requires it
    there, and OpenAI accepts an equivalent `instructions`), so the two renderings carry
    the same prompt in the place each provider expects.
    """
    system_parts: list[str] = []
    items: list[Item] = []

    for raw in raw_items:
        kind = raw.get("type") or "message"

        if kind == "reasoning":
            continue  # summary-only, no encrypted content — unreplayable for any model

        if kind == "message":
            text = _text_of(raw)
            role_name = raw.get("role") or "user"
            if role_name == "system":
                if text.strip():
                    system_parts.append(text)
                continue
            if not text.strip():
                continue  # 308 empty assistant turns; providers reject empty text blocks
            items.append(Message(role=Role(role_name), text=text))
            continue

        if kind in ("function_call", "custom_tool_call"):
            # apply_patch carries its payload under `input`; function_call uses `arguments`.
            arguments = raw.get("arguments")
            if arguments is None:
                arguments = {"patch": raw.get("input") or ""}
            items.append(
                ToolCall(
                    call_id=str(raw.get("call_id")),
                    name=str(raw.get("name")),
                    arguments=_parse_arguments(arguments),
                )
            )
            continue

        if kind in ("function_call_output", "custom_tool_call_output"):
            items.append(
                ToolResult(
                    call_id=str(raw.get("call_id")), output=str(raw.get("output") or "")
                )
            )
            continue

    return CanonicalRequest(
        system="\n\n".join(system_parts),
        items=tuple(items),
        tools=tuple(convert_tool(tool) for tool in raw_tools),
    )


# --- canonical -> provider ---------------------------------------------------------


def render_openai(request: CanonicalRequest) -> dict[str, Any]:
    """Payload for the OpenAI Responses API, minus `model`.

    Close to the export's own shape, since the export *is* Responses format — the changes
    are the ones `to_canonical` already made for everyone.
    """
    payload_input: list[dict[str, Any]] = []
    for item in request.items:
        if isinstance(item, Message):
            payload_input.append({"role": item.role.value, "content": item.text})
        elif isinstance(item, ToolCall):
            payload_input.append(
                {
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": json.dumps(item.arguments),
                }
            )
        else:
            payload_input.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                }
            )
    return {
        "instructions": request.system,
        "input": payload_input,
        "tools": [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in request.tools
        ],
    }


def render_anthropic(request: CanonicalRequest) -> dict[str, Any]:
    """Payload for the Anthropic Messages API, minus `model` and `max_tokens`.

    Three structural rules the Messages API enforces and the Responses format does not:
    tool results are `user`-turn blocks rather than items of their own; consecutive
    same-role turns must be merged; and the first turn must be `user`. All three are
    handled here so the *content* still matches `render_openai` exactly.
    """
    messages: list[dict[str, Any]] = []

    def push(role: str, block: dict[str, Any]) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})

    for item in request.items:
        if isinstance(item, Message):
            push(item.role.value, {"type": "text", "text": item.text})
        elif isinstance(item, ToolCall):
            push(
                "assistant",
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": item.arguments,
                },
            )
        else:
            push(
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": item.call_id,
                    "content": item.output,
                },
            )

    # An assistant turn cannot open the conversation. Only reachable if a prefix begins
    # mid-call, which the sampler's cut points rule out — kept so the renderer is total.
    if messages and messages[0]["role"] == "assistant":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continue)"}]})

    return {
        "system": request.system,
        "messages": messages,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in request.tools
        ],
    }
