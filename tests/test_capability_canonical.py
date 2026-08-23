"""Canonical wire-format builder — both renderers must carry the same content for the
same prefix, and the export's non-verbatim spots (reasoning, apply_patch, images, empty
assistant turns) must be handled identically regardless of which encoding produced them.
"""
from __future__ import annotations

from research.capability_fitting.canonical import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    convert_tool,
    render_anthropic,
    render_openai,
    to_canonical,
)


class TestToCanonical:
    def test_system_messages_collect_into_system_field(self, claude_trajectory_line):
        request = to_canonical(claude_trajectory_line["input"], claude_trajectory_line["tools"])
        assert "you are Viktor" in request.system

    def test_reasoning_items_are_dropped(self, gpt_trajectory_line):
        # gpt_reasoning() has no encrypted content — unreplayable for any model (canonical.py docstring).
        request = to_canonical(gpt_trajectory_line["input"], gpt_trajectory_line["tools"])
        assert not any(isinstance(item, Message) and "thinking about the fix" in item.text for item in request.items)

    def test_apply_patch_call_becomes_a_function_call_with_patch_argument(self, gpt_trajectory_line):
        request = to_canonical(gpt_trajectory_line["input"], gpt_trajectory_line["tools"])
        calls = [item for item in request.items if isinstance(item, ToolCall)]
        assert len(calls) == 1
        assert calls[0].name == "apply_patch"
        assert "patch" in calls[0].arguments

    def test_apply_patch_tool_def_becomes_a_function_with_one_string_arg(self):
        # conftest's generic make_tool() doesn't model the export's real `type: "custom"`
        # shape (see test_convert_tool below), so this test builds it directly.
        raw_tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "edit files",
            "format": {"definition": "start: ..."},
        }
        request = to_canonical([], [raw_tool])
        patch_tool = next(t for t in request.tools if t.name == "apply_patch")
        assert patch_tool.parameters["required"] == ["patch"]
        assert patch_tool.parameters["properties"]["patch"]["type"] == "string"

    def test_empty_assistant_message_is_dropped(self):
        items = [
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": ""}]},
        ]
        request = to_canonical(items, [])
        assert not any(isinstance(item, Message) and item.role is Role.ASSISTANT for item in request.items)

    def test_input_image_becomes_a_text_placeholder(self):
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look at this: "},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,[base64 image redacted]"},
                ],
            }
        ]
        request = to_canonical(items, [])
        assert "[image redacted]" in request.items[0].text

    def test_tool_call_and_output_survive_both_encodings(self, claude_trajectory_line, gpt_trajectory_line):
        for line in (claude_trajectory_line, gpt_trajectory_line):
            request = to_canonical(line["input"], line["tools"])
            assert any(isinstance(item, ToolCall) for item in request.items)
            assert any(isinstance(item, ToolResult) for item in request.items)


class TestRenderersCarrySameContent:
    def _tool_call_names(self, rendered_items):
        return {c.get("name") for c in rendered_items if c.get("type") == "function_call"}

    def test_openai_and_anthropic_carry_the_same_tool_call(self, claude_trajectory_line):
        request = to_canonical(claude_trajectory_line["input"], claude_trajectory_line["tools"])
        openai_payload = render_openai(request)
        anthropic_payload = render_anthropic(request)

        openai_names = {i["name"] for i in openai_payload["input"] if i.get("type") == "function_call"}
        anthropic_names = {
            block["name"]
            for msg in anthropic_payload["messages"]
            for block in msg["content"]
            if block.get("type") == "tool_use"
        }
        assert openai_names == anthropic_names == {"bash"}

    def test_anthropic_messages_never_open_on_assistant(self, claude_trajectory_line):
        request = to_canonical(claude_trajectory_line["input"], claude_trajectory_line["tools"])
        payload = render_anthropic(request)
        assert payload["messages"][0]["role"] == "user"

    def test_anthropic_tool_defs_use_input_schema(self, claude_trajectory_line):
        request = to_canonical(claude_trajectory_line["input"], claude_trajectory_line["tools"])
        payload = render_anthropic(request)
        assert all("input_schema" in t for t in payload["tools"])

    def test_openai_tool_defs_use_parameters(self, claude_trajectory_line):
        request = to_canonical(claude_trajectory_line["input"], claude_trajectory_line["tools"])
        payload = render_openai(request)
        assert all("parameters" in t for t in payload["tools"])

    def test_apply_patch_renders_identically_regardless_of_source_encoding(self, gpt_trajectory_line):
        # The apply_patch conversion happens once in to_canonical, uniformly, so both
        # renderers see the already-converted function — the whole point of §2 "verbatim"
        # not being achievable and the fix being uniform, not per-model.
        request = to_canonical(gpt_trajectory_line["input"], gpt_trajectory_line["tools"])
        openai_payload = render_openai(request)
        anthropic_payload = render_anthropic(request)
        assert "apply_patch" in self._tool_call_names(openai_payload["input"])
        assert any(
            block.get("name") == "apply_patch"
            for msg in anthropic_payload["messages"]
            for block in msg["content"]
            if block.get("type") == "tool_use"
        )


class TestConvertTool:
    def test_function_tool_passes_through(self):
        raw = {"type": "function", "name": "bash", "description": "run bash", "parameters": {"type": "object"}}
        tool = convert_tool(raw)
        assert tool.name == "bash"
        assert tool.parameters == {"type": "object"}

    def test_custom_tool_grammar_is_folded_into_description(self):
        raw = {
            "type": "custom",
            "name": "apply_patch",
            "description": "edit files",
            "format": {"definition": "start: ..."},
        }
        tool = convert_tool(raw)
        assert "start: ..." in tool.description
        assert tool.parameters["required"] == ["patch"]
