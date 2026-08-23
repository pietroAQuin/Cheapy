"""§3 response classification: malformed / message / tool_call, per provider."""
from __future__ import annotations

from research.capability_fitting.canonical import ToolDef
from research.capability_fitting.parser import classify_calls, parse_anthropic, parse_openai
from research.capability_fitting.scoring import ActionType

BASH = ToolDef(
    name="bash",
    description="",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)
TOOLS = {"bash": BASH}


class TestOpenAI:
    def test_valid_tool_call(self):
        response = {"output": [{"type": "function_call", "name": "bash", "arguments": '{"command": "ls"}'}]}
        action = parse_openai(response, TOOLS)
        assert action.type is ActionType.TOOL_CALL
        assert action.tool_names == frozenset({"bash"})

    def test_unknown_tool_name_is_malformed(self):
        response = {"output": [{"type": "function_call", "name": "nope", "arguments": "{}"}]}
        assert parse_openai(response, TOOLS).type is ActionType.MALFORMED

    def test_missing_required_argument_is_malformed(self):
        response = {"output": [{"type": "function_call", "name": "bash", "arguments": "{}"}]}
        assert parse_openai(response, TOOLS).type is ActionType.MALFORMED

    def test_message_with_text(self):
        response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}
        assert parse_openai(response, TOOLS).type is ActionType.MESSAGE

    def test_mixed_valid_and_invalid_calls_is_malformed(self):
        response = {
            "output": [
                {"type": "function_call", "name": "bash", "arguments": '{"command": "ls"}'},
                {"type": "function_call", "name": "nope", "arguments": "{}"},
            ]
        }
        assert parse_openai(response, TOOLS).type is ActionType.MALFORMED

    def test_parallel_valid_calls_collect_all_names(self):
        response = {
            "output": [
                {"type": "function_call", "name": "bash", "arguments": '{"command": "ls"}'},
                {"type": "function_call", "call_id": "2", "name": "bash", "arguments": '{"command": "pwd"}'},
            ]
        }
        # two calls to the same tool -> set collapses (§5.1 Stage B duplicate rule)
        assert parse_openai(response, TOOLS).tool_names == frozenset({"bash"})

    def test_empty_output_is_malformed(self):
        assert parse_openai({"output": []}, TOOLS).type is ActionType.MALFORMED

    def test_unparseable_response_is_malformed(self):
        assert parse_openai({"garbage": True}, TOOLS).type is ActionType.MALFORMED


class TestAnthropic:
    def test_valid_tool_use(self):
        response = {"content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls"}}]}
        action = parse_anthropic(response, TOOLS)
        assert action.type is ActionType.TOOL_CALL
        assert action.tool_names == frozenset({"bash"})

    def test_invalid_arguments_is_malformed(self):
        response = {"content": [{"type": "tool_use", "name": "bash", "input": {}}]}
        assert parse_anthropic(response, TOOLS).type is ActionType.MALFORMED

    def test_text_block(self):
        response = {"content": [{"type": "text", "text": "done"}]}
        assert parse_anthropic(response, TOOLS).type is ActionType.MESSAGE

    def test_empty_text_block_is_a_stop_not_malformed(self):
        # An empty/whitespace message is how a "stopped without acting" turn is encoded —
        # the GPT logs do exactly this. It must classify the same as a Claude stop turn
        # carrying real text, or the difference reads as served_model. See classify_run.
        response = {"content": [{"type": "text", "text": "   "}]}
        assert parse_anthropic(response, TOOLS).type is ActionType.MESSAGE

    def test_no_output_at_all_is_malformed(self):
        assert parse_anthropic({"content": []}, TOOLS).type is ActionType.MALFORMED


class TestClassifyCalls:
    def test_no_calls_is_malformed(self):
        assert classify_calls([], TOOLS).type is ActionType.MALFORMED

    def test_bad_schema_on_the_tool_definition_does_not_fail_the_call(self):
        broken_tool = ToolDef(name="weird", description="", parameters={"type": "not-a-real-type"})
        action = classify_calls([("weird", {})], {"weird": broken_tool})
        assert action.type is ActionType.TOOL_CALL
