"""src/cheapy/preprocessing/trajectory_analyzer.py: normalization, call recovery, and analyze()."""
import json

from cheapy.models.trajectory import ItemKind, Tool, Trajectory, ViktorEnvironment
from cheapy.preprocessing import trajectory_analyzer as ta
from tests.conftest import (
    claude_function_call,
    claude_function_call_output,
    claude_message,
    gpt_custom_tool_call,
    gpt_custom_tool_call_output,
    gpt_message,
    gpt_reasoning,
    make_line,
)


# --- normalize(): both encodings collapse to the same ItemKind ---

def test_normalize_reads_text_from_both_encodings():
    items = ta.normalize([
        claude_message("user", "typed encoding"),
        gpt_message("user", "untyped encoding"),
    ])
    assert [item.kind for item in items] == [ItemKind.USER_MESSAGE, ItemKind.USER_MESSAGE]
    assert [item.text for item in items] == ["typed encoding", "untyped encoding"]


def test_normalize_reasoning_item():
    items = ta.normalize([gpt_reasoning("mulling it over")])
    assert items[0].kind is ItemKind.REASONING
    assert items[0].text == "mulling it over"
    assert items[0].call_index == 0  # REASONING is a generated kind


def test_normalize_custom_tool_call_is_apply_patch():
    items = ta.normalize([
        gpt_custom_tool_call("apply_patch", "call_1", "*** Update File: a.py"),
        gpt_custom_tool_call_output("call_1", "patch applied"),
    ])
    call, output = items
    assert call.kind is ItemKind.TOOL_CALL
    assert call.is_custom_tool is True
    assert call.tool_arguments == "*** Update File: a.py"
    assert call.tool_call_id == "call_1"
    assert output.kind is ItemKind.TOOL_OUTPUT
    assert output.tool_output == "patch applied"
    assert output.tool_call_id == "call_1"


def test_normalize_function_call_arguments_not_confused_with_custom_input():
    items = ta.normalize([claude_function_call("bash", "call_1", '{"cmd": "ls"}')])
    assert items[0].is_custom_tool is False
    assert items[0].tool_arguments == '{"cmd": "ls"}'


def test_normalize_counts_input_images():
    item = {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "see attached"},
            {"type": "input_image", "image_url": "data:redacted"},
            {"type": "input_image", "image_url": "data:redacted"},
        ],
    }
    assert ta.normalize([item])[0].images == 2


def test_unknown_tool_name_decays_to_unknown():
    assert Tool("some_future_tool") is Tool.UNKNOWN


# --- call-index segmentation: maximal runs of generated kinds ---

def test_call_index_groups_consecutive_generated_items():
    items = ta.normalize([
        claude_message("system", "sys"),          # not generated -> None
        claude_message("user", "hi"),              # not generated -> None
        claude_message("assistant", "thinking"),   # generated -> call 0
        claude_function_call("bash", "c1", "{}"),  # generated -> call 0 (same run)
        claude_function_call_output("c1", "ok"),   # not generated -> breaks run
        claude_message("assistant", "done"),       # generated -> call 1
    ])
    assert [item.call_index for item in items] == [None, None, 0, 0, None, 1]


# --- analyze(): end-to-end record -> Trajectory ---

def test_analyze_claude_encoded_trajectory(claude_trajectory_line):
    trajectory = ta.analyze(claude_trajectory_line, id=0)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.id == 0
    assert trajectory.served_model == "claude-opus-5"
    assert trajectory.viktor_environment is ViktorEnvironment.SLACK
    assert trajectory.is_subagent is False
    assert trajectory.total_calls == 2
    assert trajectory.total_tool_calls == 1
    assert trajectory.total_ai_messages == 2
    assert trajectory.total_user_messages == 1
    assert trajectory.toolset_size == 5
    assert Tool.BASH in trajectory.toolset
    assert trajectory.avg_tools_per_call == 0.5  # 1 tool call over 2 recovered calls


def test_analyze_gpt_encoded_trajectory_is_teams(gpt_trajectory_line):
    trajectory = ta.analyze(gpt_trajectory_line, id=1)
    assert trajectory.served_model == "gpt-5.6-terra"
    assert trajectory.viktor_environment is ViktorEnvironment.TEAMS
    assert trajectory.total_calls == 1
    assert trajectory.total_tool_calls == 1  # custom_tool_call counts as a tool call


def test_analyze_detects_subagent_trajectory():
    line = make_line(
        model="claude-sonnet-5",
        tools=["submit_subagent_result", "view_image", "submit_draft"],
        input_items=[claude_message("user", "do the subtask")],
    )
    trajectory = ta.analyze(line, id=0)
    assert trajectory.is_subagent is True


def test_analyze_counts_draft_submit_calls_not_as_terminator():
    line = make_line(
        model="claude-opus-5",
        tools=["submit_draft", "view_image"],
        input_items=[
            claude_function_call("submit_draft", "c1", "{}"),
            claude_function_call_output("c1", "approved"),
        ],
    )
    trajectory = ta.analyze(line, id=0)
    assert trajectory.total_draft_submit_calls == 1


def test_analyze_line_accepts_dict_or_json_string(claude_trajectory_line):
    from_dict = ta.analyze(claude_trajectory_line, id=0)
    from_json = ta.analyze(json.dumps(claude_trajectory_line), id=0)
    assert from_dict.total_tokens == from_json.total_tokens
    assert from_dict.total_calls == from_json.total_calls


def test_analyze_single_call_trajectory_has_no_cached_tokens():
    # Only one call means there is no predecessor prompt to have been cached.
    line = make_line(
        model="claude-opus-5",
        tools=["view_image"],
        input_items=[claude_message("user", "hi"), claude_message("assistant", "hello")],
    )
    trajectory = ta.analyze(line, id=0)
    assert trajectory.total_cached_tokens == 0


def test_analyze_cached_tokens_identity_with_forced_fallback_tokenizer(monkeypatch):
    # Force the chars/4 fallback so token counts are hand-verifiable regardless of
    # whether tiktoken happens to be installed in the environment running this test.
    monkeypatch.setattr(ta, "_get_encoder", lambda: None)

    line = make_line(
        model="claude-opus-5",
        tools=[],  # no tool schemas, so tools_tokens == 0 and math stays simple
        input_items=[
            claude_message("user", "a" * 40),        # call 0's prompt content
            claude_message("assistant", "b" * 40),   # call 0's output
            claude_message("user", "c" * 40),        # extends call 1's prompt
            claude_message("assistant", "d" * 40),   # call 1's output
        ],
    )
    trajectory = ta.analyze(line, id=0)

    # Call 0's prompt = item 0 (10 tokens at chars/4). Call 1's prompt = items 0-2
    # (30 tokens). total_tokens = 10 + 30 = 40; with CACHE_HIT_RATE=1.0 the cached
    # share is total_tokens minus the last call's own (uncached) prompt: 40 - 30 = 10,
    # i.e. exactly call 0's prompt getting reused as call 1's cached prefix.
    assert trajectory.total_tokens == 40
    assert trajectory.total_cached_tokens == 10


def test_count_tokens_empty_string_is_zero():
    assert ta.count_tokens("") == 0


def test_count_tokens_falls_back_without_tokenizer(monkeypatch):
    monkeypatch.setattr(ta, "_get_encoder", lambda: None)
    assert ta.count_tokens("a" * 40) == 10  # chars/4


def test_analyze_file_reads_one_trajectory_per_line(tmp_path, claude_trajectory_line, gpt_trajectory_line):
    path = tmp_path / "export.jsonl"
    path.write_text(
        json.dumps(claude_trajectory_line) + "\n" + json.dumps(gpt_trajectory_line) + "\n"
    )
    trajectories = list(ta.analyze_file(path))
    assert [t.id for t in trajectories] == [0, 1]
    assert [t.served_model for t in trajectories] == ["claude-opus-5", "gpt-5.6-terra"]
