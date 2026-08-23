"""Shared fixtures/helpers for the test suite.

The real export (`data/*.jsonl`) is gitignored and not present in the repo, so every
test here builds minimal synthetic records by hand instead of reading it. Records
follow the two encodings documented in `src/cheapy/preprocessing/trajectory_analyzer.py`.
"""
from __future__ import annotations

import pytest


def make_tool(name: str) -> dict:
    """Minimal tool schema entry, as it appears in a line's top-level `tools` list."""
    return {"name": name, "description": f"stub schema for {name}", "parameters": {}}


def make_line(model: str, tools: list[str], input_items: list[dict]) -> dict:
    """One export line: the three top-level fields `analyze()` reads."""
    return {
        "model": model,
        "tools": [make_tool(name) for name in tools],
        "input": input_items,
    }


# --- typed / list encoding (claude-*) ---

def claude_message(role: str, text: str) -> dict:
    return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}


def claude_function_call(name: str, call_id: str, arguments: str = "{}") -> dict:
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


def claude_function_call_output(call_id: str, output: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# --- untyped / string encoding (gpt-*) ---

def gpt_message(role: str, text: str) -> dict:
    return {"role": role, "content": text}


def gpt_reasoning(summary_text: str) -> dict:
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": summary_text}]}


def gpt_custom_tool_call(name: str, call_id: str, patch_input: str) -> dict:
    return {"type": "custom_tool_call", "name": name, "call_id": call_id, "input": patch_input}


def gpt_custom_tool_call_output(call_id: str, output: str) -> dict:
    return {"type": "custom_tool_call_output", "call_id": call_id, "output": output}


@pytest.fixture
def claude_trajectory_line() -> dict:
    """A two-call, Slack-flavored trajectory in the typed/list encoding."""
    return make_line(
        model="claude-opus-5",
        tools=["bash", "file_read", "submit_draft", "view_image", "coworker_send_slack_message"],
        input_items=[
            claude_message("system", "you are Viktor"),
            claude_message("user", "list the files"),
            claude_message("assistant", "I'll check the directory."),
            claude_function_call("bash", "call_1", '{"cmd": "ls"}'),
            claude_function_call_output("call_1", "file1\nfile2"),
            claude_message("assistant", "Done, there are two files."),
        ],
    )


@pytest.fixture
def gpt_trajectory_line() -> dict:
    """A one-call, Teams-flavored trajectory in the untyped/string encoding."""
    return make_line(
        model="gpt-5.6-terra",
        tools=["shell_command", "apply_patch", "coworker_send_msteams_message"],
        input_items=[
            gpt_message("system", "you are Viktor"),
            gpt_message("user", "fix the bug"),
            gpt_reasoning("thinking about the fix"),
            gpt_custom_tool_call("apply_patch", "call_1", "*** Update File: a.py\n..."),
            gpt_custom_tool_call_output("call_1", "patch applied"),
        ],
    )
