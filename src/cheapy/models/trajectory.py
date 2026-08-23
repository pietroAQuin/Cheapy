"""Trajectory — everything known about one Viktor trajectory.

One JSONL line of the export is one `Trajectory`: a single decision point at which
the router picks the model to serve the next call. Lines are independent — no line's
input continues another's — so parsing is line-by-line, with no grouping step.

The per-call figures below are recovered by segmenting that line's `input` history:
every prior call's output is present in it as assistant / function_call / reasoning
items.

ESTIMATES: the export carries no `usage` field, so every token count here is derived
from character length (see CHARS_PER_TOKEN) and is an estimate, never a measurement.
Carry that label into any output or slide built on these fields.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

#: Token-count convention. No `usage` in the export and no tokenizer wired in, so every
#: `*_tokens` field on this model is (characters / 4) — an estimate, not a measurement.
CHARS_PER_TOKEN = 4


class ViktorEnvironment(str, Enum):
    """The chat platform Viktor is embedded in for this trajectory.

    Mutually exclusive and always present: of the 1,000 exported lines, 929 are Slack
    and 71 are Teams. Independent of the served model (~7% Teams within both model
    families), so this is a genuine free signal about the task.
    """

    SLACK = "slack"
    TEAMS = "teams"


class Tool(str, Enum):
    """A tool offered to the trajectory — the 26 distinct names observed in the export.

    Unknown names decay to `UNKNOWN` rather than raising, so a future chunk with new
    tools still parses.
    """

    # Offered in every request
    SUBMIT_DRAFT = "submit_draft"
    VIEW_IMAGE = "view_image"

    # Thread control (absent from subagent trajectories)
    SEND_MESSAGE_TO_THREAD = "send_message_to_thread"
    WAIT_FOR_BACKGROUND_WORK = "wait_for_background_work"
    SUBMIT_SUBAGENT_RESULT = "submit_subagent_result"

    # Slack surface
    COWORKER_SEND_SLACK_MESSAGE = "coworker_send_slack_message"
    COWORKER_SLACK_REACT = "coworker_slack_react"
    COWORKER_UPLOAD_TO_SLACK = "coworker_upload_to_slack"
    COWORKER_DOWNLOAD_FROM_SLACK = "coworker_download_from_slack"

    # Teams surface
    COWORKER_SEND_MSTEAMS_MESSAGE = "coworker_send_msteams_message"
    COWORKER_MSTEAMS_REACT = "coworker_msteams_react"
    COWORKER_CHANNEL_HISTORY = "coworker_channel_history"
    COWORKER_DELETE_CHANNEL_MESSAGE = "coworker_delete_channel_message"
    COWORKER_DOWNLOAD_FROM_CHANNEL = "coworker_download_from_channel"
    COWORKER_UPLOAD_TO_CHANNEL = "coworker_upload_to_channel"
    COWORKER_GET_CHANNEL_REACTIONS = "coworker_get_channel_reactions"
    COWORKER_LIST_MSTEAMS_CHANNELS = "coworker_list_msteams_channels"
    COWORKER_LIST_MSTEAMS_USERS = "coworker_list_msteams_users"

    # Sandbox — note these two blocks perfectly separate the model families in the
    # export (bash/file_* <=> claude-*, shell_command/apply_patch <=> gpt-*). They are a
    # provider harness artifact, not a task signal: a score that reads them is reading
    # `served_model` by proxy, which the router contract forbids.
    BASH = "bash"
    FILE_READ = "file_read"
    FILE_EDIT = "file_edit"
    FILE_WRITE = "file_write"
    SHELL_COMMAND = "shell_command"
    APPLY_PATCH = "apply_patch"

    # Rare
    MEMORY_SEARCH = "memory_search"
    COWORKER_DISPLAY_UI_ELEMENTS = "coworker_display_ui_elements"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "Tool":
        return cls.UNKNOWN


class ItemKind(str, Enum):
    """One item of a trajectory, in a representation common to both encodings.

    The export ships two shapes for the same information — `claude-*` lines use typed
    items with list content, `gpt-*` lines use untyped items with string content — and
    the two dialects also disagree on how a tool call is expressed (`function_call`
    versus `custom_tool_call`). This enum is the union both collapse into.
    """

    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"


#: Kinds the model produces. A maximal consecutive run of these is one call.
GENERATED_KINDS = frozenset(
    {ItemKind.ASSISTANT_MESSAGE, ItemKind.REASONING, ItemKind.TOOL_CALL}
)


class NormalizedItem(BaseModel):
    """One item of the trajectory, encoding-independent.

    Carries everything the raw item held that is not pure envelope, so a later stage
    can re-derive signals the current field set does not cover — retry patterns, error
    strings, per-tool cost, prefix boundaries — without re-parsing the export or
    re-learning which of the two encodings a line uses.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ItemKind
    call_index: int | None = Field(
        default=None,
        description=(
            "Index of the call that produced this item, or None for items the model did "
            "not generate (system/user messages and tool outputs)."
        ),
    )
    text: str = Field(
        default="",
        description=(
            "Unified text: message content from either encoding, or the summary text of "
            "a reasoning item."
        ),
    )
    tool_name: str | None = None
    tool_arguments: str | None = Field(
        default=None,
        description="Raw argument string as sent — JSON for function_call, patch text for apply_patch.",
    )
    tool_output: str | None = Field(
        default=None, description="Raw output string returned to the model."
    )
    tool_call_id: str | None = Field(
        default=None, description="Pairs a TOOL_CALL with its TOOL_OUTPUT."
    )
    is_custom_tool: bool = Field(
        default=False,
        description=(
            "Item used the custom_tool_call dialect (apply_patch). DIAGNOSTIC ONLY: this "
            "dialect occurs on gpt-* lines exclusively, so a score reading it is reading "
            "served_model by proxy."
        ),
    )
    images: int = Field(
        default=0, ge=0, description="`input_image` parts carried on this item."
    )
    est_tokens: int = Field(
        default=0, ge=0, description="ESTIMATE of this item's billable tokens."
    )
    item_id: str | None = Field(
        default=None, description="Provider item id, where the raw item carried one."
    )


class Trajectory(BaseModel):
    """One Viktor trajectory, as parsed from a single line of the export.

    Built by `src/cheapy/preprocessing/trajectory_analyzer.py`; consumed read-only by the
    `src/cheapy/routing/` scoring stages, which enrich `ModelLLM` objects instead.
    """

    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="Zero-based index of this trajectory's line in the export.")

    served_model: str = Field(
        description=(
            "The anonymized id of the model that actually served this call. For the "
            "HOLD / CHANGE TO comparison and switching-cost reasoning only — the price "
            "and performance scores must not read it, or they predict the log instead "
            "of improving on it."
        )
    )

    normalized_items: list[NormalizedItem] = Field(
        default_factory=list,
        repr=False,
        description=(
            "The whole trajectory in the common representation, in order — the raw "
            "material behind every count on this model. Kept so a downstream stage can "
            "infer something this field set does not precompute without going back to "
            "the export or handling the two encodings again. Excluded from repr: a long "
            "trajectory holds hundreds of items."
        ),
    )

    is_subagent: bool = Field(
        description=(
            "Trajectory runs as a subagent: it is offered submit_subagent_result and no "
            "thread-control tools. 8 of the 1,000 exported lines."
        )
    )

    viktor_environment: ViktorEnvironment = Field(
        description="Chat platform this trajectory runs on, inferred from the toolset."
    )

    toolset: list[Tool] = Field(
        description="Tools offered to this trajectory, as declared in the line's `tools` field."
    )

    toolset_size: int = Field(ge=0, description="Number of tools offered. Bimodal at 12 and 10.")

    # --- per-call averages (over the calls recovered from this line's input history) ---

    avg_tools_per_call: float = Field(
        ge=0, description="Mean tool invocations per recovered model call."
    )

    avg_images_received_per_call: float = Field(
        ge=0, description="Mean images received per call across the trajectory."
    )

    avg_output_tokens_per_call: float = Field(
        ge=0,
        description=(
            "ESTIMATE (chars/4). Mean tokens generated per recovered model call. The final "
            "call's output is not in the export, so it is excluded from this mean."
        ),
    )

    avg_input_tokens_per_call: float = Field(
        ge=0,
        description="ESTIMATE (chars/4). Mean prompt size per recovered model call.",
    )

    # --- totals ---

    total_calls: int = Field(
        ge=0, description="Model calls recovered from this line's input history."
    )

    total_tokens: int = Field(
        ge=0,
        description=(
            "ESTIMATE (chars/4). Total input tokens billed across the trajectory's calls, "
            "before any cache discount."
        ),
    )

    total_cached_tokens: int = Field(
        ge=0,
        description=(
            "ESTIMATE. Share of `total_tokens` served from the provider's prefix cache, "
            "inferred from item-level prefix overlap between consecutive calls — the "
            "export has no `usage.cached_tokens`. A model switch resets this cache, which "
            "is the penalty the price model has to weigh against per-token savings."
        ),
    )

    last_call_input_tokens: int = Field(
        ge=0,
        description=(
            "ESTIMATE (chars/4). Size of the most recently recovered call's full prompt "
            "(tool schemas + history up to that point) — not a sum across calls. Since "
            "every call resends the whole history to date, this is the best available "
            "proxy for the size of the *next* call's prompt. Deliberately distinct from "
            "`total_tokens`, which sums every past call's own (already-superseded) "
            "full-history snapshot and so overstates next-call size, often by a wide "
            "margin on long trajectories. 0 if the trajectory has no recovered calls."
        ),
    )

    total_tool_calls: int = Field(ge=0, description="`function_call` items in the history.")

    total_images_received: int = Field(
        ge=0,
        description=(
            "Images that came IN to Viktor: `input_image` content parts in the history. "
            "338 parts across 64 of the 1,000 lines, always carried on user messages. "
            "Redacted to placeholder data URLs, so their real token cost is unrecoverable."
        ),
    )

    total_ai_messages: int = Field(ge=0, description="Assistant `message` items in the history.")

    total_user_messages: int = Field(ge=0, description="User `message` items in the history.")

    total_draft_submit_calls: int = Field(
        ge=0,
        description=(
            "Calls to submit_draft — the human-approval gate that executes a previously "
            "drafted, side-effecting action. Offered in every request but invoked in only "
            "14 of 1,000 lines, and never as a terminator: read it as 'performed a gated "
            "irreversible action', not as a completion signal."
        ),
    )
