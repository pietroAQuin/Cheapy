"""A quality **base value** per known model id, derived from public leaderboards.

Per docs/FULL_REPORT.md §5 ("Performance"), no quality labels exist in the export — any
`performance_score` is necessarily a proxy. This module is the fallback layer beneath
that proxy: when a trajectory's own signal is ambiguous or candidates disagree, the
router needs *something* to fall back on that isn't a vibe. These four dictionaries are
that something — a quality prior derived from three independent, public, reproducible
benchmarks rather than this project's own judgment.

Snapshot captured 2026-08-23 from the live leaderboards below. All three are dashboards
that move as providers ship new checkpoints, so these numbers are a point-in-time
reading, not a fixed fact — re-scrape and rebuild this file if the scores drift enough to
matter. Every score records the specific model *variant* (reasoning-effort setting) the
leaderboard listed it under, because none of these boards guarantee one row per model
family — see the per-entry comments below.

Sources (one dict each), all restricted to the 9 model ids this project's candidate pool
uses (`src/cheapy/preprocessing/model_list.py`'s `_MODEL_FAMILIES`):

1. **Artificial Analysis Intelligence Index (v4.1.1)**
   https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index
   Automated capability benchmark (9 fixed evals, pass@1, weighted 0-100). Answers: does
   it solve hard, verifiable tasks?
2. **Agent Arena** — https://arena.ai/leaderboard/agent
   Real-world agentic performance via causal tracing over live "Agent Mode" sessions.
   Headline metric: Net Improvement, percentage points vs. the average model. Answers:
   does using this model make real work turn out better?
3. **Text Arena / LMArena** — https://arena.ai/leaderboard/text
   Human preference: blind pairwise votes, Bradley-Terry model with style control.
   Answers: do humans prefer its responses?

Why these three and not one: they measure partially independent things (capability,
real-world outcome, subjective preference), each has its own bias (AA can run
STEM-heavy, Text rewards likeability, Agent reflects Arena's own user mix), and gaming
all three at once is much harder than gaming one. Equal-weighting them lets the biases
cancel instead of compounding — see `QUALITY_BASE_SCORE` below for how that weighting is
actually applied.

Every score below reflects **peak capability**: the highest reasoning-effort variant of
each model that the given leaderboard listed a score for, not necessarily the same
effort setting across benchmarks (a board that only ever tested one variant of a model
gets that variant; a board that tested several gets the best of those).
"""

from __future__ import annotations

# --- 1. Artificial Analysis Intelligence Index ------------------------------------
#
# Scale: 0-100, higher is better. One score per model; AA's page for each model lists
# only the reasoning-effort variant(s) it evaluated, so "peak" here is whatever AA itself
# scored highest for that model id (never a different provider-side variant AA didn't
# test) -- see the inline comment on each row for the exact labeled variant used.
AA_INTELLIGENCE_INDEX: dict[str, float] = {
    "claude-opus-5": 63,        # "Claude Opus 5 (Adaptive Reasoning, Max Effort / Xhigh Effort)" -- both 63
    "claude-opus-4-8": 57,      # "Claude Opus 4.8 (Adaptive Reasoning, Max Effort)"
    "claude-opus-4-6": 39,      # "Claude Opus 4.6 (Non-reasoning, High Effort)" -- only variant AA lists
    "claude-sonnet-5": 55,      # "Claude Sonnet 5 (Adaptive Reasoning, Max Effort)"
    "claude-sonnet-4-6": 37,    # "Claude Sonnet 4.6 (Non-reasoning)" -- deprecated model, only variant AA lists
    "claude-fable-5": 62,       # "Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback)"
    "gpt-5.6-sol": 61,          # "GPT-5.6 Sol (max)"
    "gpt-5.6-terra": 57,        # "GPT-5.6 Terra (max)"
    "gpt-5.6-luna": 52,         # "GPT-5.6 Luna (max)"
}

# --- 2. Agent Arena -----------------------------------------------------------------
#
# Scale: Net Improvement, percentage points vs. the average model (can be negative in
# general; every model in our pool happens to score positive). Higher is better. Several
# effort variants are listed per model on this board; the value here is the best-scoring
# variant for that model id.
AGENT_ARENA_NET_IMPROVEMENT: dict[str, float] = {
    "claude-opus-5": 12.47,     # "Claude Opus 5 (High)" -- beats "(Max)" at 12.00
    "claude-opus-4-8": 9.55,    # "Claude Opus 4.8 (High)" -- beats unlabeled variant at 2.51
    "claude-opus-4-6": 6.60,    # "Claude Opus 4.6" -- only variant listed
    "claude-sonnet-5": 6.62,    # "Claude Sonnet 5 (High)" -- only variant listed
    "claude-sonnet-4-6": 2.88,  # "Claude Sonnet 4.6" -- only variant listed
    "claude-fable-5": 11.57,    # "Claude Fable 5 (High)" -- only variant listed
    "gpt-5.6-sol": 9.74,        # "GPT 5.6 Sol (xHigh)"
    "gpt-5.6-terra": 3.19,      # "GPT 5.6 Terra (xHigh)"
    "gpt-5.6-luna": 4.04,       # "GPT 5.6 Luna (xHigh)"
}

# --- 3. Text Arena / LMArena ---------------------------------------------------------
#
# Scale: Bradley-Terry rating with style control (Elo-like; this snapshot's 9 values
# span roughly 1450-1510). Higher is better. Rows are keyed by exact model+variant slug
# on the leaderboard; the value here is the best-scoring slug for that model id.
TEXT_ARENA_RATING: dict[str, float] = {
    "claude-opus-5": 1493,      # "claude-opus-5-high" -- beats "claude-opus-5-max" at 1487
    "claude-opus-4-8": 1482,    # "claude-opus-4-8-high" -- beats unlabeled slug at 1473
    "claude-opus-4-6": 1504,    # "claude-opus-4-6-high" -- beats unlabeled slug at 1497
    "claude-sonnet-5": 1461,    # "claude-sonnet-5-high" -- only slug listed
    "claude-sonnet-4-6": 1472,  # "claude-sonnet-4-6" -- only slug listed
    "claude-fable-5": 1508,     # "claude-fable-5" -- only slug listed
    "gpt-5.6-sol": 1482,        # "gpt-5.6-sol-xhigh" -- only slug listed
    "gpt-5.6-terra": 1465,      # "gpt-5.6-terra-xhigh" -- only slug listed
    "gpt-5.6-luna": 1451,       # "gpt-5.6-luna-xhigh" -- only slug listed
}

# --- 4. Composite quality base score --------------------------------------------------
#
# Scale: [0, 1]; 1.0 is the ceiling a model would hit by topping all three benchmarks
# simultaneously, not a value forced onto whoever happens to lead this particular
# 9-model pool -- no model here reaches it. Built in two steps so the three source
# benchmarks -- on wildly different native scales (0-100, a percentage, a ~1450-1510
# rating) -- contribute *equally* rather than whichever has the widest raw numeric
# spread dominating a naive average:
#
#   1. Min-max normalize each of the three dicts above to [0, 1] **within this same
#      9-model pool** (worst id in that benchmark -> 0.0, best -> 1.0).
#   2. Average the three normalized values per model, unweighted (1/3 each, per "all
#      benchmarks should have the same weight"). No further rescaling: the composite is
#      that weighted average, full stop, so it stays a direct read of "how this model's
#      three normalized scores actually average out," not a value stretched back out to
#      make the current pool's leader hit exactly 1.0.
#
# This is a base value, not a substitute for `performance_score`: it has no idea which
# model is best for *this* trajectory, only which one benchmarks best in general. Use it
# as the fallback prior when a trajectory's own signal doesn't clearly favor one
# candidate -- see docs/FULL_REPORT.md §5.
QUALITY_BASE_SCORE: dict[str, float] = {
    "claude-fable-5": 0.9559,
    "claude-opus-5": 0.9123,
    "gpt-5.6-sol": 0.7274,
    "claude-opus-4-8": 0.6695,
    "claude-opus-4-6": 0.4649,
    "claude-sonnet-5": 0.4192,
    "gpt-5.6-terra": 0.3491,
    "gpt-5.6-luna": 0.2326,
    "claude-sonnet-4-6": 0.1228,
}
