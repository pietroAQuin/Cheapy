# Viktor LLM-Router

A router for **Viktor**, an autonomous LLM coworker that operates inside Slack and Microsoft
Teams. Viktor runs long, tool-heavy trajectories; today a single model serves every call of a
trajectory. This project decides, **per message**, which model out of the available pool should
serve the *next* call — and whether that means staying put or switching.

Built for the Viktor Challenge at the TUM.ai hackathon (Munich, 22–23 Aug 2026).

> **Read this first.** `routing_scheme.md` is the design spec written by the team;
> `code_agent_utils/AGENTS.md` is the challenge briefing from the organizers. If those two
> disagree with this README, they win — and this README should be updated to match.

---

## 1. What the router does

**Input:** a JSON/JSONL file containing one Viktor trajectory or a series of them.
**Output:** for each trajectory, an ordered list of candidate models, best first, and a decision:

| Decision | Meaning |
|---|---|
| `HOLD` | The top-scored model is the one already serving the trajectory — keep it. |
| `CHANGE TO <model>` | A different model scored highest — switch for the next call. |

The score behind that decision is a **weighted average of a price score and a performance
score**. The weights are user-configurable, so a caller can dial the router from
"cheapest acceptable" to "best regardless of cost" and trace out a cost–quality frontier
rather than committing to one operating point.

---

## 2. Pipeline

```
                data/*.jsonl (one JSON line = one LLM request)
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                pre_processing/                  │
        │  model_list.py           trajectory_analyzer.py │
        │  builds List[LLMModel]   builds one Trajectory  │
        └────────────────────────┬────────────────────────┘
                                 │  (models, trajectory)
                                 ▼
                    router_models/price_model.py
                    → sets LLMModel.price_score
                                 │
                                 ▼
                 router_models/performance_model.py
                    → sets LLMModel.performance_score
                                 │
                                 ▼
                      router_models/model.py
        final_score = w_price·price_score + w_perf·performance_score
                                 │
                                 ▼
        ordered List[LLMModel]  →  HOLD | CHANGE TO <model>
```

Each stage **enriches the same `LLMModel` objects in place** rather than returning new types.
A scoring stage is therefore a function of `(Trajectory, List[LLMModel]) -> List[LLMModel]`,
and stages compose in the order above.

---

## 3. Repository layout

| Path | Role | Status |
|---|---|---|
| `data_models/LLMModel.py` | `ModelLLM` — everything known about one candidate model | **empty stub** |
| `data_models/Trajectory.py` | `Trajectory` — everything known about one trajectory | **empty stub** |
| `pre_processing/model_list.py` | Constructs the candidate pool of `ModelLLM` objects | **empty stub** |
| `pre_processing/trajectory_analyzer.py` | Parses one JSONL line/group into a `Trajectory` | **empty stub** |
| `router_models/price_model.py` | Assigns `price_score` | **empty stub** |
| `router_models/performance_model.py` | Assigns `performance_score` | **empty stub** |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision | **empty stub** |
| `data/` | The redacted export (gitignored, challenge-use only) | present locally |
| `routing_scheme.md` | The team's design spec — source of truth for architecture | written |
| `code_agent_utils/` | Organizer-supplied briefing + `/setup`, `/make-presentation`, `/prepare-submission` skills | supplied |

Every Python file listed above is currently a **zero-byte placeholder**. The structure is
decided; the implementations are not. Nothing imports anything yet — you are not breaking an
existing contract, you are writing the first one.

Data models are **Pydantic `BaseModel`s**. Keep them that way: validation at the boundary is
what makes a half-finished pipeline debuggable.

---

## 4. The dataset

`data/trajectories_v1_01.jsonl` — 1,000 lines, ~105 MB, one **LLM request** per line in the
OpenAI-compatible Responses format. Exactly three top-level fields:

- **`model`** — the anonymized id that actually served the call.
- **`input`** — the full request history up to that call.
- **`tools`** — the function definitions available to that call.

There is **no `output` and no `usage`**. No token counts, no latency, no quality labels, no
trajectory ids.

### What is actually in this chunk (measured, not assumed)

Model distribution across the 1,000 requests:

| Model | Requests |
|---|---|
| `claude-opus-5` | 331 |
| `claude-sonnet-5` | 281 |
| `gpt-5.6-terra` | 113 |
| `gpt-5.6-sol` | 112 |
| `claude-fable-5` | 71 |
| `claude-opus-4-8` | 69 |
| `gpt-5.6-luna` | 20 |
| `claude-opus-4-6` | 2 |
| `claude-sonnet-4-6` | 1 |

Input item types: `function_call` (10,123), `function_call_output` (10,123), `message` (8,583 —
5,602 assistant / 1,981 user / 1,000 system), `reasoning` (1,792, gpt-family only, summaries
empty), `custom_tool_call` + `custom_tool_call_output` (299 each, e.g. `apply_patch`).

(These counts are summed over requests, and each request re-sends the whole prior history, so
they over-count long trajectories — read them as proportions, not as totals.)

**This workload is overwhelmingly tool-calling, not chat:** roughly 1.2 `function_call` items
for every message item, and 5,602 of the 8,583 messages are assistant turns. Any performance
score that only looks at user-facing message text is scoring the wrong thing.

Toolsets cluster into recognizable Viktor surfaces — a Slack-flavored set
(`coworker_send_slack_message`, `coworker_upload_to_slack`, …, 929 requests), a Teams-flavored
set (`coworker_*_msteams_*`, 71 requests), and two coding dialects: `bash`/`file_read`/
`file_edit`/`file_write` (755) versus `shell_command`/`apply_patch` (245). Tool count per
request is bimodal at 12 (692) and 10 (224). **The toolset is a strong, free signal about what
kind of task a trajectory is** — use it.

### Redaction

Entities are replaced by stable named placeholders (`PII_PERSON_7`, `PII_COMPANY_1`,
`PII_URL_8`, `<ID_13>`), consistent *within* a trajectory so references still resolve. Images
are placeholder data URLs. Do not try to de-anonymize anything.

### Reconstructing trajectories

The export has no trajectory ids. Requests must be grouped by their **opening messages** (same
system prompt + same first user text) and ordered by **input length** — each call's input
contains every item of the call before it. This is what
`pre_processing/trajectory_analyzer.py` is for.

Two consequences worth internalizing:

1. **Earlier model outputs are recoverable.** What the model returned on call *i* shows up
   inside call *i+1*'s input as assistant / `function_call` items. Only the *final* call's
   output is genuinely lost. That is where most of the available quality signal lives.
2. **One model serves all calls of a trajectory** — that is the premise, and the variation
   *across* trajectories is the natural experiment the whole evaluation rests on. The loader
   should flag any violation rather than silently averaging over it.

---

## 5. Scoring

> **Both scoring criteria are undefined as of this writing.** `routing_scheme.md` says so
> explicitly. This section records the constraints any implementation must respect, not a
> design that has been agreed.

### Shared contract

- Both `price_score` and `performance_score` must be on the **same, comparable scale** —
  otherwise the weighted average in `model.py` is meaningless. Normalize to `[0, 1]`, higher
  is better (so cheap → high price score), and normalize *within the candidate pool for this
  trajectory*, not globally.
- Both are functions of `(Trajectory, ModelLLM)`. Neither may look at the `model` field of the
  trajectory it is scoring, except where the router deliberately reasons about switching cost
  — see the cache trap below. Leaking the served model into the score is how you accidentally
  build a classifier that predicts the log instead of a router that improves on it.

### Price

- **Token counts are estimates.** There is no `usage` field. This repo's convention is
  chars/4 unless someone wires in a real tokenizer. Every number derived from them is an
  estimate and must be labeled as one in the writeup. Never present estimated tokens as
  measured.
- **Pricing is an assumption.** The model ids are anonymized, so no public price sheet applies
  and the tier order is not published. If the organizers post prices, they win; otherwise
  state the assumption explicitly and keep it in one place so it can be swapped.
- **The cache trap — this is the crux of the price model.** Providers cache the shared input
  prefix across a task's calls. A model switch **resets that cache**, so every call after a
  switch pays full price for the entire accumulated prefix. In trajectories this long, that
  penalty can dwarf the per-token saving that motivated the switch. Since there is no
  `usage.cached_tokens`, the cached share has to be *inferred* from item-level prefix overlap.
  A `CHANGE TO` decision that ignores this will look brilliant and be wrong.

### Performance

- No quality labels exist in the export. Any performance score is a **proxy** — derive it from
  observable structure (task type inferred from the toolset, trajectory length, tool-call
  density, error/retry patterns in `function_call_output`, whether the trajectory reached
  `submit_draft`), or from judge-model rescoring of individual calls.
- Judge-model rescoring is permitted with the team's own API keys, but the core pipeline must
  stay runnable **offline on a laptop, with no GPU and no API keys required**.

### Aggregation

`model.py` owns the weighted average, the ranking, and the HOLD/CHANGE decision. Keep the
weights a parameter threaded through the call, not a module-level constant — sweeping them is
how the cost–quality frontier gets produced, and the frontier is the headline artifact.

---

## 6. Evaluating the router (the hard part)

The log shows only the model that actually ran. Estimating what a *different* route would have
cost or delivered is off-policy evaluation, and it is the depth of this challenge — matching or
weighting across comparable trajectories, judge-model rescoring of single calls.

Naming where your estimate can fail is **scored, not penalized**. The deliverable is a
cost–quality frontier, not a single point, and a five-minute presentation with one chart, one
claim, and one known weakness.

---

## 7. Conventions for agents working here

- **Follow the file structure in `routing_scheme.md`.** If a change needs a new stage or a new
  module, update that spec in the same commit — it is the architecture's source of truth and
  the next agent reads it before this README.
- **Pydantic models in `data_models/`, nowhere else.** Import them; don't redefine shapes
  inline.
- **Enrich, don't replace.** Scoring stages set attributes on the `ModelLLM` objects they are
  handed and return the same list.
- **Never commit the dataset.** It is challenge-use only, no redistribution (`data/LICENSE`).
  `*.jsonl` and `*.jsonl.tar.gz` are gitignored — keep it that way, and never upload traces to
  any external service.
- **Label estimates as estimates.** Token counts, prices, and quality proxies are all inferred
  here. Carry that honesty into code comments, output columns, and slides.
- **Don't invent challenge facts** — deadlines, prizes, credits, price sheets. If it isn't in
  `AGENTS.md` or posted by the organizers, ask in the challenge Discord.

---

## 8. Open decisions

Live list — resolve these in `routing_scheme.md`, then update here.

1. Price scoring formula, and the assumed price sheet behind it.
2. Performance scoring formula, and which observable proxies feed it.
3. `ModelLLM` field set — beyond `price_score`, `performance_score`, and the final score.
4. `Trajectory` field set — what `trajectory_analyzer.py` extracts and precomputes.
5. Where the candidate pool comes from: hardcoded in `model_list.py`, or derived from the
   models observed in the export.
6. Routing granularity: `routing_scheme.md` frames the decision as "the model for the next
   message", while the dataset premise is one model per trajectory. Decide whether the router
   may switch mid-trajectory — and if so, the cache penalty is not optional.
7. Default weights, and the sweep used to produce the frontier.
