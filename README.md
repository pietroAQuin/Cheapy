# Viktor LLM-Router

A router for **Viktor**, an autonomous LLM coworker that operates inside Slack and Microsoft
Teams. Viktor runs long, tool-heavy trajectories; today a single model serves every call of a
trajectory. This project ingests those trajectories and decides which model, out of a pool of
available ones, should serve the next message.

Built for the Viktor Challenge at the TUM.ai hackathon (Munich, 22–23 Aug 2026).

**This README is the architecture spec.** It is the only design document in the repo — there is
no separate spec file to consult. When the design changes, change it here, in the same commit
as the code. `code_agent_utils/AGENTS.md` is the organizers' challenge briefing; where it
contradicts this file on dataset shape, licensing, or judging, it wins.

---

## 1. What the router does

**Input:** a JSON/JSONL file containing one Viktor trajectory or a series of them.
**Output:** for each trajectory, an ordered list of candidate models, best first, and a decision:

| Decision | Meaning |
|---|---|
| `HOLD` | The top-scored model is the one already serving the trajectory — keep it. |
| `CHANGE TO <model>` | A different model scored highest — switch for the next call. |

The score behind that decision is a **weighted average of a price score and a performance
score**. The weights are caller-configurable, so the router can be dialed from "cheapest
acceptable" to "best regardless of cost" — sweeping them traces a cost–quality frontier instead
of committing to a single operating point.

---

## 2. Architecture

```
                data/*.jsonl (one JSON line = one LLM request)
                                 │
        ┌────────────────────────┴────────────────────────┐
        │                pre_processing/                  │
        │  model_list.py           trajectory_analyzer.py │
        │  builds List[ModelLLM]   builds one Trajectory  │
        └────────────────────────┬────────────────────────┘
                                 │  (trajectory, models)
                                 ▼
                    router_models/price_model.py
                    → sets ModelLLM.price_score
                                 │
                                 ▼
                 router_models/performance_model.py
                    → sets ModelLLM.performance_score
                                 │
                                 ▼
                      router_models/model.py
        final_score = w_price·price_score + w_perf·performance_score
                                 │
                                 ▼
        ordered List[ModelLLM]  →  HOLD | CHANGE TO <model>
```

### The two core classes

Both live in `data_models/` as Pydantic `BaseModel`s. Keep them there and keep them Pydantic —
validation at the boundary is what makes a half-finished pipeline debuggable. Import them; never
redefine these shapes inline.

- **`ModelLLM`** (`data_models/model_llm.py`) — everything known about one model in the candidate
  pool, including the scores the router stages attach to it. Current fields: `name`, `family`,
  `performance_score`, `price_score`, `final_score` (all `None` until the matching stage in the
  scoring chain above sets them — never treat an unset score as `0`), `context_window_size`, and
  the four per-1M-token price fields (`input_price_per_1m`, `cached_input_price_per_1m`,
  `output_price_per_1m`, `cached_output_price_per_1m`).
- **`Trajectory`** (`data_models/Trajectory.py`) — everything known about one trajectory.

### The two pre-processing entry points

- **`pre_processing/model_list.py`** — where `ModelLLM` objects are first created. Returns the
  list of every model to be considered for routing a trajectory. Currently a static pool: the 9
  ids/families observed in one scan of `data/trajectories_v1_01.jsonl`, paired with their real
  published pricing and context-window figures — not derived by scanning the export on every
  run. Revisit if a later chunk introduces new ids.
- **`pre_processing/trajectory_analyzer.py`** — where `Trajectory` objects are first created.
  Takes a single trajectory as JSON lines and returns one `Trajectory`.

### The scoring chain

A `Trajectory` and the model list are then fed through `router_models/`, in order:

1. `price_model.py` enriches each `ModelLLM` with a `price_score`.
2. `performance_model.py` enriches each `ModelLLM` with a `performance_score`.
3. `model.py` combines the two into a final score, sorts, and emits the HOLD / CHANGE TO
   decision.

Each stage **enriches the same `ModelLLM` objects in place** rather than returning new types. A
scoring stage is therefore a function of `(Trajectory, List[ModelLLM]) -> List[ModelLLM]`, and
stages compose in the order above. Adding a third criterion means adding a stage that follows
the same contract and a term in `model.py`'s average.

---

## 3. Repository layout

| Path | Role | Status |
|---|---|---|
| `data_models/model_llm.py` | `ModelLLM` — one candidate model | **implemented** |
| `data_models/Trajectory.py` | `Trajectory` — one trajectory | **empty stub** |
| `pre_processing/model_list.py` | Builds the candidate pool | **implemented** (static pool + real pricing; no scoring — that's `router_models/`'s job) |
| `pre_processing/trajectory_analyzer.py` | Parses JSON lines into a `Trajectory` | **empty stub** |
| `router_models/price_model.py` | Assigns `price_score` | **empty stub** |
| `router_models/performance_model.py` | Assigns `performance_score` | **empty stub** |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision | **empty stub** |
| `data/` | The redacted export (gitignored, challenge-use only) | present locally |
| `code_agent_utils/` | Organizers' briefing + `/setup`, `/make-presentation`, `/prepare-submission` skills | supplied |

`data_models/model_llm.py` and `pre_processing/model_list.py` are implemented; every other Python
file listed above is still a **zero-byte placeholder**. Nothing downstream imports the two
implemented files yet — you are not breaking an existing contract, you are writing the first one.

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
contains every item of the call before it. This is what `trajectory_analyzer.py` is for.

Two consequences worth internalizing:

1. **Earlier model outputs are recoverable.** What the model returned on call *i* shows up
   inside call *i+1*'s input as assistant / `function_call` items. Only the *final* call's
   output is genuinely lost. That is where most of the available quality signal lives.
2. **One model serves all calls of a trajectory** — that is the premise, and the variation
   *across* trajectories is the natural experiment the whole evaluation rests on. The analyzer
   should flag any violation rather than silently averaging over it.

---

## 5. Scoring

> **Both scoring criteria are undefined as of this writing.** This section records the
> constraints any implementation must respect, not a design that has been agreed. When you
> settle one, write it here.

### Shared contract

- Both `price_score` and `performance_score` must be on the **same, comparable scale** —
  otherwise the weighted average in `model.py` is meaningless. Normalize to `[0, 1]`, higher is
  better (so cheap → high price score), and normalize *within the candidate pool for this
  trajectory*, not globally.
- Both are functions of `(Trajectory, ModelLLM)`. Neither may look at the `model` field of the
  trajectory it is scoring, except where the router deliberately reasons about switching cost —
  see the cache trap below. Leaking the served model into the score is how you accidentally
  build a classifier that predicts the log instead of a router that improves on it.

### Price

- **Token counts are estimates.** There is no `usage` field. The convention here is chars/4
  unless someone wires in a real tokenizer. Every number derived from them is an estimate and
  must be labeled as one in the writeup. Never present estimated tokens as measured.
- **Pricing is an assumption.** The model ids are anonymized, so no public price sheet applies
  and the tier order is not published. If the organizers post prices, they win; otherwise state
  the assumption explicitly and keep it in one place so it can be swapped.
- **The cache trap — this is the crux of the price model.** Providers cache the shared input
  prefix across a task's calls. A model switch **resets that cache**, so every call after a
  switch pays full price for the entire accumulated prefix. In trajectories this long, that
  penalty can dwarf the per-token saving that motivated the switch. Since there is no
  `usage.cached_tokens`, the cached share has to be *inferred* from item-level prefix overlap. A
  `CHANGE TO` decision that ignores this will look brilliant and be wrong.

### Performance

- No quality labels exist in the export. Any performance score is a **proxy** — derive it from
  observable structure (task type inferred from the toolset, trajectory length, tool-call
  density, error and retry patterns in `function_call_output`, whether the trajectory reached
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

- **This README is the spec.** New stage, new module, changed contract — update the relevant
  section in the same commit. The next agent has no other design doc to read.
- **Pydantic models in `data_models/`, nowhere else.**
- **Enrich, don't replace.** Scoring stages set attributes on the `ModelLLM` objects they are
  handed and return the same list.
- **Never commit the dataset.** Challenge use only, no redistribution (`data/LICENSE`).
  `*.jsonl` and `*.jsonl.tar.gz` are gitignored — keep it that way, and never upload traces to
  any external service.
- **Label estimates as estimates.** Token counts, prices, and quality proxies are all inferred
  here. Carry that honesty into code comments, output columns, and slides.
- **Don't invent challenge facts** — deadlines, prizes, credits, price sheets. If it isn't in
  `code_agent_utils/AGENTS.md` or posted by the organizers, ask in the challenge Discord.

---

## 8. Open decisions

Live list — resolve, then fold the answer into the sections above.

1. Price scoring formula, and the assumed price sheet behind it.
2. Performance scoring formula, and which observable proxies feed it.
3. `Trajectory` field set — what `trajectory_analyzer.py` extracts and precomputes.
4. Routing granularity: the decision is framed as "the model for the next message", while the
   dataset premise is one model per trajectory. Decide whether the router may switch
   mid-trajectory — and if so, the cache penalty is not optional.
5. Default weights, and the sweep used to produce the frontier.

Resolved: `ModelLLM` field set (§2) and where the candidate pool comes from (§2,
`pre_processing/model_list.py`).
