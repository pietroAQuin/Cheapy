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
            data/*.jsonl (one JSON line = one complete trajectory)
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

- **`ModelLLM`** (`data_models/LLMModel.py`) — everything known about one model in the candidate
  pool, including the scores the router stages attach to it.
- **`Trajectory`** (`data_models/Trajectory.py`) — everything known about one trajectory: 19
  fields covering identity, toolset, per-call averages, and totals, plus `normalized_items`,
  the whole trajectory in an encoding-independent form (`NormalizedItem`, `ItemKind`) so a later
  stage can derive signals this field set does not precompute.

### The two pre-processing entry points

- **`pre_processing/model_list.py`** — where `ModelLLM` objects are first created. Returns the
  list of every model to be considered for routing a trajectory.
- **`pre_processing/trajectory_analyzer.py`** — where `Trajectory` objects are first created.
  `analyze(line, id)` takes one JSON line and returns one `Trajectory`; `analyze_file(path)`
  yields one per line. `normalize()` is the only function in the pipeline aware that the export
  ships two encodings — everything downstream reads `NormalizedItem`.

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
| `data_models/LLMModel.py` | `ModelLLM` — one candidate model | **empty stub** |
| `data_models/Trajectory.py` | `Trajectory` — one trajectory, plus `NormalizedItem` | **implemented** |
| `pre_processing/model_list.py` | Builds the candidate pool | **empty stub** |
| `pre_processing/trajectory_analyzer.py` | Parses one JSON line into a `Trajectory` | **implemented** |
| `router_models/price_model.py` | Assigns `price_score` | **empty stub** |
| `router_models/performance_model.py` | Assigns `performance_score` | **empty stub** |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision | **empty stub** |
| `data/` | The redacted export (gitignored, challenge-use only) | present locally |
| `code_agent_utils/` | Organizers' briefing + `/setup`, `/make-presentation`, `/prepare-submission` skills | supplied |

The files still marked **empty stub** are zero-byte placeholders — the structure is decided,
those implementations are not. `Trajectory` and `trajectory_analyzer` are written and
cross-checked against the export; treat their contracts as real.

---

## 4. The dataset

`data/trajectories_v1_01.jsonl` — 1,000 lines, ~105 MB, one **LLM request** per line in the
OpenAI-compatible Responses format. Exactly three top-level fields:

- **`model`** — the anonymized id that actually served the call.
- **`input`** — the full request history up to that call.
- **`tools`** — the function definitions available to that call.

There is **no `output` field and no `usage`**. No token counts, no latency, no quality labels,
no trajectory ids. The final assistant message is nonetheless present *inside* `input` — see
"Trajectory structure" below.

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
5,602 assistant / 1,981 user / 1,000 system), `reasoning` (1,792, gpt-family only — **1,011 of
them carry summary text**, median 453 chars; the earlier claim that summaries are empty was
wrong), `custom_tool_call` + `custom_tool_call_output` (299 each, e.g. `apply_patch`).

Every line is a distinct trajectory, so these are totals over the whole export, not
over-counts.

### Two encodings, and they give the model family away

The export ships the same information in two shapes:

| Encoding | Shape | Lines |
|---|---|---|
| typed / list | `{"type": "message", "role": …, "content": [{"type": "input_text", "text": …}]}` | 755 — **all `claude-*`** |
| untyped / string | `{"role": …, "content": "…"}`, no `type` key | 245 — **all `gpt-*`** |

772 items carry no `type` field at all. A parser that only understands the typed form returns
**zero tokens for every GPT line** — a silent under-count perfectly correlated with model
family. `trajectory_analyzer.normalize()` is the single place that resolves this; everything
downstream reads `NormalizedItem` and never sees the difference.

**This workload is overwhelmingly tool-calling, not chat:** roughly 1.2 `function_call` items
for every message item, and 5,602 of the 8,583 messages are assistant turns. Any performance
score that only looks at user-facing message text is scoring the wrong thing.

Toolsets cluster into recognizable Viktor surfaces — a Slack-flavored set
(`coworker_send_slack_message`, `coworker_upload_to_slack`, …, 929 requests), a Teams-flavored
set (`coworker_*_msteams_*`, 71 requests), and two coding dialects: `bash`/`file_read`/
`file_edit`/`file_write` (755) versus `shell_command`/`apply_patch` (245). Tool count per
request is bimodal at 12 (692) and 10 (224). Only two tools — `submit_draft` and `view_image` —
appear in every single request; there are 8 distinct toolsets in all.

**The platform half of that signal is free and safe to use.** Slack vs Teams is independent of
the served model (~7% Teams within both families), and is what `Trajectory.viktor_environment`
carries.

**The coding-dialect half is not a task signal at all — it is the model family restated.**
`bash`/`file_*` occurs on 755 lines, every one of them `claude-*`; `shell_command`/`apply_patch`
on 245, every one `gpt-*`. 100% separation. It is a provider harness artifact, so a score that
reads it is reading `served_model` by proxy. See "Leakage" below.

### Redaction

Entities are replaced by stable named placeholders (`PII_PERSON_7`, `PII_COMPANY_1`,
`PII_URL_8`, `<ID_13>`), consistent *within* a trajectory so references still resolve. Images
are placeholder data URLs. Do not try to de-anonymize anything.

### Trajectory structure

**No reconstruction is needed. One line is one complete trajectory.** An earlier version of this
spec called for grouping requests by their opening messages and ordering them by input length;
that was wrong for this chunk, and the procedure is a no-op here. Measured three ways:

- grouping by (system prompt + first user text) yields **1,000 groups of size 1**;
- testing whether any line's item list is a strict **prefix** of another's finds **0 pairs**;
- repeating that test with redaction normalized away (structure-only item signatures, so
  reassigned `PII_*` numbering cannot hide a match) still finds **0 pairs**.

871 distinct system prompts across 1,000 lines, so some lines share a workspace — none continues
another. The export was subsampled to one request per trajectory. Parse it with
`for line in file`.

The per-call structure lives *inside* each line instead. Segmenting `input` into maximal runs of
model-generated items recovers the call sequence: **10,845 calls across the export**, median 5
per trajectory, max 151. This is what `Trajectory.total_calls` and the `avg_*_per_call` fields
are computed over, and what `NormalizedItem.call_index` stamps.

Two consequences worth internalizing:

1. **Every call's output is recoverable, including the last.** What the model returned on call
   *i* appears in the history as assistant / `function_call` items, and every line ends with the
   trajectory's closing assistant message. The catch is that this closing message is **empty on
   97.1% of `gpt-*` lines (238 of 245) and on 0% of `claude-*` lines**. So the final response
   text is a real quality signal for one family and absent for the other — using it raw is a
   family-biased score, not a quality score.
2. **One model serves all calls of a trajectory.** Each line carries exactly one `model`, so
   within this export the premise is not observable and needs no policing. The variation *across*
   trajectories is still the natural experiment the evaluation rests on.

### Leakage — five ways to accidentally read `served_model`

The scoring contract in §5 forbids reading the trajectory's `model`. These five features are that
field in disguise, each verified at or near 100% separation on the export:

| Feature | Separation |
|---|---|
| coding dialect: `bash`/`file_*` vs `shell_command`/`apply_patch` | 755 claude / 245 gpt, exact |
| item encoding: typed-list vs untyped-string | 755 claude / 245 gpt, exact |
| `custom_tool_call` present (`apply_patch`) | gpt only |
| final assistant message empty | 97.1% gpt / 0% claude |
| `reasoning` items present | gpt only |

`NormalizedItem` deliberately keeps all of this reachable — it is raw material a later stage may
need. Reaching for it in a *score* is the mistake.

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

- **Token counts are estimates.** There is no `usage` field. The pipeline counts with
  **`tiktoken`, `o200k_base`, applied uniformly to every line of both families**
  (`trajectory_analyzer.count_tokens`); chars/4 remains only as a fallback when the vocab file is
  absent, so a machine with no network degrades instead of failing. Every number derived from
  either is an estimate and must be labeled as one in the writeup. Never present estimated tokens
  as measured.
  - *Why not chars/4:* measured against the real tokenizer on this export, chars/4 understates
    Claude text by 4% and GPT text by 9% — a 5-point **family-correlated** error, because
    tool-call JSON tokenizes at ~3.2 chars/token against ~4.15 for prose and the GPT lines are
    more tool-heavy. On a workload this tool-dominated it is wrong exactly where it matters.
  - *Why one tokenizer for both:* the ids are anonymized and map to no public tokenizer
    (`encoding_for_model("claude-opus-5")` raises), and Claude's tokenizer is not available
    offline. A per-family tokenizer would make the two families' prices incomparable — which is
    the only comparison the router makes. Uniform is the honest choice; it is still an estimate.
  - Cost: ~16 s for the full export, offline after a one-time 3.6 MB vocab fetch cached in
    `.tokenizer_cache/` (gitignored).
- **Tool schemas count.** All ~12 schemas are re-sent on every call — a median **4,203** est.
  tokens, and they sit at the very front of the prefix, so they are also the most cacheable part.
  `total_tokens` includes them.
- **Pricing is an assumption.** The model ids are anonymized, so no public price sheet applies
  and the tier order is not published. If the organizers post prices, they win; otherwise state
  the assumption explicitly and keep it in one place so it can be swapped.
- **The cache trap — this is the crux of the price model.** Providers cache the shared input
  prefix across a task's calls. A model switch **resets that cache**, so every call after a
  switch pays full price for the entire accumulated prefix. In trajectories this long, that
  penalty can dwarf the per-token saving that motivated the switch. A `CHANGE TO` decision that
  ignores this will look brilliant and be wrong.
  - Measured: **92.1% of all billed input across the export is cacheable prefix.** Total billed
    input is 338 M est. tokens; per trajectory the median is 116 K and the maximum **11.7 M** —
    cost is *quadratic* in trajectory length, because every call re-sends the whole history.
    Input dominates output ~130:1, so price here is almost entirely an input-side story.
  - No overlap search is required, and pretending otherwise would overstate what we know. Within
    a line, call *i*'s prompt is by construction a prefix-extension of call *i−1*'s, so under a
    perfect cache the cached share of call *i* is exactly call *i−1*'s whole prompt, and the sum
    collapses to `total_tokens − prompt_tokens[last]`. `Trajectory.total_cached_tokens` is that
    identity times **`trajectory_analyzer.CACHE_HIT_RATE`** (default `1.0`, the optimistic
    bound). Real caches have TTLs and minimum block sizes and the export has no timestamps, so
    that constant is the single place the assumption lives — sweep it from `price_model.py`.

### Performance

- No quality labels exist in the export. Any performance score is a **proxy** — derive it from
  observable structure (platform inferred from the toolset, trajectory length, tool-call density,
  error and retry patterns in tool outputs), or from judge-model rescoring of individual calls.
  `Trajectory.normalized_items` is there so a proxy can be derived without re-parsing the export.
- **Check every candidate proxy against the leakage table in §4 first.** Coding dialect, item
  encoding, `custom_tool_call`, `reasoning`, and an empty final message are all `served_model`
  wearing a hat.
- **`submit_draft` is not a completion signal.** It is the human-approval gate: Viktor prepares a
  consequential action as a draft, a human approves it in thread state, and `submit_draft`
  executes it (the output names the real tool behind the gate, e.g. `mcp_google_ads_remove_ad`).
  It is offered in all 1,000 requests but invoked in only **14**, 37 calls total, and never as a
  terminator. Read it as "performed a gated, irreversible action" — a task-type signal at ~1.4%
  prevalence, not a success signal.
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
3. `ModelLLM` field set — beyond `price_score`, `performance_score`, and the final score.
4. ~~`Trajectory` field set~~ — **settled.** 19 fields; see `data_models/Trajectory.py`, which
   documents each one and its estimate/leak caveats inline.
5. Where the candidate pool comes from: hardcoded in `model_list.py`, or derived from the models
   observed in the export.
6. Routing granularity: the decision is framed as "the model for the next message", while each
   exported line is a *complete* trajectory served end-to-end by one model. Decide whether the
   router may switch mid-trajectory — and if so, the cache penalty is not optional, and §4's
   92.1% cacheable share sets its size.
7. Default weights, and the sweep used to produce the frontier.
8. Whether to vendor the 3.6 MB `o200k_base` vocab into the repo. Today it is fetched once into
   a gitignored `.tokenizer_cache/`, so a fresh clone needs network exactly once; vendoring would
   make it offline from the first run at the cost of a binary blob in git.
