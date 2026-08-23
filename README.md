# Cheapy — a cost/quality router for Viktor

**Viktor** is an autonomous LLM coworker that lives in Slack and Microsoft Teams. It runs long,
tool-heavy trajectories, and today a *single* model serves every call of a trajectory. Cheapy
ingests those trajectories and decides which model, out of a pool of nine, should serve the next
call — and whether switching is worth what a switch costs.

Built for the Viktor Challenge at the TUM.ai hackathon (Munich, 22–23 Aug 2026).

**This README is the architecture spec.** It is the only design document at repo root. When the
design changes, change it here, in the same commit as the code. Two deeper documents sit beside
their own code and are linked from the sections that summarise them:
`analysis/complexity_model/complexity_scoring_spec.md` (the capability model) and
`analysis/data_exploration/README.md` (the clustering notebook).
`code_agent_utils/AGENTS.md` is the organizers' briefing; where it contradicts this file on
dataset shape, licensing, or judging, it wins.

**Status: the pipeline is complete and runs end to end.** Every stage is implemented, the
capability model is fitted, and `./run_simul.sh` routes all 1,000 exported trajectories in ~20 s
with no network and no API keys.

---

## 1. What the router does

**Input:** a JSONL file of Viktor trajectories (one JSON line = one complete trajectory).
**Output:** for each trajectory, an ordered list of candidate models, best first, and a verdict:

| Decision | Meaning |
|---|---|
| `HOLD` | The top-scored model is the one already serving the trajectory — keep it. |
| `CHANGE TO <model>` | A different model scored highest — switch for the next call. |

The score behind that verdict is a **weighted average of a price score and a performance score**,
both normalized to `[0, 1]` within the candidate pool for that trajectory. The weights are call
arguments, never constants, so the router can be dialed from "cheapest acceptable" to "best
regardless of cost" — sweeping them traces a cost–quality frontier instead of committing to one
operating point. The frontier is the headline artifact, not any single routing decision.

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
        cache-aware next-call cost → ModelLLM.price_score
                                 │
                                 ▼
                 router_models/performance_model.py
        prior-weighted conformity → ModelLLM.performance_score
        (thin adapter onto analysis/complexity_model/)
                                 │
                                 ▼
                      router_models/model.py
        final_score = (w_price·price + w_perf·perf) / (w_price + w_perf)
                                 │
                                 ▼
        ordered List[ModelLLM]  →  HOLD | CHANGE TO <model>
                                 │
                                 ▼
                  app/main.py → data/routing_simul.csv
```

### The two core classes

Both live in `data_models/` as Pydantic `BaseModel`s. Keep them there and keep them Pydantic —
validation at the boundary is what makes a multi-stage pipeline debuggable. Import them; never
redefine these shapes inline.

- **`ModelLLM`** (`data_models/model_llm.py`) — everything known about one candidate model:
  `name`, `family`, `context_window_size`, the four per-1M-token price fields
  (`input_price_per_1m`, `cached_input_price_per_1m`, `output_price_per_1m`,
  `cached_output_price_per_1m`), and the three scores the router attaches — `price_score`,
  `performance_score`, `final_score`. All three are `None` until the matching stage sets them.
  **Never treat an unset score as `0`**: unset means "this stage did not score this model", and
  reading it as 0 silently ranks it worst on evidence that was never gathered.
- **`Trajectory`** (`data_models/Trajectory.py`) — one trajectory in **20 fields** covering
  identity, toolset, per-call averages, and totals, plus `normalized_items`: the whole trajectory
  in an encoding-independent form (`NormalizedItem`, `ItemKind`) so a later stage can derive
  signals this field set does not precompute. Each field documents its own estimate/leak caveats
  inline.

### The two pre-processing entry points

- **`pre_processing/model_list.py`** — where `ModelLLM` objects are created. `build_model_list()`
  returns the candidate pool: the 9 model ids observed in `data/trajectories_v1_01.jsonl`, each
  paired with assumed published pricing and a context-window figure. It is a static pool, not
  re-derived by scanning the export on every run — revisit if a later chunk introduces new ids.
- **`pre_processing/trajectory_analyzer.py`** — where `Trajectory` objects are created.
  `analyze(line, id)` takes one JSON line and returns one `Trajectory`; `analyze_file(path)`
  yields one per line. `normalize()` is the **only** function in the pipeline aware that the
  export ships two encodings — everything downstream reads `NormalizedItem`.

### The scoring chain

A `Trajectory` and its own model list are fed through `router_models/`, in order:

1. `price_model.score_price` sets `price_score`.
2. `performance_model.score_performance` sets `performance_score`.
3. `model.aggregate_scores` combines them; `model.rank` sorts; `model.decide` emits the verdict.
   `model.route` is the one-call wrapper that runs all four.

Each stage **enriches the same `ModelLLM` objects in place** rather than returning new types. A
scoring stage is therefore a function of `(Trajectory, list[ModelLLM]) -> list[ModelLLM]`, and
stages compose in the order above. Adding a third criterion means adding a stage with the same
contract and a term in `model.py`'s average.

Because stages mutate in place, **every trajectory needs its own pool** — a shared list would
carry the previous trajectory's scores. `app/main.py` rebuilds it per line.

---

## 3. Repository layout

| Path | Role |
|---|---|
| `data_models/model_llm.py` | `ModelLLM` — one candidate model |
| `data_models/Trajectory.py` | `Trajectory` (20 fields) + `NormalizedItem` / `ItemKind` |
| `data_models/quality_base_value.py` | `QUALITY_BASE_SCORE` — static `[0, 1]` quality prior per model id, from three public leaderboards (§5) |
| `pre_processing/model_list.py` | Builds the candidate pool with real pricing + context windows |
| `pre_processing/trajectory_analyzer.py` | Parses one JSON line into a `Trajectory`; owns `normalize()` and `count_tokens()` |
| `router_models/price_model.py` | `price_score` — cache-aware next-call cost, min-max normalized in-pool |
| `router_models/performance_model.py` | `performance_score` — thin adapter onto the capability model |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision, `RoutingDecision` |
| `analysis/complexity_model/` | The capability model: offline elicitation + fitting pipeline, plus the shipped `capability_model.py`. Design notes in `complexity_scoring_spec.md` (read its REVISION header first) |
| `analysis/complexity_model/artifacts/pair_model.json` | The fitted artifact. Loaded automatically; inference needs no network and no keys |
| `analysis/data_exploration/` | Colab notebook clustering toolsets and complexity bands — exploratory, not on the router's path (see its README) |
| `app/main.py` | Routing simulation over the whole export → `data/routing_simul.csv` |
| `run_simul.sh` | One-liner wrapper around `app/main.py` (prefers `.venv/bin/python`) |
| `tests/` | 138 unit tests over hand-built fixtures, never the export |
| `scripts/` | **Legacy.** First-pass loader and cost model from before `pre_processing/` existed; they use chars/4 tokens and the since-disproved trajectory-grouping assumption (§4). Kept for reference; nothing imports them |
| `data/` | The redacted export — gitignored, challenge use only (`data/LICENSE`) |
| `code_agent_utils/` | Organizers' briefing + `/setup`, `/make-presentation`, `/prepare-submission` skills |

### Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest        # 138 passed
./run_simul.sh                    # ~20 s over the 1,000-line chunk
```

`requirements-dev.txt` covers the router, the fitted capability model, and the tests
(`pytest`, `tiktoken`, `numpy`, `scikit-learn`, `jsonschema`, `tenacity`).
`requirements-elicit.txt` (`openai`, `anthropic`, `python-dotenv`) is needed **only** to re-run
the offline elicitation pilot — the shipped router never imports a provider SDK, which is what
keeps its "no API keys required" guarantee honest.

Tests build their own synthetic trajectory lines, in **both** encodings, in `tests/conftest.py`
rather than reading `data/` — the real export is gitignored and not guaranteed to be present.

---

## 4. The dataset

`data/trajectories_v1_01.jsonl` — 1,000 lines, ~105 MB, one **LLM request** per line in the
OpenAI-compatible Responses format. Exactly three top-level fields:

- **`model`** — the anonymized id that actually served the call.
- **`input`** — the full request history up to that call.
- **`tools`** — the function definitions available to that call.

There is **no `output` field and no `usage`**. No token counts, no latency, no quality labels, no
trajectory ids. The final assistant message is nonetheless present *inside* `input`.

### What is actually in this chunk (measured, not assumed)

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
them carry summary text**, median 453 chars), `custom_tool_call` + `custom_tool_call_output`
(299 each, e.g. `apply_patch`).

**This workload is overwhelmingly tool-calling, not chat:** roughly 1.2 `function_call` items per
message item, and 5,602 of 8,583 messages are assistant turns. Any performance score that only
looks at user-facing message text is scoring the wrong thing.

Toolsets cluster into recognizable Viktor surfaces — a Slack-flavored set
(`coworker_send_slack_message`, `coworker_upload_to_slack`, …, 929 requests), a Teams-flavored set
(`coworker_*_msteams_*`, 71 requests), and two coding dialects. Tool count per request is bimodal
at 12 (692) and 10 (224); only `submit_draft` and `view_image` appear in every request; there are
8 distinct toolsets. `analysis/data_exploration/` works this up into 8 tool families and 5
complexity bands, and finds toolset profile is fully determined by *environment × editing
toolchain × agent role* — which is also why toolset and model effects are not separable here.

**The platform half of that signal is free and safe to use.** Slack vs Teams is independent of
the served model (~7% Teams within both families) and is what `Trajectory.viktor_environment`
carries. **The coding-dialect half is not a task signal at all — it is the model family
restated** (see Leakage below).

### Two encodings, and they give the model family away

| Encoding | Shape | Lines |
|---|---|---|
| typed / list | `{"type": "message", "role": …, "content": [{"type": "input_text", "text": …}]}` | 755 — **all `claude-*`** |
| untyped / string | `{"role": …, "content": "…"}`, no `type` key | 245 — **all `gpt-*`** |

772 items carry no `type` field at all. A parser that only understands the typed form returns
**zero tokens for every GPT line** — a silent under-count perfectly correlated with model family.
`trajectory_analyzer.normalize()` is the single place that resolves this.

### Redaction

Entities are replaced by stable named placeholders (`PII_PERSON_7`, `PII_COMPANY_1`, `PII_URL_8`,
`<ID_13>`), consistent *within* a trajectory so references still resolve. Images are placeholder
data URLs. Do not try to de-anonymize anything.

### Trajectory structure

**No reconstruction is needed. One line is one complete trajectory.** An earlier version of this
spec called for grouping requests by their opening messages and ordering them by input length;
that is a no-op on this chunk, and `scripts/load_trajectories.py` is the code that still assumes
it. Measured three ways:

- grouping by (system prompt + first user text) yields **1,000 groups of size 1**;
- testing whether any line's item list is a strict **prefix** of another's finds **0 pairs**;
- repeating that with redaction normalized away (structure-only signatures, so reassigned `PII_*`
  numbering cannot hide a match) still finds **0 pairs**.

871 distinct system prompts across 1,000 lines, so some lines share a workspace — none continues
another. The export was subsampled to one request per trajectory. Parse it with `for line in file`.

The per-call structure lives *inside* each line. Segmenting `input` into maximal runs of
model-generated items recovers the call sequence: **10,845 calls across the export**, median 5 per
trajectory, max 151. That is what `Trajectory.total_calls` and the `avg_*_per_call` fields are
computed over, and what `NormalizedItem.call_index` stamps.

Two consequences worth internalizing:

1. **Every call's output is recoverable, including the last** — which is what makes the capability
   model's corpus recovery possible (§6). The catch: the closing assistant message is **empty on
   97.1% of `gpt-*` lines (238 of 245) and on 0% of `claude-*` lines**, so final response text is
   a family-biased score, not a quality score.
2. **One model serves all calls of a trajectory.** Each line carries exactly one `model`, so the
   premise is not observable within a line and needs no policing. The variation *across*
   trajectories is the natural experiment the evaluation rests on.

### Leakage — five ways to accidentally read `served_model`

The scoring contract in §5 forbids reading the trajectory's `model`. These five features are that
field in disguise, each verified at or near 100% separation:

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

### Shared contract

- Both scores are on the **same scale**: `[0, 1]`, higher is better (so cheap → high price score),
  normalized *within the candidate pool for this trajectory*, not globally. Otherwise the weighted
  average in `model.py` is meaningless.
- Both are functions of `(Trajectory, list[ModelLLM])`. Neither may read the trajectory's
  `served_model`, **except** the one carve-out below where the router deliberately reasons about
  switching cost. Leaking the served model into a score is how you accidentally build a classifier
  that predicts the log instead of a router that improves on it.

### Price

**Token counts are estimates.** There is no `usage` field. The pipeline counts with **`tiktoken`,
`o200k_base`, applied uniformly to every line of both families** (`trajectory_analyzer.count_tokens`);
chars/4 remains only as a fallback when the vocab file is absent, so a machine with no network
degrades instead of failing. Every derived number is an estimate and must be labeled as one.

- *Why not chars/4:* measured against the real tokenizer on this export, chars/4 understates
  Claude text by 4% and GPT text by 9% — a 5-point **family-correlated** error, because tool-call
  JSON tokenizes at ~3.2 chars/token against ~4.15 for prose and the GPT lines are more
  tool-heavy. On a workload this tool-dominated it is wrong exactly where it matters.
- *Why one tokenizer for both:* the ids are anonymized and map to no public tokenizer
  (`encoding_for_model("claude-opus-5")` raises), and Claude's tokenizer is not available offline.
  A per-family tokenizer would make the two families' prices incomparable — the only comparison the
  router makes. Uniform is the honest choice; it is still an estimate.
- Cost: ~16 s for the full export, offline after a one-time 3.6 MB vocab fetch cached in
  `.tokenizer_cache/` (gitignored).

**Tool schemas count.** All ~12 schemas are re-sent on every call — a median **4,203** est. tokens
— and they sit at the very front of the prefix, so they are also the most cacheable part.
`total_tokens` includes them.

**Pricing is an assumption.** The ids are anonymized, so no public price sheet applies. The
assumed sheet lives in one place, `pre_processing/model_list.py`, so it can be swapped wholesale
if the organizers post real prices. `cached_output_price_per_1m` is `0.0` everywhere — not a
placeholder; neither pricing source discounts a freshly generated output token.

**The cache trap — the crux of the price model.** Providers cache the shared input prefix across a
trajectory's calls. A model switch **resets that cache**, so every call after a switch pays full
price for the entire accumulated prefix. In trajectories this long that penalty can dwarf the
per-token saving that motivated the switch. A `CHANGE TO` that ignores this looks brilliant and is
wrong.

- Measured: **92.1% of all billed input across the export is cacheable prefix.** Total billed input
  is 338 M est. tokens; per trajectory the median is 116 K and the maximum **11.7 M** — cost is
  *quadratic* in trajectory length, because every call re-sends the whole history. Input dominates
  output ~130:1, so price here is almost entirely an input-side story.
- No overlap search is required. Within a line, call *i*'s prompt is by construction a
  prefix-extension of call *i−1*'s, so under a perfect cache the cached share of call *i* is
  exactly call *i−1*'s whole prompt, and the sum collapses to `total_tokens − prompt_tokens[last]`.
  `Trajectory.total_cached_tokens` is that identity times **`trajectory_analyzer.CACHE_HIT_RATE`**
  (default `1.0`, the optimistic bound). Real caches have TTLs and minimum block sizes and the
  export has no timestamps, so that constant is the single place the assumption lives.

#### The formula (`router_models/price_model.py`)

`score_price(trajectory, models)` estimates the USD cost of each candidate serving the
trajectory's **next call**, then min-max normalizes those costs into `price_score ∈ [0, 1]` across
the pool (cheapest → `1.0`, priciest → `0.0`).

1. **Shape the next call, from `Trajectory` fields alone:**
   - `last_call_input_tokens` — the size of the most recently recovered call's full prompt (tool
     schemas + history to that point). Since every call resends the whole history, this is the best
     proxy for the next call's prompt size. **Not `total_tokens`**, which sums every past call's
     own full-history snapshot and so overstates the next call by roughly `n/2` on an `n`-call
     trajectory.
   - `already_cached_tokens = min(total_cached_tokens, last_call_input_tokens)`.
     `total_cached_tokens` is a whole-trajectory aggregate and can exceed any single call's prompt;
     only that much of it is part of *this* call's bill, hence the cap.
   - `increment_tokens = last_call_input_tokens / total_calls` — an ESTIMATE of how much new
     content (tool outputs, replies, …) a typical call in this trajectory appends. Not
     `total_tokens / total_calls`, which repeats the superseded-snapshots mistake and settles near
     half of `last_call_input_tokens` regardless of trajectory length.
   - `next_output_tokens = avg_output_tokens_per_call` (already excludes the unobserved final
     output, per `Trajectory`'s field contract).
2. **No candidate is excluded for a small context window.** If `last_call_input_tokens` exceeds a
   candidate's `context_window_size`, the formula assumes the context is **compacted** down to that
   capacity: the value is clamped and everything downstream runs through the *same* formula on the
   clamped value. This replaced a hard feasibility gate that returned `price_score = 0.0` and could
   silently drop half the pool on long trajectories. The clamp keeps every candidate priced on the
   same terms — a small-window model still gets its normal cache treatment, just on a capped input.
3. **Apply the cache trap:**
   - **HOLD** (`model.name == trajectory.served_model`) — this is the `served_model` carve-out:
     `already_cached_tokens` bills at `cached_input_price_per_1m`; the rest,
     `(last_call_input_tokens − already_cached_tokens) + increment_tokens`, bills at
     `input_price_per_1m`.
   - **CHANGE** (any other candidate) — the switch resets the cache, so the *entire*
     `last_call_input_tokens + increment_tokens` bills at `input_price_per_1m`.
   - Either way `next_output_tokens` bills at `output_price_per_1m`.
   - `cost = (uncached·input + cached·cached_input + output·output) / 1e6`.
4. **Normalize:** `price_score = (max_cost − cost) / (max_cost − min_cost)`; all `1.0` on a total
   tie.

This reproduces the cache trap directly: a candidate with rates identical to `served_model` scores
strictly lower than holding, because a switch loses the discount on `already_cached_tokens`; a much
cheaper candidate can still outscore holding if its base rate absorbs the reset penalty. With
`total_cached_tokens = 0` holding and switching to an identically-priced model cost the same —
there is no cache to lose. Covered by `tests/test_price_model.py` (full/partial/no cache,
switching, the aggregate cap, the window clamp, and a regression test pinning the
`last_call_input_tokens` fix), and run against the full export (§7).

**Known simplifications, stated rather than hidden:**
- `increment_tokens` is a same-trajectory average, not a prediction; front- or back-loaded growth
  is mis-estimated. Bounded — it affects only the increment term, not the base.
- The formula scores **one hypothetical next call**, not the rest of the trajectory. It matches
  §1's "model for the next message" framing but projects no compounding cache effects.
- Compaction is modeled as a size clamp only. Real summarization costs a call and loses
  information, so this favors small-context candidates more than reality would.
- `total_cached_tokens` is a whole-trajectory aggregate capped down, not re-derived as "the last
  call's own cache share" from first principles.

### Performance

Implemented as a **pairwise-agreement (conformity) model** — full treatment in §6, design notes in
`analysis/complexity_model/complexity_scoring_spec.md`. In one line: cut a trajectory mid-flight,
ask how much two models' *next actions* would differ, and score each model by the prior-weighted
share of the panel that agrees with it. On a step where everyone does the same thing all scores go
to 1.0, the capability gap closes, and the router decides on cost alone.

Constraints that shaped it, and still bind anything replacing it:

- **No quality labels exist in the export**, so any performance score is a **proxy**. Derive it
  from observable structure (platform, trajectory length, tool-call density, error/retry patterns)
  or from model rescoring — `Trajectory.normalized_items` exists so a proxy can be built without
  re-parsing the export.
- **Check every candidate proxy against the leakage table in §4 first.**
- **`submit_draft` is not a completion signal.** It is the human-approval gate: Viktor prepares a
  consequential action as a draft, a human approves it, and `submit_draft` executes it (the output
  names the real tool behind the gate, e.g. `mcp_google_ads_remove_ad`). Offered in all 1,000
  requests, invoked in **14**, 37 calls total, never as a terminator. Read it as "performed a
  gated, irreversible action" — a task-type signal at ~1.4% prevalence, not a success signal.
- The core pipeline must stay runnable **offline on a laptop, no GPU, no API keys.** It is.

**`data_models/quality_base_value.py`** holds `QUALITY_BASE_SCORE`, a static `[0, 1]` quality prior
per model id, derived from three independent public leaderboards (Artificial Analysis Intelligence
Index, Agent Arena, Text Arena/LMArena). Each is min-max normalized within this project's 9-model
pool first — so three wildly different native scales contribute equally instead of the widest one
dominating — then averaged with **equal weight**. `1.0` is the ceiling a model would reach by
topping all three at once, not a value forced onto this pool's leader; no model here reaches it.
It is the leaderboard-sourced prior the capability model's `BASE_CAPABILITY` (§6) mirrors, and the
tie-breaker/floor a simpler proxy would fall back on — not a substitute for a per-trajectory score.

### Aggregation (`router_models/model.py`)

`route(trajectory, models, *, w_price, w_performance, beta=3.0, min_gain=0.0)` runs the whole chain
for one trajectory and returns a `RoutingDecision`.

- `final_score` is the weighted average **divided by the weight sum**, so any pair of non-negative
  weights keeps it in `[0, 1]` — `0.7/0.3` and `7/3` are the same operating point.
- **Weights are arguments, never module constants.** Sweeping them produces the frontier.
- A model missing either sub-score is **left out of the ranking**, not scored as 0, and reported in
  `RoutingDecision.unscored`.
- **Exact ties resolve to the incumbent** — switching has costs the score does not fully capture,
  so a dead-even score is not a reason to switch.
- **`min_gain`** is the `final_score` improvement a challenger must show before the switch is
  taken: a deadband for suppressing switches whose predicted gain sits inside the noise of an
  estimate built on estimated tokens and predicted agreement.
- `RoutingDecision.score_gap` doubles as the "how much does this decision matter here" signal — a
  gap near zero means any candidate would do.

`served_model` is read exactly once here, in `decide`, to compare the winner against the incumbent.
That is the decision itself, not a scoring feature.

---

## 6. The capability model (how `performance_score` is computed)

Implemented in `analysis/complexity_model/`, exposed through `router_models/performance_model.py`.
Design notes: `analysis/complexity_model/complexity_scoring_spec.md` — **read its REVISION header
first**, it records the OpenAI-only pivot that superseded several of the original sections.

### The idea, in plain terms

We have no answer key. Nothing in the dataset says "this was a good next move", so we can never
directly measure whether a model is *right*.

What we can do is **ask a panel and see who agrees with whom.** Picture a task paused halfway
through. Show the same half-finished work to nine models and ask each: *what would you do next?*

- If **everybody proposes the same next move**, the step is easy. No model has an edge, all score
  near 1.0, and the router should pick the cheapest.
- If **the answers scatter**, the step is genuinely hard. Now it matters who you side with, and we
  reward agreeing with the models we have reason to trust.

**A model's score is the share of the panel that agrees with it, weighted by how much we trust each
panel member.** The spread between best and worst is a by-product the router uses: narrow means
"anyone can handle this", wide means "this one is worth paying for".

### How it actually works

**Done once, offline (~$152, 957 steps):**

1. **Cut 957 real trajectories mid-task.** Everything before the cut is the "prefix".
2. **Ask the three probe models what to do next.** Only OpenAI credits are available, so
   `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` are the only queryable models.
3. **Get the other six models' answers for free.** Every line records which model served it *and*
   its full action log, so at any interior cut the served model's real next action is already in
   the data. No API call needed — this is what took pair coverage from 3/36 to **21/36 at zero
   extra API cost**.
4. **Compare every pair of answers.** Same action type and same tools → 1.0; different action types
   → a fixed table value; same type, partly overlapping tools → scored by set overlap.
5. **Learn the pattern.** Fit a ridge/logit model predicting agreement from cheap prefix features
   (calls so far, tool mix, whether the cut lands right after a user message, signs of recent
   errors) plus a descriptor of the two models. Saved to
   `analysis/complexity_model/artifacts/pair_model.json`.

**At runtime, per trajectory — no API calls, milliseconds:** predict each pair's agreement from the
prefix, Nyström-complete the unobservable Claude×Claude block, and combine:

```
score(i) = ( w_i  +  Σ_{j≠i} w_j · agreement(i, j) )  /  Σ_all w_j       where   w_i = prior_i ** β
```

The `w_i` in the numerator is the model agreeing with itself: it keeps the result inside `[0, 1]`
and stops a model being penalised for its own vote.

### Quality metrics — the honest numbers

From the shipped artifact, measured with **5-fold `GroupKFold` grouped by step**, so no step
appears in both halves.

| | |
|---|---|
| Trajectory cuts used | **947** (of 957 sampled; 10 lost to content-policy refusals) |
| Pairwise comparisons (rows) | **10,882** |
| Model MSE | **0.16893** |
| Baseline MSE (always predict the mean) | 0.17135 |
| **R²** | **+0.0141 (1.4%)** |
| Beats baseline | ✅ yes |

**Read that R² honestly: it is small.** The model explains ~1.4% of the variation in pairwise
agreement — a real, reproducible improvement over guessing the average, and nothing in this repo
should be presented as more.

The main reason is the shape of the target:

| target value | share of rows |
|---|---|
| exactly 1.0 (full agreement) | **69.2%** |
| exactly 0.0 (total disagreement) | **20.9%** |
| anything in between | 9.9% |

Mean 0.7358, sd 0.4139. Ninety percent of the target is a coin-flip-like bit — "same action or
not" — much of it genuinely unpredictable from the prefix alone. Roughly 53% of the variance sits
*between* steps and is in principle reachable, so most of the available signal is **not** being
captured by the current prefix features. That is the clearest place to improve.

**Held-out generalisation (leave-one-probe-out)** — fit on two probes, predict the third:

| held-out probe | test MSE | baseline MSE | R² |
|---|---|---|---|
| `gpt-5.6-sol` | 0.16110 | 0.16229 | **+0.007** |
| `gpt-5.6-terra` | 0.16036 | 0.16621 | **+0.035** |
| `gpt-5.6-luna` | 0.16770 | 0.17237 | **+0.027** |

Small, but positive in every case — it is learning something transferable, not memorising probes.

**Measurement noise floor.** None of the probes accepts `temperature = 0`, so asking the same model
the same question twice does not give the same answer. Measured self-agreement: `gpt-5.6-sol`
**0.860**, `gpt-5.6-terra` **0.773**, `gpt-5.6-luna` **0.890** (50-step control; an earlier 25-step
pilot put it at 0.74 and is superseded). **This is the ceiling on every agreement number here.** A
pair scoring 0.85 is not "85% similar" — it is at roughly the level two runs of the *same* model
reach.

**Pair coverage.** 21 of 36 unordered pairs measured (42 of 72 ordered cells) over 10,882
observations, but unevenly: the best-covered cell has **1,168** observations, the worst has **1**.
The remaining 15 (Claude×Claude) are Nyström-completed, never observed, and capped at rank 3 by
having only three probes. `ModelScore.measured_fraction` reports the split: **1.0 for the three
probes, 3/8 for the six Anthropic candidates.**

**One calibration check failed, and it is not silently ignored.** The logged-vs-elicited offset δ —
disagreement caused by *how* an action was collected rather than by model identity — was estimated
three times independently: `gpt-5.6-terra` −0.0947, `gpt-5.6-luna` −0.0175, `gpt-5.6-sol` +0.0298.
Spread **0.1245**, above the 0.10 consistency threshold, so `is_consistent = False` and **no
correction is applied.** The estimates disagree on sign; correcting would inject noise dressed as a
fix. Reported, not solved.

### Design decisions worth knowing

**Why not just use published benchmark scores for the six models we cannot query?** It was tried,
and it provably collapses. Substituting a prior-distance function into the score formula makes the
step-dependent term factor out entirely, producing an **identical ranking on all 947 steps** — one
that ranks models by how *central* their benchmark score is, penalising the strongest model for
being an outlier. Corpus recovery replaced it.

**Why β = 3 and not 1?** β controls how much a model's prior weights its vote. The design has two
intents, and at β = 1 only one held. Measured over 200 trajectories:

| β | ρ(score ranking, prior) on contested steps | top model on contested steps |
|---|---|---|
| 0 | 0.20 | `gpt-5.6-sol` |
| 1 | 0.52 | `gpt-5.6-sol` |
| 2 | 0.70 | `gpt-5.6-sol` |
| **3** | **0.73** | **`claude-fable-5`** |
| 5 | 0.75 | `claude-fable-5` |

At β = 1 the model rewarded for conformity was `gpt-5.6-sol` — a **mid-prior** model (0.7274 vs
`claude-fable-5`'s 0.9559) that wins by sitting at the centre of an all-GPT panel. β = 3 is the
lowest value at which the strongest model tops the ranking when it matters, without flattening the
score into the prior. Both intended behaviours hold there:

| | spread (best − worst) | lowest score |
|---|---|---|
| settled steps (top quartile by agreement) | 0.088 | 0.795 |
| contested steps (bottom quartile) | 0.168 | — |

Agreement compresses the field; disagreement fans it out by capability. That ~2× ratio is the
signal the router consumes. β is applied *after* the regressor, so changing it costs neither a
retrain nor an API call — pass `--beta` to `app/main.py`.

**Why the parsing rule matters more than it sounds.** Two encoding quirks each manufacture a
roughly **5× fake gap between providers** if parsed naively — Claude puts preamble text in
assistant messages next to tool calls; GPT logs stop-turns as empty messages. Either one is
`served_model` in disguise, which would turn the model into a provider classifier.
`parser.classify_run` is the single shared rule applied to both the elicited and the logged path,
neutralising both.

**The score never reads `served_model`** — not as a feature, not through any of §4's five proxies.
It appears inside `analysis/complexity_model/` only as an explicitly logged diagnostic.

### Known limitations

- **R² is 1.4%.** Real, reproducible, and small. The per-pair prediction is weak; the router
  consumes the *aggregate* spread across nine models, which averages many pair predictions and is
  steadier than any one — but this is never a strong model.
- **The 15 Claude-vs-Claude pairs can never be validated.** No trajectory is served by two models,
  so no data exists that could check them. Flagged `extrapolated`.
- **The whole panel is GPT.** GPT-vs-GPT agreement (0.762) runs higher than Claude-vs-GPT (0.694),
  so conformity scoring structurally favours GPT. β = 3 compensates; it does not remove the cause.
- **`claude-opus-4-6` and `claude-sonnet-4-6` served 2 and 1 trajectories.** Their scores rest on
  roughly one task each; they carry a `near_unmeasured` flag and stay out of headline claims.
- **δ is unresolved**, so no logged-vs-elicited correction is applied.
- **Changing `BASE_CAPABILITY` requires a retrain.** The priors feed both the runtime weights *and*
  five of the fitted model's features, so editing them desynchronises the saved artifact. β alone
  is free to change.

---

## 7. Results — the simulation over the full export

```bash
./run_simul.sh                                     # whole export, 50/50 weights
./run_simul.sh --w-price 0.15 --w-performance 0.85 # one point on the frontier
./run_simul.sh --limit 50 --min-gain 0.02 --beta 1.0
```

`app/main.py` parses every `data/*.jsonl` line, builds a **fresh** candidate pool per trajectory,
routes it, writes one row per trajectory to `data/routing_simul.csv`, and prints the HOLD /
CHANGE TO split. ~20 s for the 1,000-line chunk on a laptop, no network.

Every candidate is scored on every trajectory — the `unscored` column is empty throughout, so all
nine models compete on all 1,000 lines.

**Weight sweep** (`min_gain = 0`, `β = 3`, all 1,000 trajectories, measured):

| `w_price` | `w_perf` | HOLD | CHANGE | Where the changes go |
|---|---|---|---|---|
| 0.00 | 1.00 | 137 (13.7%) | 863 | `gpt-5.6-sol` 530, `claude-fable-5` 333 |
| 0.25 | 0.75 | **449 (44.9%)** | 551 | `gpt-5.6-luna` 551 |
| 0.50 | 0.50 | 40 (4.0%) | 960 | `gpt-5.6-luna` 960 |
| 0.75 | 0.25 | 20 (2.0%) | 980 | `gpt-5.6-luna` 980 |
| 1.00 | 0.00 | 20 (2.0%) | 980 | `gpt-5.6-luna` 980 |

Three things to read out of this, and one caution:

1. **The two ends behave as designed.** Pure performance routes to the two highest-prior models
   (`claude-fable-5`, prior 0.9559) or the panel-central one (`gpt-5.6-sol`); pure price routes
   almost everything to `gpt-5.6-luna`, which at $0.20/1M input is 50× cheaper than `claude-fable-5` and 25× cheaper than `claude-opus-5`.
2. **HOLD share is not monotonic in `w_price`** — it peaks at ~45% around `w_price = 0.25`. That is
   the crossover band where the incumbent's cache advantage and its capability standing balance
   against `luna`'s price advantage; outside it, one term dominates and sweeps the pool. The
   interesting operating points are in that band, not at the ends.
3. **Price saturates early.** 0.75 and 1.00 give identical splits: once price outweighs
   performance, the cheapest model wins every pool and further weight changes nothing.

**Caution: 96% "switch to the cheapest model" is not a result to present as a recommendation.**
It is what a 50/50 average of a strong price signal and a weak (R² = 1.4%) capability signal
mechanically produces. The price score separates candidates by orders of magnitude; the performance
score separates them by ~0.17 on contested steps and less elsewhere. Any honest reading of the
frontier says so.

---

## 8. Evaluating the router (the hard part, still open)

The log shows only the model that actually ran. Estimating what a *different* route would have cost
or delivered is **off-policy evaluation**, and it is the depth of this challenge — matching or
weighting across comparable trajectories, judge-model rescoring of individual calls. The simulation
in §7 reports *decisions*, not realized savings or realized quality; turning it into a cost–quality
frontier with axes in dollars and quality requires that counterfactual estimate.

Naming where the estimate can fail is **scored, not penalized**. The deliverable is a frontier, not
a point, and a five-minute presentation with one chart, one claim, and one known weakness.

---

## 9. Conventions for agents working here

- **This README is the spec.** New stage, new module, changed contract — update the relevant
  section in the same commit. The next agent has no other design doc at root.
- **Pydantic models in `data_models/`, nowhere else.**
- **Enrich, don't replace.** Scoring stages set attributes on the `ModelLLM` objects they are handed
  and return the same list. Build a fresh pool per trajectory.
- **An unset score is not zero.** Leave it `None` and report it as unscored.
- **Never commit the dataset.** Challenge use only, no redistribution (`data/LICENSE`). `*.jsonl`
  and `*.jsonl.tar.gz` are gitignored — keep it that way, and never upload traces to any external
  service.
- **Label estimates as estimates.** Token counts, prices, and quality proxies are all inferred here.
  Carry that into code comments, output columns, and slides.
- **Don't invent challenge facts** — deadlines, prizes, credits, price sheets. If it isn't in
  `code_agent_utils/AGENTS.md` or posted by the organizers, ask in the challenge Discord.

---

## 10. Open decisions

Live list — resolve, then fold the answer into the sections above.

1. **Off-policy evaluation (§8).** The largest open item: no counterfactual cost or quality estimate
   exists yet, so §7 reports decisions rather than savings.
2. **Better prefix features for the capability model.** ~53% of the target variance is between-step
   and in principle reachable; the current features capture 1.4%.
3. **Routing granularity.** The decision is framed as "the model for the next message", while each
   exported line is a complete trajectory served end-to-end by one model, so the simulation makes
   exactly one decision per trajectory. Decide whether the router may switch *mid*-trajectory — if
   so, the cache penalty compounds across the remaining calls and §5's one-call formula needs to
   project forward.
4. **Default weights.** 50/50 is a placeholder, and §7 shows it sits outside the interesting
   crossover band.
5. **Resolve δ** (§6) or state permanently that no logged-vs-elicited correction will be applied.
6. **Whether to vendor the 3.6 MB `o200k_base` vocab.** Today it is fetched once into a gitignored
   `.tokenizer_cache/`, so a fresh clone needs network exactly once; vendoring makes it offline from
   the first run at the cost of a binary blob in git.

**Resolved:** `ModelLLM` and `Trajectory` field sets and the candidate pool (§2); the price scoring
formula and its assumed price sheet (§5); the performance scoring formula and its fitted artifact
(§6); end-to-end wiring and the full-export run (§7).
