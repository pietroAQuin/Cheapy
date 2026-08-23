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

- **`ModelLLM`** (`data_models/model_llm.py`) — everything known about one model in the candidate
  pool, including the scores the router stages attach to it. Current fields: `name`, `family`,
  `performance_score`, `price_score`, `final_score` (all `None` until the matching stage in the
  scoring chain above sets them — never treat an unset score as `0`), `context_window_size`, and
  the four per-1M-token price fields (`input_price_per_1m`, `cached_input_price_per_1m`,
  `output_price_per_1m`, `cached_output_price_per_1m`).
- **`Trajectory`** (`data_models/Trajectory.py`) — everything known about one trajectory: 19
  fields covering identity, toolset, per-call averages, and totals, plus `normalized_items`,
  the whole trajectory in an encoding-independent form (`NormalizedItem`, `ItemKind`) so a later
  stage can derive signals this field set does not precompute.

### The two pre-processing entry points

- **`pre_processing/model_list.py`** — where `ModelLLM` objects are first created. Returns the
  list of every model to be considered for routing a trajectory. Currently a static pool: the 9
  ids/families observed in one scan of `data/trajectories_v1_01.jsonl`, paired with their real
  published pricing and context-window figures — not derived by scanning the export on every
  run. Revisit if a later chunk introduces new ids.
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
| `data_models/model_llm.py` | `ModelLLM` — one candidate model | **implemented** |
| `data_models/Trajectory.py` | `Trajectory` — one trajectory, plus `NormalizedItem` | **implemented** |
| `data_models/quality_base_value.py` | Static `[0, 1]` quality prior per model id, from public leaderboards | **implemented** — see §5 "Performance" |
| `pre_processing/model_list.py` | Builds the candidate pool | **implemented** (static pool + real pricing; no scoring — that's `router_models/`'s job) |
| `pre_processing/trajectory_analyzer.py` | Parses one JSON line into a `Trajectory` | **implemented** |
| `router_models/price_model.py` | Assigns `price_score` | **empty stub** |
| `router_models/performance_model.py` | Assigns `performance_score` | **implemented** — thin adapter onto `analysis/complexity_model/` |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision | **implemented** — weights are call arguments, never constants |
| `app/main.py` | Routing simulation over the whole export -> `data/routing_simul.csv` | **implemented** |
| `run_simul.sh` | One-liner wrapper around `app/main.py` (picks `.venv/bin/python`) | **implemented** |
| `analysis/complexity_model/` | Offline pairwise-agreement pipeline + the shipped `capability_model.py` — see its `complexity_scoring_spec.md` | **implemented and fitted** on a 947-step run over the export |
| `router_models/price_model.py` | Assigns `price_score` | **implemented** — see §5 |
| `router_models/performance_model.py` | Assigns `performance_score` | **empty stub** |
| `router_models/model.py` | Weighted aggregation, ranking, HOLD/CHANGE decision | **empty stub** |
| `tests/test_price_model.py` | Unit tests for the price scoring formula (`unittest`, stdlib only) | **implemented** |
| `data/` | The redacted export (gitignored, challenge-use only) | present locally |
| `code_agent_utils/` | Organizers' briefing + `/setup`, `/make-presentation`, `/prepare-submission` skills | supplied |
| `tests/` | Unit tests for the implemented modules, against hand-built fixtures (not the export) | covers `data_models/`, `pre_processing/`, `analysis/complexity_model/` |

`data_models/model_llm.py`, `data_models/Trajectory.py`, `pre_processing/model_list.py`, and
`pre_processing/trajectory_analyzer.py` are implemented and cross-checked against the export —
treat their contracts as real. `router_models/price_model.py` is a **zero-byte placeholder on
this branch** — the implementation lives on `main`, and `app/main.py` cannot run until that is
merged in; everything downstream is written against its
`score_price(trajectory, models) -> list[ModelLLM]` contract.
`router_models/performance_model.py` is implemented and backed
by a fitted model: the pipeline in `analysis/complexity_model/` was run end-to-end over **947 of
the 957 eligible trajectories** (10 dropped to content-policy refusals), eliciting the 3 OpenAI
probes on each and recovering the served model's logged action, for **10,882 measured ordered
pairs covering 42 of 72 ordered cells**. `artifacts/pair_model.json` is the fitted result and
`capability_model.py` loads it automatically; inference needs no network and no keys.

**Read the honest scorecard before quoting any number from it** — see the REVISION header of
`analysis/complexity_model/complexity_scoring_spec.md`. In short: the model beats the
constant-predictor baseline on the main gate and on all three leave-one-probe-out splits, but
only weakly (per-row R² ≈ 0.014; step-level Spearman ≈ 0.20; the steps it calls most contested
average 0.658 agreement against 0.846 for the least). Roughly 53% of the target variance sits
between steps and is in principle reachable, so most of the available signal is *not* being
captured by the current prefix features.
treat their contracts as real. `router_models/price_model.py` is implemented and sanity-checked
against synthetic trajectories (see §5) but not yet run against the real export. Every other file
under `router_models/` is still a **zero-byte placeholder**: the structure is decided, those
implementations are not. Nothing yet wires `router_models/model.py`'s aggregation into a runnable
end-to-end pipeline — you are not breaking an existing contract, you are writing the first one.

Run the suite with `pip install -r requirements-dev.txt && python -m pytest`. Tests build their
own synthetic trajectory lines (both encodings) in `tests/conftest.py` rather than reading
`data/`, since the real export is gitignored and not guaranteed to be present.

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

> **`price_score` is still undefined.** `performance_score` is now settled and implemented —
> see `analysis/complexity_model/complexity_scoring_spec.md` for the full design and the
> "### Performance" note below for the summary. This section otherwise records constraints
> any implementation must respect.
> **Performance scoring is undefined as of this writing.** Price scoring is implemented — see
> "Price scoring formula" below. This section records the constraints any implementation must
> respect, not a design that has been agreed for the parts still open. When you settle one, write
> it here.

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

#### Price scoring formula (implemented, `router_models/price_model.py`)

`score_price(trajectory, models)` estimates the USD cost of each candidate serving the
trajectory's **next call**, then min-max normalizes those costs into `price_score ∈ [0, 1]`
across the candidate pool for this trajectory (cheapest → `1.0`, priciest → `0.0`). It reads only
`Trajectory` fields plus one `ModelLLM` at a time — the only place it looks at
`trajectory.served_model` is the hold/switch cache check below, which is the exception the shared
contract above carves out.

1. **Shape the next call, from `Trajectory` fields alone:**
   - `last_call_input_tokens` — the size of the most recently recovered call's full prompt (tool
     schemas + history up to that point). Since every call resends the whole history, this is the
     best available proxy for the size of the *next* call's prompt.
     - **This replaced an earlier, incorrect base quantity, `total_tokens`.** `total_tokens` is a
       *sum* across every past call's prompt — each of which was itself a full-history resend at
       the time — so it adds up several superseded snapshots of a growing thing instead of
       reporting the size of the latest one. On a trajectory whose prompt grows roughly linearly
       over `n` calls, `total_tokens` overstates the real next-call size by very roughly a factor
       of `n/2` — see "Known simplifications" below for the worked example this used to get wrong.
       `last_call_input_tokens` is one call's real, measured size, not an accumulating sum, so it
       doesn't have that problem.
   - `already_cached_tokens = total_cached_tokens`, capped at `last_call_input_tokens`.
     `total_cached_tokens` is documented (per its own field docstring) as a share of
     `total_tokens` — a whole-trajectory cache aggregate, not specifically the last call's cache
     share — so it can exceed `last_call_input_tokens` on a long trajectory. Only that much of it
     is actually part of *this* call's bill, hence the cap. (This cap was present even in the
     `total_tokens`-based formula, where it was effectively vacuous — `total_cached_tokens` is
     always `≤ total_tokens` by construction. Against `last_call_input_tokens` the cap is now the
     binding, meaningful constraint in the common case.)
   - `increment_tokens = last_call_input_tokens / total_calls` — an ESTIMATE of how much *new*
     content (tool outputs, replies, ...) a typical call in this trajectory appends, taken as the
     trajectory's own average per-call growth. **Not `total_tokens / total_calls`** — that repeats
     the same superseded-snapshots mistake `last_call_input_tokens` exists to fix, just applied to
     the increment instead of the base: it averages in every past call's own (smaller) prompt size
     rather than measuring how much this trajectory actually grows per step, and settles around
     roughly half of `last_call_input_tokens` regardless of trajectory length instead of shrinking
     as trajectories get longer. The next call's actual new content is by definition not yet in the
     log, so dividing `last_call_input_tokens` by `total_calls` is the best available stand-in; it's
     a smaller, separate known simplification from the `last_call_input_tokens` fix above — see
     "Known simplifications".
   - `next_output_tokens = avg_output_tokens_per_call` (already excludes the trajectory's
     unobserved final output, per `Trajectory`'s own field contract).
2. **No candidate is excluded for having too small a context window.** If
   `last_call_input_tokens` is bigger than a candidate's `context_window_size`, the formula assumes
   the context gets **compacted** (summarized) down to that model's max capacity before the call is
   sent: `last_call_input_tokens` is clamped to `min(last_call_input_tokens, context_window_size)`,
   and everything downstream — the cache split, `increment_tokens`, `next_output_tokens` — runs
   through the *exact same* formula on that clamped value. This replaced an earlier hard
   feasibility gate that returned `None`/`price_score = 0.0` for any candidate whose window
   couldn't hold the estimated next call; that gate meant a large fraction of trajectories could
   silently drop half the candidate pool instead of scoring it. The clamp keeps every candidate
   priced on the same terms — a small-window model still gets its normal cache treatment (see
   step 3), it just gets it on a capped input size, so it isn't artificially rewarded for having
   less to bill.
3. **Apply the cache trap to the remaining candidates:**
   - **HOLD** (`model.name == trajectory.served_model`): `already_cached_tokens` bills at
     `cached_input_price_per_1m`; everything else —
     `(last_call_input_tokens − already_cached_tokens) + increment_tokens` — bills at
     `input_price_per_1m`.
   - **CHANGE** (any other candidate): the switch resets the provider's cache, so the *entire*
     `next_input_tokens` bills at `input_price_per_1m` — none of it is cached.
   - Either way, `next_output_tokens` bills at `output_price_per_1m`
     (`cached_output_price_per_1m` is 0.0 for every seeded model — see `pre_processing/model_list.py`
     — since neither pricing source discounts a freshly generated output token).
   - `cost = (uncached_tokens · input_price_per_1m + cached_tokens · cached_input_price_per_1m
     + next_output_tokens · output_price_per_1m) / 1e6`.
4. **Normalize:** `price_score = (max_cost − cost) / (max_cost − min_cost)` over the whole candidate
   pool; if every candidate ties, all get `1.0`.

This reproduces the cache trap directly: a candidate with identical per-token rates to
`served_model` scores strictly lower than holding, because a switch loses the cache discount on
`already_cached_tokens`; a much cheaper candidate can still outscore holding if its base rate is
low enough to absorb that reset penalty. With `total_cached_tokens = 0`, holding and switching to
an identically-priced model cost exactly the same — there is no cache to lose. Every candidate
receives a `price_score` — none are excluded or forced to `0.0` — so `score_price`'s min-max range
always spans the whole pool passed in. Covered by `tests/test_price_model.py` (full cache, partial
cache, no cache, switching, the cache-aggregate cap, the context-window clamp, and a regression
test pinning the `last_call_input_tokens` fix itself); verified against synthetic trajectories
only, not yet against the real export (no chunk was available locally when this was written —
see §3).

**Known simplifications, stated rather than hidden:**
- `increment_tokens` is a same-trajectory average (`last_call_input_tokens / total_calls`), not a
  real prediction of the next call's size; a trajectory with unusually front-loaded or back-loaded
  growth will be mis-estimated. Unlike the old `accumulated_context_tokens` bug, this is a
  bounded, secondary error — it only affects the *increment* term, not the whole next-call base.
- The formula scores one hypothetical next call, not the rest of the trajectory to come — it
  matches the README's "model for the next message" framing (§1) but doesn't project multiple
  future calls or their compounding cache effects.
- Compaction is modeled as a size clamp only — real summarization has its own cost (a call to
  compact the context) and loses information, neither of which this formula accounts for. It
  assumes "fits after compacting" is free and lossless, which favors small-context candidates more
  than reality would.
- `total_cached_tokens` is still a whole-trajectory aggregate (see `Trajectory`'s field docstring
  and `CACHE_HIT_RATE` in `pre_processing/trajectory_analyzer.py`), just capped down to
  `last_call_input_tokens` before use — it is not itself re-derived as "the last call's own cache
  share" from first principles.

### Performance

> **Implemented** as a pairwise-agreement model — see
> `analysis/complexity_model/complexity_scoring_spec.md` (read its REVISION header first)
> and `analysis/complexity_model/` for the pipeline.
>
> Cut a trajectory mid-flight and ask: how much would two models' *next actions* differ?
> Per-model scores are prior-weighted conformity over those pairwise values, so on a step
> where every model does the same thing all scores go to 1.0, the capability gap closes, and
> the router decides on cost alone.
>
> **Only OpenAI credits are available**, so just 3 of the 9 candidates can be queried. The
> other 6 are not guessed at from their benchmark score — that approach was tried and
> provably collapses to ranking models by prior-centrality, identically on every step.
> Instead their **real next actions are recovered from the corpus**: every line carries
> `model` plus the full action log, so at any interior cut the served model's actual next
> action is already there. That measures **21 of 36 model pairs at zero extra API cost**;
> the remaining 15 (Claude vs Claude, structurally unobservable since one model serves a
> whole trajectory) are Nyström-completed from the measured columns and flagged
> `extrapolated`. A ridge/logit model then learns prefix-features → agreement, so
> `capability_model.py` reproduces the scores at inference with **zero API calls**.
>
> Three measured findings worth carrying into any writeup: none of the probes accepts
> `temperature=0`, so every measurement carries sampling noise; measured **self**-agreement
> is **0.773–0.890** depending on the probe (§6), which is the ceiling on any agreement
> figure here — an earlier 25-step pilot put it at 0.74, superseded by the shipped
> artifact's 50-step control; and two encoding quirks
> (Claude preamble beside tool calls, GPT stop-turns logged as empty messages) each produce
> a ~5× spurious provider gap if parsed naively — i.e. they are `served_model` in disguise.
> `parser.classify_run` is the single rule that neutralises both.

- No quality labels exist in the export. Any performance score is a **proxy** — derive it from
  observable structure (platform inferred from the toolset, trajectory length, tool-call density,
  error and retry patterns in tool outputs), or from judge-model rescoring of individual calls.
  `Trajectory.normalized_items` is there so a proxy can be derived without re-parsing the export.
- **`data_models/quality_base_value.py` supplies a base quality value for when candidates disagree.**
  A proxy built from one trajectory's own observable structure can be noisy or silent on a given
  call — e.g. everything ties, or the trajectory is too short to have a real tool-error or
  retry-pattern signal yet. For those cases the router needs a fallback quality prior that isn't
  just "pick one" — `QUALITY_BASE_SCORE` in that module is exactly that: a `[0, 1]` value per model
  id, derived from three independent public leaderboards (Artificial Analysis Intelligence Index,
  Agent Arena, Text Arena/LMArena). Each benchmark is min-max normalized to `[0, 1]` within this
  project's 9-model pool first (so the three -- on wildly different native scales -- contribute
  equally instead of whichever has the widest raw spread dominating), then averaged with **equal
  weight**, unweighted, no further rescaling — the composite is that weighted average, full stop.
  `1.0` is therefore the ceiling a model would hit by topping all three benchmarks at once, not a
  value forced onto whoever leads this particular pool (no model here reaches it). See that
  module's docstring for the exact method, sourcing, and the "peak effort variant per model"
  convention it uses. It is a static, reproducible prior, not a substitute for a real
  per-trajectory `performance_score`; treat it as the tie-breaker/floor `performance_model.py`
  falls back on, not the primary signal.
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

Implemented as `route(trajectory, models, w_price=..., w_performance=...)`, which runs the whole
chain for one trajectory and returns a `RoutingDecision`. `final_score` is the weighted average
divided by the weight sum, so any pair of non-negative weights keeps it inside `[0, 1]`; a model
missing either sub-score is left out of the ranking rather than scored as 0. Two decision knobs
sit here: exact ties resolve to the incumbent, and `min_gain` sets how much `final_score` a
challenger must add before the switch is taken.

### Running the simulation

```
./run_simul.sh                                     # whole export, 50/50 weights
./run_simul.sh --w-price 0.15 --w-performance 0.85 # one point on the frontier
./run_simul.sh --limit 50 --min-gain 0.02
```

`app/main.py` parses every `data/*.jsonl` line, builds a **fresh** candidate pool per trajectory
(the stages enrich `ModelLLM` in place, so a shared pool would leak the previous trajectory's
scores), routes it, writes one row per trajectory to `data/routing_simul.csv`, and prints the
HOLD / CHANGE TO split with the most-changed-to model. ~23 s for the 1,000-line chunk.

---

## 6. The capability model (how `performance_score` is computed)

Implemented in `analysis/complexity_model/`, exposed to the router through
`router_models/performance_model.py`. Full design notes live in
`analysis/complexity_model/complexity_scoring_spec.md`.

### The idea, in plain terms

We have no answer key. Nothing in the dataset says "this was a good next move" — so we can
never directly measure whether a model is *right*.

What we can do is **ask a panel and see who agrees with whom.**

Picture a task paused halfway through. We show the same half-finished work to nine
different models and ask each one: *what would you do next?* Then we compare the answers.

- If **everybody proposes the same next move**, the step is easy. No model has an edge, so
  they all score near 1.0 — and the router should just pick the cheapest one.
- If **the answers scatter**, the step is genuinely hard. Now it matters who you side with,
  and we reward agreeing with the models we have good reason to trust.

That's the whole idea: **a model's score is how much of the panel agrees with it, where the
smarter panel members' agreement counts for more.** The score always lands between 0 and 1.

The gap between the best and worst score is a useful by-product: it tells the router *how
much this decision matters here*. A narrow gap means "anyone can handle this." A wide gap
means "this one is worth paying for."

### How it actually works

**Done once, offline:**

1. **Cut 957 real trajectories mid-task.** Everything before the cut is the "prefix" — what
   the agent had seen so far.
2. **Ask the three probe models what to do next.** Only OpenAI credits are available, so
   `gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-5.6-luna` are the only models we can query.
3. **Get the other six models' answers for free.** Every line in the export records which
   model served it *and* its full action log — so at any interior cut, the real next action
   of the model that served that trajectory is already sitting in the data. No API call
   needed.
4. **Compare every pair of answers.** Same action type and same tools → 1.0. Different
   action types → a value from a fixed table. Same type but only partly overlapping tools →
   scored by how much the tool sets overlap.
5. **Learn the pattern.** Fit a model that predicts "how much will these two agree?" from
   cheap properties of the prefix (how many calls so far, the mix of tools used, whether the
   cut lands right after a user message, signs of recent errors) plus a description of the
   two models being compared. The result is saved to `artifacts/pair_model.json`.

**Then at runtime, per trajectory — no API calls, a few milliseconds:**

Read the prefix, predict how much each pair of models would agree, and combine:

```
score(i) = ( w_i  +  Σ_{j≠i} w_j · agreement(i, j) )  /  Σ_all w_j

where   w_i = base_capability(i) ** 3
```

In words: **the weighted share of the panel that agrees with model i.** The `w_i` in the
numerator is the model agreeing with itself, which keeps the result inside [0, 1] and stops
a model being penalised for its own vote.

### Quality metrics — the honest numbers

Everything below comes from the shipped artifact (`artifacts/pair_model.json`), measured
with **5-fold `GroupKFold` grouped by step**, so no step ever appears in both the training
and test halves.

| | |
|---|---|
| Trajectory cuts used | **947** (of 957 sampled; 10 incomplete) |
| Pairwise comparisons (rows) | **10,882** |
| Cross-validation | 5-fold, grouped by step |
| Model MSE | **0.16893** |
| Baseline MSE (predict the mean, always) | 0.17135 |
| **R²** | **+0.0141** (1.4%) |
| Beats baseline | ✅ yes |

**Read that R² honestly: it is small.** The model explains about **1.4%** of the variation
in pairwise agreement. It is a real, reproducible improvement over guessing the average —
but it is not a strong predictor, and nothing in this repo should be presented as if it
were.

The main reason is the shape of what we're predicting:

| target value | share of rows |
|---|---|
| exactly 1.0 (full agreement) | **69.2%** |
| exactly 0.0 (total disagreement) | **20.9%** |
| anything in between | 9.9% |

Mean 0.7358, standard deviation 0.4139. Ninety percent of the target is a coin-flip-like
bit — "same action or not" — and much of that is genuinely unpredictable from the prefix
alone. There is a hard ceiling on how much any model could explain here.

**Held-out generalisation (leave-one-probe-out).** Fit on two probes, predict the third —
a probe never seen during training. All three beat their baseline:

| held-out probe | test MSE | baseline MSE | R² |
|---|---|---|---|
| `gpt-5.6-sol` | 0.16110 | 0.16229 | **+0.007** |
| `gpt-5.6-terra` | 0.16036 | 0.16621 | **+0.035** |
| `gpt-5.6-luna` | 0.16770 | 0.17237 | **+0.027** |

Small, but positive in every case — the model is learning something transferable, not
memorising probes.

**Measurement noise floor.** None of the probes accepts `temperature = 0`, so asking the
same model the same question twice does *not* give the same answer. Measured self-agreement:

| probe | self-agreement |
|---|---|
| `gpt-5.6-sol` | 0.860 |
| `gpt-5.6-terra` | 0.773 |
| `gpt-5.6-luna` | 0.890 |

**This is the ceiling on every agreement number in this repo.** A pair scoring 0.85 is not
"85% similar" — it is at roughly the level two runs of the *same model* reach.

**Pair coverage.** 21 of the 36 model pairs are measured (42 of 72 ordered cells), backed by
10,882 observations — but they are unevenly supported: the best-covered cell has **1,168**
observations, the worst has **1**. The remaining 15 pairs (Claude vs Claude) are
reconstructed, never observed. `ModelScore.measured_fraction` reports the split per model:
**1.0 for the three probes, 3/8 for the six Anthropic candidates.**

**One calibration check failed, and it is not silently ignored.** The logged-vs-elicited
offset δ — the part of disagreement caused by *how* an action was collected rather than by
model identity — was estimated three times independently:

| probe | δ |
|---|---|
| `gpt-5.6-terra` | −0.0947 |
| `gpt-5.6-luna` | −0.0175 |
| `gpt-5.6-sol` | +0.0298 |

Spread **0.1245**, above the 0.10 consistency threshold, so `is_consistent = False` and
**no correction is applied.** The three estimates disagree — including on sign — so
correcting would inject noise dressed as a fix. The gap is reported instead. This is a known
open weakness, not a solved problem.

### Design decisions worth knowing

**Why not just use published benchmark scores for the six models we cannot query?** It was
tried, and it provably collapses. Substituting a prior-distance function into the score
formula makes the step-dependent term factor out entirely, producing an **identical ranking
on all 947 steps** — and one that ranks models by how *central* their benchmark score is,
penalising the strongest model for being an outlier. Recovering real logged actions from the
corpus instead took pair coverage from 3/36 to **21/36 at zero extra API cost**.

**Why β = 3 and not 1?** β controls how much a model's benchmark prior weights its vote
(`w_i = prior_i ** β`). The design has two intents, and at β = 1 only one of them held.
Measured over 200 trajectories:

| β | ρ(score ranking, benchmark prior) on contested steps | top model on contested steps |
|---|---|---|
| 0 | 0.20 | `gpt-5.6-sol` |
| 1 | 0.52 | `gpt-5.6-sol` |
| 2 | 0.70 | `gpt-5.6-sol` |
| **3** | **0.73** | **`claude-fable-5`** |
| 5 | 0.75 | `claude-fable-5` |

At β = 1 the model rewarded for conformity was `gpt-5.6-sol` — a **mid-prior** model
(0.7274, against `claude-fable-5`'s 0.9559) that wins simply by sitting at the centre of an
all-GPT probe panel. β = 3 is the lowest value at which the strongest model actually tops
the ranking when it matters, without flattening the score into the prior.

Both intended behaviours hold at β = 3, measured:

| | spread (best − worst) | lowest score |
|---|---|---|
| settled steps (top quartile by agreement) | 0.088 | 0.795 |
| contested steps (bottom quartile) | 0.168 | — |

Agreement compresses the field; disagreement fans it out by capability. That ~2× ratio is
the signal the router consumes.

**Why the parsing rule matters more than it sounds.** Two encoding quirks each manufacture a
roughly **5× fake gap between providers** if parsed naively — Claude puts preamble text in
assistant messages next to tool calls, GPT logs stop-turns as empty messages. Either one is
`served_model` in disguise, which would turn the whole model into a provider classifier.
`parser.classify_run` is the single shared rule applied to both the elicited and the logged
path, which neutralises both.

**The score never reads `served_model`** — not as a feature, not indirectly through any of
the five proxies listed in §4. That leak would make the router predict the log instead of
improving on it.

### Known limitations

- **R² is 1.4%.** Real, reproducible, and small. The per-pair prediction is weak; the router
  consumes the *aggregate* spread across nine models, which averages many pair predictions
  and is steadier than any single one — but this should never be described as a strong model.
- **The 15 Claude-vs-Claude pairs can never be validated.** No trajectory is served by two
  models, so no data exists that could check them. They are capped at rank 3 by having only
  three probes, and flagged `extrapolated`.
- **The whole panel is GPT.** GPT-vs-GPT agreement (0.762) runs higher than Claude-vs-GPT
  (0.694), so conformity scoring structurally favours GPT. Raising β to 3 compensates; it
  does not remove the cause.
- **`claude-opus-4-6` and `claude-sonnet-4-6` served 2 and 1 trajectories.** Their scores
  rest on roughly one task each. They carry a `near_unmeasured` flag and must stay out of
  headline claims.
- **δ is unresolved** (above), so no logged-vs-elicited correction is applied.
- **Changing `BASE_CAPABILITY` requires a retrain.** The priors feed both the runtime weights
  *and* five of the fitted model's features, so editing them puts the saved artifact out of
  sync with the features it scores. β alone is free to change — it is applied after the
  regressor.

---

## 7. Evaluating the router (the hard part)

The log shows only the model that actually ran. Estimating what a *different* route would have
cost or delivered is off-policy evaluation, and it is the depth of this challenge — matching or
weighting across comparable trajectories, judge-model rescoring of single calls.

Naming where your estimate can fail is **scored, not penalized**. The deliverable is a
cost–quality frontier, not a single point, and a five-minute presentation with one chart, one
claim, and one known weakness.

---

## 8. Conventions for agents working here

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

## 9. Open decisions

Live list — resolve, then fold the answer into the sections above.

1. Performance scoring formula, and which observable proxies feed it.
2. Routing granularity: the decision is framed as "the model for the next message", while each
   exported line is a *complete* trajectory served end-to-end by one model. Decide whether the
   router may switch mid-trajectory — and if so, the cache penalty is not optional, and §4's
   92.1% cacheable share sets its size.
3. Default weights, and the sweep used to produce the frontier.
4. Whether to vendor the 3.6 MB `o200k_base` vocab into the repo. Today it is fetched once into
   a gitignored `.tokenizer_cache/`, so a fresh clone needs network exactly once; vendoring would
   make it offline from the first run at the cost of a binary blob in git.
5. Run `price_model.py` against the real export once a chunk is available locally, and fold any
   surprises (e.g. `increment_tokens` behaving badly on outlier trajectories) back into §5.

Resolved: `ModelLLM` field set (§2) and where the candidate pool comes from (§2,
`pre_processing/model_list.py`); `Trajectory` field set (§2) — 19 fields, see
`data_models/Trajectory.py`, which documents each one and its estimate/leak caveats inline; price
scoring formula and the assumed price sheet behind it (§5, `router_models/price_model.py`).
