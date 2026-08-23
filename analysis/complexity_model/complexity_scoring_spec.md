# REVISION — OpenAI-only pivot (supersedes §1.0, §2, §4, §6 below)

**Read this first.** Only OpenAI credits are available, so the six Anthropic candidates can
never be queried. The design below assumed all candidates would be elicited; that premise is
gone. Everything else — the pair table (§5.1 Stage A/B), the score formula (§5.1 level 2),
the response store (§2.4), step-major order (§2.3), the leakage rules (§6.2) — is unchanged
and still governs.

## What changed

| | as specced | as built |
|---|---|---|
| Candidates scored | 7 | **9** (the two rare models return — they cost nothing now) |
| Candidates queried | 7 | **3** (`gpt-5.6-sol`, `-terra`, `-luna`) |
| Anthropic agreement | elicited | **recovered from the corpus log** |
| Pairs measured | 21/21 | **21/36**, the rest completed |
| Cost | ~$143 / 200 steps | **~$152 / 957 steps** |

## The rejected fix, and why

The obvious move — synthesize Anthropic agreement from distance on the `base_capability`
axis, `pair = 1 − (1−A(step))·S(d)` — is **degenerate**. Substituting it into §5.1:

```
score(i) = 1 − (1 − A(step)) · C_i ,      C_i = Σ_{j≠i} w_j·S(d_ij) / W
```

`C_i` carries no step term, so the model ranking is **identical on every step**; `A(step)`
only scales the gaps. Worse, `C_i` is minimised by whichever model sits nearest the weighted
centroid of the priors, so it scores *centrality*, not capability: on the real priors it puts
`gpt-5.6-sol` above `claude-opus-5` and `claude-fable-5` on all 957 steps, and at `BETA=0`
`claude-fable-5` — the highest prior — ranks last. §4's own sanity check fails by
construction, before any API call. Two independent analyses derived this; it was verified
numerically (50 consensus values → 2 distinct rankings, the second being the all-tied case).

## What replaces it — corpus recovery

Every export line carries `model` **and the full action history**, so at any interior cut the
served model's real next action is already in the log: it is exactly the run of items the
sampler cuts away. We cannot query Anthropic; we do not need to.

Per step: elicit the 3 probes, recover the served model's logged action, score **12 ordered
rows** (6 probe×probe, 6 logged×probe). Pooled over the corpus this fills all three probe
*columns* for all 9 models — **21 of 36 unordered pairs measured, at zero extra API cost** —
and, because the pairs are now real observations rather than a function of prior distance,
the ranking varies by step as it must.

The 15 Claude↔Claude pairs remain unobservable (one model serves a whole trajectory) and are
Nyström-completed from the probe columns (`completion.py`): *two Claude models agree to the
extent they agree with the same probes*. Exact if agreement is rank-3; ~0.07 mean / ~0.26
worst-case error by rank 4, and **with 3 probes we cannot tell which regime we are in**.
Those cells are flagged `extrapolated` and never presented as measurements.

## Findings that overturned stated assumptions

- **No temperature control.** All three probes return 400 on `temperature=0` — §2.1 assumed
  only some models would. Every measurement therefore carries real sampling noise, and §8's
  "sampling noise is unmeasured" becomes central rather than a footnote.
- **The measurement ceiling is not 1.0.** Measured self-agreement (same model, same prefix,
  two draws) was **0.74** over a 25-step sample, with a 72% exact-match rate. No pair of
  *different* models can be measured as agreeing more than that. Every reported agreement
  must be read against this ceiling, and it is why the self-pair control (§2.1) is a real
  measurement here, not a formality.
- **Two encoding traps, both ~100% correlated with provider** — i.e. both are `served_model`
  in disguise, which README §4 forbids any score from reading. Reading "any assistant message
  means the turn was a message" gives Claude a 56–72% message-rate against 8–14% for GPT
  (74–83% of those Claude items are preamble beside a tool call). And a GPT stop-turn is
  logged as an *empty* assistant message where Claude carries text — scoring that `malformed`
  penalised GPT for an encoding quirk, while dropping it deleted GPT's stop-turns and kept
  Claude's. All 65 such turns are followed by a user message, so they are genuine stops.
  `parser.classify_run` resolves both with one rule (*did it act, or stop?*), after which all
  nine models sit at 93.7–97.1% `tool_call` and 0–0.4% `malformed`.
- **The 1M context filter (§1.1) is a no-op** — the largest prefix is ~123K tokens.

## Honest limits

- 15 of 36 pairs are completed, never observed, and **not validatable**. Leave-one-probe-out
  tests the machinery on cross-provider cells only.
- `claude-opus-4-6` and `claude-sonnet-4-6` served 2 and 1 trajectories. They are scored and
  flagged `near_unmeasured`; cuts within one trajectory are near-duplicates, so the effective
  sample is ~1 task each. Keep them out of headline claims.
- Logged actions carry home-field advantage (the model wrote the whole prefix) and the
  harness's sampling settings. `calibration.py` measures that offset per probe and **refuses
  to correct when the three estimates disagree**, rather than hiding the contradiction.
- Conformity is still not correctness (§8). Now an empirical question rather than a formula
  artifact: check whether `claude-fable-5` is penalised for behaving idiosyncratically.
- **The probes are the coordinate system, and all three are GPT.** Measured on the full run,
  gpt-vs-gpt agreement averages **0.762** against **0.694** for claude-vs-gpt. Conformity
  scoring rewards agreeing with everyone, so a model that resembles the probes scores higher
  *by construction* — and every probe is OpenAI. Part of that gap is regime (gpt-vs-gpt is
  elicited/elicited, claude-vs-gpt is logged/elicited; `delta` bounds that at roughly ±0.03,
  so at most half) and part is genuine same-provider similarity, but the two cannot be
  separated with a GPT-only probe set. **Consequence for the router: it will tilt toward GPT,
  and that tilt is partly an artifact of which models we were allowed to query.** The fix is
  a non-GPT probe, which the credits constraint forbids. Not uniform, though —
  `claude-opus-4-8` vs `gpt-5.6-terra` scores 0.797, above several gpt-gpt pairs.

---

# `capability_model.py` — Design Spec

Handoff document for implementation. Stages 1–6 are offline and run once.
Stage 7 is the shipped artifact.

---

## 0. Context and rationale

The router scores each candidate model 0–1 on a mid-flight agent trajectory.
`capability_model.py` produces the capability half; `cost_model.py` produces the
cost half; `model.py` combines them in a weighted average.

**What is being predicted.** Not task complexity. There is no difficulty scale, no
latent ability, no IRT. The quantity is **pairwise divergence**: given a trajectory
prefix, how much would model `a`'s next action differ from model `b`'s? Per-model
scores are then derived from those pairwise values by rewarding conformity, with
agreement against a high-prior model counting for more (§4, §5.1).

**Required property.** The gap between models must depend on the step. Easy step →
all models pick the same action, all score 1.0, gap ≈ 0, cost decides. Contested
step → the gap opens and prior-weighted conformity decides. This falls out of the
scoring formula in §5.1 directly; no link function or tuning is needed to produce
it.

**Supervision.** The training corpus has no outcome labels and the agent harness is
not available, so tool results cannot be simulated. Supervision is derived from
*inter-model divergence on a single inference round*: given a prefix, query every
candidate model for its next action and measure how much they diverge. No tool
execution is required — the prefix already contains all prior tool results.

**Pipeline shape.**

```
offline:    prefixes → query N models → pair(a,b) for all pairs, per step
            → regress prefix features onto pair(a,b)
inference:  prefix → features → predicted pair(a,b) for all pairs
            → prior-weighted conformity → score per model
```

---

## 1. Sampling

### 1.0 Candidate set

Seven models. Every step is scored against all seven; every pair among them is
compared (21 unordered pairs).

```
claude-opus-5
claude-sonnet-5
claude-fable-5
claude-opus-4-8
gpt-5.6-terra
gpt-5.6-sol
gpt-5.6-luna
```

**`claude-opus-4-6` and `claude-sonnet-4-6` are excluded.** Both are still active
on the API — Opus 4.6's retirement is dated no sooner than February 2027, and
Sonnet 4.6 is the current migration target for the retired Sonnet 4 — so this is
not a deprecation cut. The justification is corpus share: they served 2 and 1
trajectories respectively out of 1000. At 200 sampled steps the expected count for
both combined is under one trajectory, so neither would be meaningfully
characterized either way.

The saving is the point. Dropping them removes two of six Anthropic queries per
step — a ~33% cut in private-credit calls, which is the binding budget (§1.2).

Trajectories *served by* those two models are still eligible for sampling. The
served model is irrelevant to sampling; only the candidate set matters.

**Pilot: 200 samples.**

- 200 trajectories drawn at random from the eligible corpus (see §1.1).
- One sample per trajectory. Steps within a trajectory are near-duplicates — same
  task, same files, same tools — so sampling across trajectories yields far more
  effective diversity than sampling within.
- **Cut at a step in the middle of the trajectory, not the end** — see §1.2.
- **Every model in the candidate set is queried on every step**, including the
  model that served the trajectory. No response is ever taken from the log.

Each sample record:

```
{
  step_id,             # stable, used as cache key
  trajectory_id,
  prefix_messages,     # OpenAI-format message list, verbatim from log
  system_prompt,       # verbatim from log if present
  tools,               # tool schemas — see §1.3
  prefix_token_count,  # of the prefix actually sent, not the whole trajectory
  step_index,          # position of the cut
  served_model         # generating model; logged as a diagnostic, not used
}
```

### 1.1 Context-window filter

Exclude any sample whose **prefix as actually sent** exceeds:

```
MAX_PREFIX_TOKENS = 1_000_000
```

1M is adequate for every model in the candidate set. Around 99% of the corpus falls
under it, so the exclusion costs almost nothing.

The filter exists because the official harness performs context compaction, which
lets it run past the raw window. This pipeline does not compact, so a prefix that
was fine in-flight can hard-fail here. A step that errors for some models but not
others produces a partial row, which silently distorts conformity scoring — a
missing response is not a neutral one.

Filter on the prefix length, not the full trajectory length — the cut is mid-way,
so a trajectory whose full length exceeds the cap may still be eligible. Log the
number excluded.

### 1.2 Why a mid-trajectory cut

- **Prefix length roughly halves**, cutting input tokens for every model queried.
  Input tokens dominate here — prefixes are long, responses are a single action —
  so this is the largest single saving in the pipeline, and it costs nothing in
  measurement quality.
- It keeps most trajectories under `MAX_PREFIX_TOKENS`.
- It still matches the inference-time distribution: the router always sits
  mid-flight. Any interior step is representative; the final step is not
  privileged.

Pick the cut point at random within the interior of the trajectory rather than
exactly at the midpoint, so the sample spans a range of depths. Record
`step_index` and check afterwards whether divergence correlates with depth.

### 1.3 Tool schemas — OPEN ITEM, resolve first

Everything downstream depends on this. Three cases:

1. **Logs carry `tools` in the request payload.** Use verbatim. Preferred.
2. **Not present — induce from corpus.** For each observed tool name, take the union
   of argument keys across all observed calls, infer types from observed values,
   mark always-present keys as required. ~1000 trajectories gives good coverage of
   common tools; rare tools stay underspecified and that is accepted.
3. **Tool calls logged in lossy/summarized form with no recoverable argument
   structure.** This approach is not viable — escalate.

Also check whether logs contain tool-result messages with error content. Those
retroactively mark calls as wrong and are cheap extra signal if present.

---

## 2. Elicitation

For each sampled step, query every candidate model once for its next action —
including the model that served the trajectory. Every step costs 7 queries. No
response is ever taken from the log.

- **Reuse the logged system prompt and tool schemas verbatim.** Any deviation in
  prompt or tool formatting introduces prompt-mismatch that will be measured as
  divergence. This is the single largest confound in the pipeline.
- Store **raw** responses. Do not score at this stage.
- **One sample per model per step.** Temperature 0 where the model accepts it —
  see §2.1.

### 2.1 Temperature — not uniform across the candidate set

Target behaviour is temperature 0 everywhere, for determinism. **Some models in the
set reject the parameter outright.**

`claude-opus-4-8` returns a **400 error** if `temperature`, `top_p`, or `top_k` is
set to any non-default value — inherited from Opus 4.7. Anthropic's guidance is to
omit these parameters entirely and steer via prompting. The same is likely to apply
to other recent Anthropic models in the set; **verify per model with a single test
call before the batch run** rather than assuming, and check the current model
documentation, since this constraint has moved between releases.

Rule: set `temperature = 0` for models that accept it; **omit the parameter
entirely** for models that reject it. Never send a default value explicitly to
work around the error — omission and explicit-default are different requests to
these APIs.

Record the effective temperature per model alongside the responses. Models running
at their default rather than 0 carry sampling noise the others do not, which is a
measurement asymmetry — see §8.

### 2.2 Harness

**Do not use LangChain.** This is not an agent loop — it is
`prefix_messages → API call → parse response`, one round. LangChain's tool-calling
wrappers reshape the payload silently, which is exactly the confound above.

Required components, all thin:

- **Loader** — yields `(prefix_messages, tools, system_prompt)` from JSONL.
  Applies the §1.1 filter.
- **Client wrapper** — per provider; messages + tools in, raw response out.
  Retries with backoff.
- **Response store** — see §2.4. Non-negotiable.
- **Step-major driver** — see §2.3. Non-negotiable.
- **Concurrency** — parallel within a step, across the 7 models (§2.3).
- **Parser** — see §3. This is where the real work is.

**OPEN ITEM:** is the candidate set cross-provider or all OpenAI-compatible? If
OpenAI-compatible, a single gateway collapses the client and parser layers to
near-nothing.

### 2.3 Iteration order — step-major, REQUIRED

**Iterate step-major: complete all 7 models on step 1, persist, then move to
step 2.** Never model-major (all steps for one model, then the next).

This is not a style preference. Conformity scoring (§5.1) needs every model's
response on a step to compute the denominator, so a step with 4 of 7 responses is
worth nothing. The two orders spend identically and fail completely differently:

- **Model-major, interrupted at 50%:** 200 steps each holding 3–4 responses.
  **Zero usable rows.**
- **Step-major, interrupted at 50%:** 100 complete steps, 100 untouched.
  **100 usable rows** — a smaller pilot, not a failed one.

The run will likely be interrupted — by budget, rate limits, or time. Assume it
will. Under step-major, whatever number of steps completes is a valid dataset:
getting 123 of 200 means training on 123.

Parallelize *within* a step across the 7 models, not across steps. Only mark a step
complete once every model has a stored record (success or terminal failure).

### 2.4 Response store — durability requirement

Elicitation must be **crash-safe and fully resumable**. Anthropic calls are paid
from private credits; a crash that loses completed responses burns real money and
hackathon time. This is a hard requirement, not an optimization.

**Persist each response the moment it returns, before issuing the next call.**
Never buffer results in memory for a batch write at the end. If the process dies on
call 4, calls 1–3 must already be on disk and must not be re-issued on restart.

- **Key:** `(model, step_id)`. One response per pair — there is no `sample_idx`,
  since §2 fixes one sample per model per step.
- **Write granularity:** one record per response. Append-only JSONL, or one file
  per key. Do not rewrite a whole aggregate file on each response — a crash
  mid-rewrite can corrupt everything already collected.
- **Resume:** on startup, read the store, compute the set of `(model, step_id)`
  pairs already present, and skip them. Restarting the script after any failure
  must be safe and must cost nothing for work already done.
- **Store failures too.** A response that errored after exhausting retries is
  recorded as a failed record with the error, not left absent. Absent means
  "not yet attempted" and will be retried forever; failed means "attempted, gave
  up." These must be distinguishable.
- **Store raw.** Persist the provider's unmodified response body. Parsing (§3) and
  scoring (§5) run offline over the store and must be re-runnable without
  re-querying — every later change to the scoring rules should cost zero API calls.

- **Completeness filter at scoring time.** §5 consumes only steps where all 7
  models have a successful record. Steps with any missing or failed response are
  excluded from the matrix entirely — a partial row silently distorts conformity
  scoring, because an absent response is not a neutral one. Log the count of
  excluded steps.

The store is the expensive artifact in this pipeline. Everything downstream is
cheap and reproducible from it.

---

## 3. Normalization

Parse each raw response into a canonical action. If comparing across providers,
OpenAI / Anthropic / Google tool-call representations must land in **identical**
canonical form or format differences will be measured as disagreement.

```
{
  type: "malformed" | "message" | "tool_call",
  tool_names: set[str],    # empty unless type == "tool_call"
  raw: <original response>
}
```

A step may contain parallel calls, so `tool_names` is a set. Arguments are **not**
retained for comparison — they are used only for schema validation when deciding
between `tool_call` and `malformed`, then discarded. No argument normalization is
needed anywhere in the pipeline.

### Type definitions

- **`malformed`** — tool call naming a tool absent from the toolset, or a call that
  fails schema validation against its tool definition. Also covers unparseable
  output. This is an **explicit class, never a dropped row**: a model that cannot
  emit a valid call is failing the step.
- **`message`** — human-facing response, no tool calls.
- **`tool_call`** — one or more valid calls. If a response mixes valid and invalid
  calls, classify the whole response `malformed`.

Note: "which tools were called by all models" is **not** a third type. It is a
population-level statistic, knowable only after all responses are collected, and
nothing in this pipeline consumes it — tool-set comparison (§5.1 Stage B) is
unweighted.

---

## 4. Prior weights

**There is no reference model and no strong-model subset.** Every model is scored
by how much the others agree with it, weighted by how much each of those others is
trusted a priori. Nothing is privileged as ground truth.

Each model carries a fixed weight `w_i > 0`, set once and reused for every step:

```
w_i = prior_i ** BETA
```

`prior_i` is a normalized 0–1 public-benchmark aggregate — the `base_capability`
score. Price per token is an acceptable substitute if benchmark coverage is thin;
it is the market's own capability estimate.

`BETA` controls how much the prior counts:

- `BETA = 0` → all weights equal, pure conformity, prior ignored.
- `BETA = 1` → agreement with a model counts in proportion to its prior.
- large `BETA` → approaches "only the best model's opinion matters".

Sweep `BETA` on the pilot and keep the curve — it is a single scalar and a
sensitivity plot over it is a good artifact to show.

**`BETA` must be a runtime configuration parameter, not a constant baked into the
regressor.** It is applied at inference time in `capability_model.py` (§7), after
the regressor has produced its pairwise predictions. Changing it must never require
retraining or re-querying anything. Expose it alongside the cost/capability weights
in `model.py` so the whole trade-off surface is tunable from one place.

**Sanity check on `BETA`.** As `BETA` grows the output collapses toward the prior
and the pipeline degenerates into an expensive benchmark lookup. Compute the rank
correlation between the final per-model scores and `prior` itself. If it is near
1.0, the measurement is contributing nothing and `BETA` is too high.

---

## 5. Scoring

Output is a matrix: `step_id × model → score ∈ [0, 1]`.

### 5.1 Scoring function

Three levels, bottom-up:

1. **`pair(a, b)`** — agreement between two responses, in [0, 1]. Stage A + B below.
2. **`score(i, step)`** — model `i`'s capability score on this step, from all its
   pairwise agreements plus its own prior weight.
3. The matrix is the collection of `score(i, step)` over all models and steps.

All pairs are computed: with the 7 candidates that is 21 pairwise scores per step.
No extra
API calls — the responses are already collected.

Argument comparison and semantic-equivalence judging are **out of scope**. A tool
call is compared by tool name only. Two calls to the same tool with different
arguments count as agreement. This is a deliberate simplification — see §8.

#### Level 2 — `score(i, step)`

```
score(i, step) = ( w_i + Σ_{j≠i} w_j · pair(i, j) ) / Σ_all_j w_j
```

Each model agrees with itself, weighted by its own prior. `w_j` is from §4. The
denominator is the sum over **all** models including `i`, so the result is in
[0, 1] with no clamping needed.

This single expression produces every behaviour the model needs — verify against
these three cases when implementing:

- **Unanimity.** All `pair(i,j) = 1`, so every model scores exactly 1.0 regardless
  of weights. The prior cancels out. Capability gap is zero and `model.py` routes
  on cost alone, which is correct: if every model does the same thing, buy the
  cheap one.
- **Total disagreement.** All `pair(i,j) = 0`, so `score(i) = w_i / Σw`. The
  ranking is the prior ranking — nobody knows what to do, so the strongest model
  has the best chance. No branch, no threshold; it falls out.
- **Partial clustering.** A model inside a cluster of high-prior models scores
  well; a lone dissenter falls back toward its own prior share. A model's own
  `w_i / Σw` is the floor it can never drop below.

**Scores are not normalized per step.** At total disagreement the absolute values
are small (≈ 0.125 each with 8 equal-weight models) and the spread is narrow, so a
cost-heavy weighting in `model.py` will route cheap even on contested steps. This
is intended: the cost/capability tradeoff belongs to the user's weights, not to
this module. See §8.

#### Level 1, Stage A — type pair

Constants, defined once and tunable:

```
SCORE_BOTH_MESSAGE      = 1.00
SCORE_MESSAGE_WHEN_TOOL = 0.00   # one stopped, the other kept working
SCORE_TOOL_WHEN_MESSAGE = 0.15
SCORE_MALFORMED         = 0.00
```

| a ↓ / b → | `tool_call` | `message` | `malformed` |
|---|---|---|---|
| `tool_call` | → **Stage B** | `SCORE_TOOL_WHEN_MESSAGE` | `SCORE_MALFORMED` |
| `message` | `SCORE_MESSAGE_WHEN_TOOL` | `SCORE_BOTH_MESSAGE` | `SCORE_MALFORMED` |
| `malformed` | `SCORE_MALFORMED` | `SCORE_MALFORMED` | `SCORE_MALFORMED` |

`pair` is **not symmetric**: the `tool_call`/`message` pair scores 0.15 in one
direction and 0.00 in the other. Read the table as *row = the model being scored,
column = the model it is being compared against*. So `pair(i, j)` uses `i` as the
row. Compute both directions; do not cache one and reuse it for the other.

Three things this encodes, all intentional:

- **The type mismatch is asymmetric.** A model that stops when others kept working
  is a silent task failure and scores 0. A model that keeps working when others
  stopped is recoverable — one redundant step — and gets partial credit.
- **`malformed` scores 0 in every cell.** It is never agreement, in either
  direction. Two malformed responses are not two models concurring.
- **If every response on a step is `malformed`, drop the step.** Every score would
  be `w_i/Σw`, identical to total disagreement but caused by a broken schema rather
  than a hard step. Log the count — nonzero means §1.3 needs revisiting.

`message`-vs-`message` scores 1.00 on co-classification alone. Content is not
compared. Two models could be saying opposite things and both score 1. Accepted;
see §8.

#### Level 1, Stage B — tool set overlap

Runs only when both responses are `tool_call`. Compare the **set of tool names**
invoked, not individual calls. A step may contain parallel calls, so both sides
are sets.

Plain Jaccard, unweighted:

```
pair(a, b) = |A ∩ B| / |A ∪ B|
```

Identical tool sets → 1. Disjoint tool sets → 0. Partial overlap → partial credit,
proportional to the share of the union both sides share. Stage B is symmetric even
though Stage A is not.

No per-tool weighting. Down-weighting tools that every model calls is a refinement
that can be applied later, offline, without re-querying anything — the raw
responses are cached, so rescoring is free.

**Set semantics collapse duplicates.** A model calling `read_file` three times on
different paths contributes `{read_file}`, identical to a model calling it once.
Direct consequence of dropping argument comparison.

Do **not** use embedding cosine similarity on serialized calls at any point.

### 5.2 Per-model malformed rate

Track as a standalone per-model statistic **in addition to** counting it as
disagreement. It is a direct capability fact and should not be laundered through a
divergence score. It is also the most legible single number available for a demo.

---

## 6. Regression

### 6.1 Data layout — pooled, one row per (step, pair)

**Do not train 21 separate models.** With 200 steps (possibly fewer, see §2.3),
21 independent regressors get 200 rows each — not enough for any feature set worth
having.

Reshape to long format instead: **one row per `(step, pair)`**, giving
`200 × 21 = 4,200` rows, a single model, and a single scalar target.

Each row's features are the step features (§6.2) concatenated with **pair-descriptor
features** describing which two models are being compared:

- absolute difference of the two models' `prior` values
- minimum of the two priors
- maximum of the two priors
- same-provider flag
- same-family flag (e.g. both Opus)

Target: `pair(a, b)` from §5.1, in [0, 1].

This layout is what makes the problem learnable at this sample size. It also
generalizes: the model learns *"on steps like this, two models this far apart on
the prior scale diverge this much"*, so a new candidate model can be scored without
retraining, provided it can be placed on the prior scale.

Ordering: `pair` is asymmetric only on the `tool_call`/`message` cell, so predict
the 21 unordered pairs and apply the asymmetry as a post-hoc correction. If that
proves fiddly, expand to all 42 ordered pairs — it doubles the rows and costs
nothing.

### 6.2 Features

**Not prescribed.** Use whatever can be extracted from the trajectory object.
Anything that plausibly predicts whether two models will pick different next
actions is fair game, and the implementer is better placed to judge that after
seeing the actual data shape than this document is. Build a wide feature set, let
ridge's L2 penalty suppress what doesn't carry, and inspect the coefficients.

Two hard constraints, neither of which is about feature choice:

- **No leakage.** Every feature must be computable from the prefix *as it exists at
  the cut point*. Nothing from `recorded_action`, nothing from the remainder of the
  trajectory, nothing from the responses being predicted. At inference the router
  has only the prefix; a feature that isn't available then is unusable no matter
  how well it scores offline.
- **Same extraction code offline and at inference.** Feature computation lives in
  one function called by both §6 and §7. Two implementations drift, and the drift
  shows up as a model that validates well and performs badly.

On `served_model`: check whether the router actually knows the serving model at
inference time before using it as a feature. If it does, it is legitimate and
probably informative. If it does not, it is leakage.

**Pair-descriptor features are not optional** — see §6.1. Without them the pooled
rows are indistinguishable across pairs and the model fits one average divergence
curve for all 21.

### 6.3 Model choice — ridge regression

**Use ridge regression (linear, L2, cross-validated alpha).** Not SVM, not
polynomial features, not a neural network. At a few thousand rows, linear-with-L2
is the appropriate capacity, not a compromise — anything with more capacity will
fit noise in targets that are themselves single noisy observations. Regularization
is also what lets §6.2 stay open-ended: a wide feature set is safe here because the
penalty suppresses what doesn't carry.

The one upgrade worth attempting, and only after ridge has produced a number to
beat: **shallow gradient-boosted trees**, heavily regularized (depth ≤ 3, low
learning rate, early stopping). Try it if diagnostics suggest non-linearity. Do not
start there.

**Bounded target.** `pair` is in [0, 1] and, if the distribution is bottom-loaded
as expected, heavily massed at 1.0. Ridge will emit values outside the range.
Either logit-transform the target with clipping away from the endpoints, or predict
raw and clamp at inference (§7 requires the clamp regardless).

**Cross-validate by step, not by row.** The 21 rows from one step share all step
features. Random row splits leak the step across train and test, and the score
comes back optimistic. Use grouped CV with `step_id` as the group.

**Baseline to beat: the constant predictor** — every pair gets the corpus mean
`pair` value. With a target massed near 1.0 this scores deceptively well on MSE. If
ridge does not beat it by a clear margin, the features carry no signal, the router
is a benchmark lookup with extra steps, and that is worth knowing immediately
rather than at demo time.

**Precision note.** Each target is observed once, from a single pair of responses.
Targets are noisy; the noise is independent across steps, so the regressor averages
over it. More sampled steps helps; nothing else does.

---

## 7. Inference — the shipped artifact

**Build a stub of this first**, returning constants, so `model.py` integration is
unblocked while the offline pipeline runs.

Loads: the pairwise regressor, and the prior weights `w_i` from §4.

```
prefix
  → step features (§6.2)
  → build 21 rows: step features + pair descriptors, one per model pair
  → regressor → predicted pair(i,j), clamped to [0,1]
  → score(i) = ( w_i + Σ_{j≠i} w_j · pair(i,j) ) / Σ_all_j w_j
  → { model_name: score ∈ [0,1] }
```

Step features are computed once and reused across all 21 rows; only the pair
descriptors differ. One regressor call on a 21-row batch, not 21 calls.

The prior weights stay **outside** the learned model. The regressor predicts
divergence only; conformity weighting is explicit code. This keeps `BETA` tunable
without retraining and keeps the two concerns separable when debugging.

No API calls. Milliseconds. Return shape unchanged, so `model.py` is unaffected.

Clamp predicted `pair` values to [0, 1] before aggregating — a regressor can emit
values outside the range, and the score formula's bounds depend on inputs being
within it.

**No hard-task floor is needed.** A step where all models diverge yields
`score(i) = w_i / Σw`, which ranks by prior — the strongest model wins by
construction. This is handled by the formula, not by a special case.

---

## 8. Known limitations — state these, do not hide them

- **Teacher forcing.** Candidates continue a prefix they did not generate. This is
  a real measurement bias, but it matches the inference-time distribution exactly:
  the router always sits mid-flight on history some other model produced.
- **Single-round only.** Compounding behaviour — whether a model recovers from its
  own mistake three steps later — is not measured. Acceptable, because the router
  decides one step at a time against fixed history.
- **Sampling noise is unmeasured.** Each model is queried once per step, so
  intra-model divergence is never observed and cannot be subtracted from
  inter-model divergence. Partly mitigated by temperature 0 where it is accepted
  (§2.1) and by the coarseness of the
  comparison — type classes and tool-name sets are both stable under resampling.
- **Contested steps produce compressed scores.** With no per-step normalization,
  a step where all models disagree yields scores near `w_i/Σw` — correctly ranked
  but narrowly spread. Under a cost-heavy weighting in `model.py` such a step
  routes to the cheapest model. This is by design: the capability module reports
  what it measured, and the cost/capability tradeoff is the user's weighting
  decision, not something this module pre-empts by inflating its own spread.
- **Comparison is coarse by design.** Tool arguments are ignored, so a model that
  reads the correct file and one that reads the wrong file score identically.
  Duplicate calls to the same tool collapse. Message content is never compared, so
  two models giving opposite answers to the user both score 1.00 against each
  other. This buys a pipeline with no LLM judge, no argument normalization and no
  per-tool weighting — at the cost of a signal that resolves only at the level of
  *which action type and which tools*, not *whether the action was right*. Expect
  the agreement range to be compressed toward 1.0 as a result.
- **Home-field advantage for the serving model.** Every prefix was written
  end-to-end by one model, and that model is usually still in the candidate set.
  When it is queried on its own trajectory it continues its own conventions, while
  the other six continue a stranger's. This may inflate its measured centrality,
  and centrality is the score. It is distributed in proportion to corpus share —
  claude-opus-5 served 33% of trajectories, so it gets the home-field position far
  more often than the rest. Mitigated but not removed by the fact that all seven
  models are now queried identically; the asymmetry is in the prefix, not the
  procedure. Log `served_model` and check whether a model scores systematically
  higher on its own trajectories than on others' — that comparison measures the
  effect directly and needs no extra calls.
- **Temperature is not uniform.** Models that reject the parameter (§2.1) run at
  their provider default and carry sampling noise the temperature-0 models do not.
  Their measured divergence is inflated by an unknown amount, which under
  conformity scoring reads as lower capability.
- **Conformity is not correctness.** A model is rewarded for doing what others do.
  If several models make the same mistake and one gets it right alone, the correct
  model scores lowest on that step. The prior weighting in §4 mitigates this — a
  lone high-prior dissenter keeps a floor of `w_i/Σw` — but does not remove it.

---

## 9. Execution order

1. Resolve §1.3 (tool schemas) and §2.2 (provider set). Everything depends on these.
2. Verify per-model temperature support (§2.1) with one test call each before
   running the batch.
3. Stub Stage 7, unblock `model.py`.
4. Build loader / client / step-major driver / response store / parser. The store
   (§2.4) and the step-major order (§2.3) come before any real querying — test
   resumability by killing the process mid-run and restarting it.
5. Run the 200-sample pilot. It may not finish; under §2.3 that is fine, and
   whatever completed is a usable dataset. **Inspect the `pair` distribution before
   scaling.** Expect it heavily massed at 1.0 — most agentic steps are obvious and
   every model does the same thing. If so, the signal lives in a thin tail, which
   changes the strategy: build a cheap pre-filter for contentious steps and spend
   budget there. This cannot be known before the first run.
6. Fit ridge against the constant-predictor baseline (§6.3). If it does not clearly
   beat the baseline, stop and revisit features before building anything on top.
7. Replace the stub. Sweep `BETA`.