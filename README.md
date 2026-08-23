# Cheapy — Viktor Challenge submission

**Members:** Pietro Quinzani · João Sandre · Rodrigo Takahashi

Viktor pins one model for a whole trajectory. Cheapy reads the trace and picks which of nine
models should serve the *next* call — `HOLD`, or `CHANGE TO <model>` — from a price score and a
capability score. Offline, no API keys.

**Objective:** cost per call vs. capability retained. Cost is *quadratic* in trajectory length —
every call re-sends the whole history, and 92% of billed input in the export is re-sent prefix. So
the money is in long trajectories, and the question is: which is the cheapest model still able to
take this step? We optimize the **frontier**, not a point: `W_COST` / `W_PERFORMANCE` in
[`cheapy.yaml`](cheapy.yaml) dial from "cheapest acceptable" to "best regardless of cost".

**Routing signal:** prefix-cache structure and step shape. Call `i`'s prompt is by construction a
prefix-extension of call `i−1`'s, so the cached share is exact rather than searched for, and a
switch resets that cache — the price score prices that directly. The capability score reads the
step only: prefix size, call count, tool-calls per call, distinct-tool ratio, repeated-tool runs,
recent error rate, last output length, token shares, toolset size, Slack vs Teams, subagent flag.
Model identity enters as a prior alone, so nothing is downstream of who actually served the call.

**Headline result (cache-aware, all 1,000 trajectories, estimated tokens):** at `W_COST = 0.10`,
**−63.1% estimated next-call cost** vs. today's incumbent-only policy ($0.0277 → $0.0102 per
trajectory), keeping **98.4% of the incumbent's mean capability score**; at `W_COST ≥ 0.15` it
saturates at **−78.2%**. **That second number is the router's own proxy, not a measured outcome** —
the export has no outcome to hold out. Held-out numbers exist for the proxy itself:
leave-one-probe-out R² +0.007 / +0.027 / +0.035, positive in all three folds.

**Off-policy method:** elicitation + pairwise conformity — a judge panel, not matching or
weighting. We cut 957 trajectories at a random step, asked three probes what action they would
take, scored agreement against the logged action, fit a pairwise-agreement regressor on the
features above, and Nyström-completed the 15 pairs we could never query (OpenAI credits only).
**Weakest point: the capability signal is far weaker than price, so the router collapses into
"always the cheapest model".** R² = 1.4%; capability separates candidates by ~0.17, price by orders
of magnitude. The crossover band is `W_COST` ≈ 0.05–0.08; above it, ~98% of trajectories go to one
model. Two things we don't paper over: 98% *mean* retention hides that only **37.5%** of individual
trajectories keep a pick at-or-above their incumbent, and agreement isn't quality — the noise floor
is 0.86, which is how often a probe agrees with *itself*.

**Reproduce** (clean checkout, dataset path as argument):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest                     # 168 tests, no dataset needed

# routing decisions -> routing_simul.csv beside the dataset
./run_simul.sh path/to/trajectories.jsonl --w-cost 0.10 --w-performance 0.90

# the headline cost figure, against every baseline policy
PYTHONPATH=src .venv/bin/python research/legacy/benchmark.py path/to/trajectories.jsonl
```

~17 s for 1,000 trajectories. Full design and every number above:
**[docs/FULL_REPORT.md](docs/FULL_REPORT.md)**.

---

## Run

The dataset is JSONL — one object per line, with `model`, `input`, `tools`. Pass a file, a
directory of them, or nothing (defaults to `data/`):

```bash
./run_simul.sh path/to/trajectories.jsonl    # one export
./run_simul.sh path/to/exports/              # every *.jsonl in a directory
./run_simul.sh --limit 5 --verbose           # 5 trajectories, full scoreboard each
./run_simul.sh --min-gain 0.05               # only switch on a real improvement
./run_simul.sh --help                        # everything
```

`--verbose` prints every candidate ranked by final score, with `price_score` and
`performance_score` beside it and the incumbent marked — the whole board behind each verdict.

## Configure

Defaults for every run live in [`cheapy.yaml`](cheapy.yaml):

```yaml
W_COST: 0.5          # weight on price_score — higher = cheaper picks
W_PERFORMANCE: 0.5   # weight on performance_score — higher = more capable picks
BETA: 3.0            # capability conformity weighting, w_i = prior_i ** BETA
CACHE_HIT_RATE: 1.0  # assumed prefix-cache hit fraction, 0–1
VERBOSE: false       # print the full scoreboard for every trajectory
```

Only the *ratio* `W_COST : W_PERFORMANCE` matters — they're normalized by their sum. Each key has a
flag (`--w-cost`, `--w-performance`, `--beta`, `--cache-hit-rate`, `--verbose` / `--quiet`).
**Precedence: flag > `cheapy.yaml` > default.** `--config other.yaml` reads a different file.

## Layout

```
src/cheapy/     the router: cli, config, models/, preprocessing/, routing/, capability/
research/       how the numbers were made: capability_fitting/, data_exploration/, legacy/
docs/           FULL_REPORT.md + the capability-model and data-exploration specs
data/  tests/
```

> The dataset is challenge use only, no redistribution (`data/LICENSE`). `*.jsonl` is gitignored;
> keep it that way.
