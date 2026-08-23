#!/usr/bin/env python3
"""Cost-only benchmark: router vs. baseline policies, over the real export.

Five policies are compared on total estimated USD across every trajectory's *next
call* (the same decision point `src/cheapy/routing/` scores):

  - **router**         — `src/cheapy/routing/router.py`'s weighted price/performance pick,
                        at a single fixed weight (`DEFAULT_WEIGHT`). Cost comes
                        straight out of `price_model.py`'s cache-aware cost formula.
                        The router's actual pick still depends on `performance_score`
                        (`src/cheapy/routing/performance_model.py`, a real fitted
                        pairwise-conformity model, not a mock -- but still a proxy,
                        not ground-truth quality) even though this script only
                        reports the resulting cost.
  - **always-cheapest** — always route to the cheapest model in the pool.
  - **always-strongest** — always route to `claude-fable-5`, regardless of the log.
  - **always hold**     — always HOLD: route to `trajectory.served_model`, i.e.
                        today's production behavior with no router at all.
  - **random routing**  — expected cost of picking a model uniformly at random.

This is a pure simulation over `src/cheapy/preprocessing`/`src/cheapy/routing` -- no API calls, no
network cost, runs offline once the export is on disk (`export/*.jsonl`; falls back
to `data/*.jsonl` per the README's original path).

Usage: python research/legacy/benchmark.py [export_dir_or_file] [--limit N]
Outputs: results/benchmark_trajectories.csv, results/total_cost.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cheapy.models.llm import ModelLLM
from cheapy.preprocessing.model_list import build_model_list
from cheapy.preprocessing.trajectory_analyzer import analyze_file
from cheapy.routing import router as model_stage
from cheapy.routing.performance_model import score_performance
from cheapy.routing.price_model import estimate_costs, score_price

STRONGEST_NAME = "claude-fable-5"  # highest BASE_CAPABILITY prior in the pool (priors.py)
CHEAPEST_NAME = "gpt-5.6-luna"     # lowest listed input rate in the pool
DEFAULT_WEIGHT = 0.1  # w_cost for the router's single fixed operating point (w_perf =
# 1 - this). performance_score's dynamic range in this data is narrow (~0.74-0.80) next
# to price_score's near-unit range, so the router's behavior barely changes past
# w_cost~0.15 -- 0.1 sits in the part of that range where it still meaningfully listens
# to performance_score while beating the incumbent-only baseline on cost, rather than
# collapsing to the same picks as always-cheapest (w_cost=1.0) or drifting into
# "quality-only" territory that can cost more than doing nothing (w_cost<0.05).


def find_export(arg: str | None) -> list[Path]:
    """Resolve the trajectories file(s). Prefers `export/*.jsonl` (docs/FULL_REPORT.md
    §3); falls back to `data/*.jsonl` for the README's originally-documented path.

    Returns every `*.jsonl` chunk in the resolved directory, sorted -- not just the
    first one. A single file arg is returned as a one-element list either way, so
    callers always iterate a list.
    """
    if arg:
        path = Path(arg)
        if path.is_file():
            return [path]
        candidates = sorted(path.glob("*.jsonl"))
        if candidates:
            return candidates
        sys.exit(f"no *.jsonl found in {path}")
    for directory in (Path("export"), Path("data")):
        candidates = sorted(directory.glob("*.jsonl"))
        if candidates:
            return candidates
    sys.exit("no export found — pass a path, or place a chunk under export/ or data/")


def score_trajectory(trajectory, base_models: list[ModelLLM]) -> dict:
    """Run both scoring stages on a fresh copy of the pool, then read off each
    policy's cost for this trajectory's next call.

    Both scoring stages run regardless -- the router's pick at DEFAULT_WEIGHT depends
    on performance_score even though this script only reports cost.
    """
    models = [m.model_copy(deep=True) for m in base_models]
    score_price(trajectory, models)
    score_performance(trajectory, models)
    costs = estimate_costs(trajectory, models)

    served = trajectory.served_model
    random_cost = sum(costs.values()) / len(models)

    model_stage.aggregate_scores(
        models, w_cost=DEFAULT_WEIGHT, w_performance=round(1 - DEFAULT_WEIGHT, 2)
    )
    decision = model_stage.decide(trajectory, models)

    return {
        "id": trajectory.id,
        "served_model": served,
        "total_calls": trajectory.total_calls,
        "strongest_cost": costs[STRONGEST_NAME],
        "cheapest_cost": costs[CHEAPEST_NAME],
        "incumbent_cost": costs[served],
        "random_cost": random_cost,
        "router_cost": costs[decision.top_model],
        "router_decision": decision.label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default=None, help="export dir or .jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N trajectories")
    args = parser.parse_args()

    chunks = find_export(args.export)
    base_models = build_model_list()
    print(f"reading {', '.join(c.name for c in chunks)} ...")

    # One id counter across every chunk, not per-file: analyze_file/analyze() number a
    # trajectory by its line position within a single file, so reading two chunks
    # independently would hand out id 0-999 twice and silently collide the moment a
    # second chunk (e.g. trajectories_v1_02.jsonl) lands next to the first. Re-number
    # here instead, same fix cli.py's iter_trajectories already applies.
    results = []
    count = 0
    for chunk in chunks:
        for trajectory in analyze_file(chunk):
            if args.limit and count >= args.limit:
                break
            trajectory.id = count
            results.append(score_trajectory(trajectory, base_models))
            count += 1
            if count % 200 == 0:
                print(f"  scored {count} trajectories...")
        if args.limit and count >= args.limit:
            break

    n = len(results)
    print(f"scored {n} trajectories\n")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    write_per_trajectory_csv(results_dir / "benchmark_trajectories.csv", results)
    print_summary(results, n)
    plot_total_cost(results_dir / "total_cost.png", results)


def write_per_trajectory_csv(path: Path, results: list[dict]) -> None:
    """Headline per-trajectory table at DEFAULT_WEIGHT, for a quick eyeball diff.
    Covers every trajectory in the export."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "served_model", "total_calls",
            "strongest_cost_usd", "cheapest_cost_usd", "incumbent_cost_usd",
            "router_cost_usd", "router_decision",
        ])
        for row in results:
            writer.writerow([
                row["id"], row["served_model"], row["total_calls"],
                f"{row['strongest_cost']:.6f}", f"{row['cheapest_cost']:.6f}",
                f"{row['incumbent_cost']:.6f}", f"{row['router_cost']:.6f}", row["router_decision"],
            ])
    print(f"wrote {path}")


def print_summary(results: list[dict], n: int) -> None:
    strongest_cost = sum(r["strongest_cost"] for r in results) / n
    incumbent_cost = sum(r["incumbent_cost"] for r in results) / n
    router_cost = sum(r["router_cost"] for r in results) / n
    hold_rate = sum(1 for r in results if r["router_decision"] == "HOLD") / n

    print(f"--- summary over all {n} trajectories (next-call cost, USD) ---")
    print(f"  incumbent-only (observed policy, today, always HOLD): ${incumbent_cost:.6f} avg/trajectory")
    print(f"  always-strongest (claude-fable-5):                     ${strongest_cost:.6f} avg/trajectory")
    print(f"  router @ w_cost={DEFAULT_WEIGHT}:                                   ${router_cost:.6f} avg/trajectory "
          f"({hold_rate:.0%} HOLD)")
    print(f"  router savings vs incumbent-only:   {100 * (1 - router_cost / incumbent_cost):.1f}%")
    print(f"  router savings vs always-strongest: {100 * (1 - router_cost / strongest_cost):.1f}%")
    print()
    print("NOTE: cost figures are real estimates from price_model.py's cache-aware formula (still "
          "estimates: no `usage` field in the export, tokens are tiktoken-counted, see docs/FULL_REPORT.md §5). "
          "The router's picks still depend on performance_score (src/cheapy/routing/performance_model.py, "
          "a fitted pairwise-conformity proxy, not measured ground truth) even though this summary "
          "only reports cost.")


def plot_total_cost(path: Path, results: list[dict]) -> None:
    """Bar chart: total estimated USD across every trajectory's next call, one bar per
    policy -- no quality axis, just the real cache-aware cost formula summed over the
    export.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = [
        ("random\nrouting", sum(r["random_cost"] for r in results), "#FFFFFF", "#6748FD"),
        ("always-\ncheapest", sum(r["cheapest_cost"] for r in results), "#FFFFFF", "#150079"),
        ("always-\nstrongest", sum(r["strongest_cost"] for r in results), "#FFBD9E", "#150079"),
        ("always\nhold", sum(r["incumbent_cost"] for r in results), "#150079", "#150079"),
        (f"router\n(w_cost={DEFAULT_WEIGHT})", sum(r["router_cost"] for r in results), "#6748FD", "#6748FD"),
    ]
    bars.sort(key=lambda b: b[1])

    labels = [b[0] for b in bars]
    totals = [b[1] for b in bars]
    face_colors = [b[2] for b in bars]
    edge_colors = [b[3] for b in bars]

    # Log scale: totals span ~50x (the whole point of the chart is that spread), so a
    # linear axis squashes the four cheaper bars into an unreadable sliver at the
    # bottom next to always-strongest. A floor of $1 keeps every bar's base finite
    # (log(0) is undefined) without hiding how small the cheap bars really are --
    # exact dollar amounts are annotated on every bar regardless of the axis scale.
    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(bars))
    ax.bar(x, totals, color=face_colors, edgecolor=edge_colors, linewidth=1.5, width=0.62, bottom=1)
    ax.set_yscale("log")
    ax.set_ylim(1, max(totals) * 2.2)

    for i, total in enumerate(totals):
        ax.annotate(f"${total:,.2f}", (i, total), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9.5, color="#150079")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(f"total estimated cost, next call across {len(results)} trajectories (USD, log scale)")
    ax.set_title("Total cost by policy")
    fig.text(0.01, 0.01,
              "cost is a real estimate (price_model.py's cache-aware formula); router uses a fitted-proxy\n"
              "performance_score (performance_model.py); always-cheapest = gpt-5.6-luna, "
              "always-strongest = claude-fable-5",
              fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
