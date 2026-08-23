#!/usr/bin/env python3
"""Routing simulation — replays every logged trajectory through the router.

For each JSONL line under the data directory: parse it into a `Trajectory`, build a
fresh candidate pool, run the README §2 scoring chain (price -> performance ->
weighted final score), and record the verdict — HOLD, or CHANGE TO <model>.

Writes one row per trajectory to `<data-dir>/routing_simul.csv` and prints a short
overview. Nothing is written between stages: the simulation is one pass and the CSV
is its only artifact.

Every token figure behind these decisions is an ESTIMATE (no `usage` field in the
export — see README §5), so the costs and savings reported here are estimates too.

    ./run_simul.sh                                  # whole export, 50/50 weights
    ./run_simul.sh --w-price 0.7 --w-performance 0.3
    ./run_simul.sh --limit 50 --min-gain 0.02
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python app/main.py` as well as `-m app.main`
    sys.path.insert(0, str(REPO_ROOT))

from data_models.model_llm import ModelLLM  # noqa: E402
from data_models.Trajectory import Trajectory  # noqa: E402
from pre_processing.model_list import build_model_list  # noqa: E402
from pre_processing.trajectory_analyzer import analyze  # noqa: E402
from router_models.model import RoutingDecision, route  # noqa: E402
from router_models.performance_model import DEFAULT_BETA  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "data"
OUTPUT_NAME = "routing_simul.csv"

_CSV_COLUMNS = (
    "trajectory_id",
    "source_chunk",
    "served_model",
    "decision",
    "top_model",
    "action",
    "top_final_score",
    "top_price_score",
    "top_performance_score",
    "served_final_score",
    "score_gap",
    "total_calls",
    "total_tokens",
    "ranking",
    "unscored",
)


def iter_trajectories(data_dir: Path, limit: int | None = None):
    """Yield `(Trajectory, chunk_name)` for every JSONL line under `data_dir`.

    Files are read in sorted order and ids come from one counter across all of them, so
    an id identifies a trajectory within the run — `analyze_file` restarts at 0 per file,
    which would collide the moment a second chunk lands. `Trajectory` forbids extra
    fields, so the chunk name travels alongside rather than on the model.
    """
    chunks = sorted(data_dir.glob("*.jsonl"))
    if not chunks:
        raise SystemExit(f"no *.jsonl chunks found in {data_dir}")

    count = 0
    for chunk in chunks:
        with open(chunk, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and count >= limit:
                    return
                yield analyze(line, count), chunk.name
                count += 1


def simulate(
    data_dir: Path,
    *,
    w_price: float,
    w_performance: float,
    beta: float = DEFAULT_BETA,
    min_gain: float = 0.0,
    limit: int | None = None,
) -> list[tuple[RoutingDecision, Trajectory, str]]:
    """Route every trajectory in `data_dir` and return the decisions.

    A fresh `build_model_list()` per trajectory is deliberate, not wasteful: the scoring
    stages enrich `ModelLLM` objects in place (README §2), so a pool shared across
    trajectories would hand each one the previous trajectory's scores.
    """
    results: list[tuple[RoutingDecision, Trajectory, str]] = []
    for trajectory, chunk_name in iter_trajectories(data_dir, limit=limit):
        models: list[ModelLLM] = build_model_list()
        decision = route(
            trajectory,
            models,
            w_price=w_price,
            w_performance=w_performance,
            beta=beta,
            min_gain=min_gain,
        )
        results.append((decision, trajectory, chunk_name))
    return results


def write_csv(results: list[tuple[RoutingDecision, Trajectory, str]], path: Path) -> Path:
    """One row per routed trajectory."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for decision, trajectory, chunk_name in results:
            writer.writerow(
                (
                    decision.trajectory_id,
                    chunk_name,
                    decision.served_model,
                    decision.label,
                    decision.top_model,
                    decision.action,
                    f"{decision.top_final_score:.6f}",
                    f"{decision.top_price_score:.6f}",
                    f"{decision.top_performance_score:.6f}",
                    ""
                    if decision.served_final_score is None
                    else f"{decision.served_final_score:.6f}",
                    f"{decision.score_gap:.6f}",
                    trajectory.total_calls,
                    trajectory.total_tokens,
                    "|".join(decision.ranking),
                    "|".join(decision.unscored),
                )
            )
    return path


def print_overview(results: list[tuple[RoutingDecision, Trajectory, str]]) -> None:
    """Short terminal summary: the HOLD / CHANGE split and where the changes go."""
    total = len(results)
    decisions = [decision for decision, _, _ in results]
    holds = sum(1 for d in decisions if d.action == "HOLD")
    changes = total - holds
    targets = Counter(d.top_model for d in decisions if d.action == "CHANGE")

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "n/a"

    print(f"\n  {total} trajectories routed")
    print(f"    HOLD       {holds:>5}  ({pct(holds)})")
    print(f"    CHANGE TO  {changes:>5}  ({pct(changes)})")

    if targets:
        top_model, top_count = targets.most_common(1)[0]
        print(f"\n  most-changed-to: {top_model}  ({top_count} of {changes} changes)")
        for name, count in targets.most_common():
            print(f"    {name:<20} {count:>5}  ({pct(count)} of all trajectories)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory holding the export chunks; the CSV is written here too",
    )
    parser.add_argument(
        "--w-price", type=float, default=0.5, help="weight on price_score in the final score"
    )
    parser.add_argument(
        "--w-performance",
        type=float,
        default=0.5,
        help="weight on performance_score in the final score",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="conformity weighting of the capability model (w_i = prior_i ** beta)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="final_score improvement a challenger must beat before a switch is taken",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="route at most N trajectories (quick runs)"
    )
    args = parser.parse_args(argv)

    results = simulate(
        args.data_dir,
        w_price=args.w_price,
        w_performance=args.w_performance,
        beta=args.beta,
        min_gain=args.min_gain,
        limit=args.limit,
    )
    output = write_csv(results, args.data_dir / OUTPUT_NAME)

    print(
        f"weights: price {args.w_price} / performance {args.w_performance}   "
        f"beta {args.beta}   min-gain {args.min_gain}"
    )
    print_overview(results)
    print(f"\n  -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
