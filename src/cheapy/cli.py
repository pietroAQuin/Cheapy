#!/usr/bin/env python3
"""Routing simulation — replays every logged trajectory through the router.

For each JSONL line of the dataset: parse it into a `Trajectory`, build a
fresh candidate pool, run the docs/FULL_REPORT.md §2 scoring chain (price -> performance ->
weighted final score), and record the verdict — HOLD, or CHANGE TO <model>.

Writes one row per trajectory to `routing_simul.csv` beside the dataset and prints a short
overview. Nothing is written between stages: the simulation is one pass and the CSV
is its only artifact.

Settings come from `cheapy.yaml` at the repo root (W_COST, W_PERFORMANCE, BETA,
CACHE_HIT_RATE, VERBOSE); every one of them has a flag here that wins for a single run.

Every token figure behind these decisions is an ESTIMATE (no `usage` field in the
export — see docs/FULL_REPORT.md §5), so the costs and savings reported here are estimates too.

    ./run_simul.sh                                  # data/, cheapy.yaml settings
    ./run_simul.sh path/to/export.jsonl             # any dataset, CSV written beside it
    ./run_simul.sh --w-cost 0.7 --w-performance 0.3
    ./run_simul.sh --limit 5 --verbose              # per-trajectory scoreboards
    ./run_simul.sh --limit 50 --min-gain 0.02
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]        # src/
REPO_ROOT = SRC_ROOT.parent                           # repo root
if str(SRC_ROOT) not in sys.path:  # allow `python src/cheapy/cli.py` as well as `-m cheapy.cli`
    sys.path.insert(0, str(SRC_ROOT))

from cheapy.config import SimulationConfig, load_config  # noqa: E402
from cheapy.models.llm import ModelLLM  # noqa: E402
from cheapy.models.trajectory import Trajectory  # noqa: E402
from cheapy.preprocessing.model_list import build_model_list  # noqa: E402
from cheapy.preprocessing.trajectory_analyzer import analyze  # noqa: E402
from cheapy.routing.router import RoutingDecision, route  # noqa: E402

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


def resolve_dataset(path: Path) -> tuple[list[Path], Path]:
    """`(chunks, output_dir)` for a dataset given as either a `.jsonl` file or a directory.

    A file routes just that file and writes the CSV beside it; a directory routes every
    `*.jsonl` in it, sorted, and writes the CSV inside it.
    """
    if path.is_file():
        return [path], path.parent
    if path.is_dir():
        chunks = sorted(path.glob("*.jsonl"))
        if not chunks:
            raise SystemExit(f"no *.jsonl chunks found in {path}")
        return chunks, path
    raise SystemExit(f"dataset not found: {path}")


def iter_trajectories(chunks: list[Path], limit: int | None = None, *, cache_hit_rate: float):
    """Yield `(Trajectory, chunk_name)` for every JSONL line of every chunk.

    Files are read in the order given and ids come from one counter across all of them, so
    an id identifies a trajectory within the run — `analyze_file` restarts at 0 per file,
    which would collide the moment a second chunk lands. `Trajectory` forbids extra
    fields, so the chunk name travels alongside rather than on the model.
    """

    count = 0
    for chunk in chunks:
        with open(chunk, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and count >= limit:
                    return
                yield analyze(line, count, cache_hit_rate=cache_hit_rate), chunk.name
                count += 1


def simulate(
    chunks: list[Path],
    config: SimulationConfig,
    *,
    min_gain: float = 0.0,
    limit: int | None = None,
    on_result=None,
) -> list[tuple[RoutingDecision, Trajectory, str]]:
    """Route every trajectory in `chunks` and return the decisions.

    A fresh `build_model_list()` per trajectory is deliberate, not wasteful: the scoring
    stages enrich `ModelLLM` objects in place (docs/FULL_REPORT.md §2), so a pool shared across
    trajectories would hand each one the previous trajectory's scores.

    `on_result` is called with each `(decision, trajectory, chunk_name)` as it is
    produced — that is how `--verbose` prints a scoreboard per trajectory instead of
    waiting for the whole export to finish.
    """
    results: list[tuple[RoutingDecision, Trajectory, str]] = []
    for trajectory, chunk_name in iter_trajectories(
        chunks, limit=limit, cache_hit_rate=config.cache_hit_rate
    ):
        models: list[ModelLLM] = build_model_list()
        decision = route(
            trajectory,
            models,
            w_cost=config.w_cost,
            w_performance=config.w_performance,
            beta=config.beta,
            price_exponent=config.price_exponent,
            min_gain=min_gain,
        )
        result = (decision, trajectory, chunk_name)
        results.append(result)
        if on_result is not None:
            on_result(*result)
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


def print_scoreboard(decision: RoutingDecision, trajectory: Trajectory) -> None:
    """The full candidate ranking behind one verdict — what `--verbose` prints.

    Every candidate the router scored, best `final_score` first, with the two numbers
    that produced it. The incumbent is marked `<-- served`, so a HOLD reads as "the
    marked row is on top" and a CHANGE reads as "these rows beat the marked one".
    """
    header = (
        f"  trajectory {decision.trajectory_id}  "
        f"({trajectory.total_calls} calls, {trajectory.total_tokens:,} est. tokens)  "
        f"-> {decision.label}"
    )
    print(f"\n{header}")
    print(f"  {'':<3} {'model':<20} {'final':>8} {'price':>8} {'perf':>8}")
    print(f"  {'-' * 3} {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")
    for position, row in enumerate(decision.scoreboard, start=1):
        marker = "  <-- served" if row.is_served else ""
        print(
            f"  {position:<3} {row.name:<20} {row.final_score:>8.4f} "
            f"{row.price_score:>8.4f} {row.performance_score:>8.4f}{marker}"
        )
    if decision.unscored:
        print(f"      unscored: {', '.join(decision.unscored)}")


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
        epilog="Defaults come from cheapy.yaml; any flag given here overrides it for this run.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=None,
        help="a .jsonl export, or a directory of them; the CSV is written beside it "
        "(default: data/)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="same thing, named. Ignored if a dataset is given positionally",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="settings file to read (default: cheapy.yaml at the repo root)",
    )
    # Every setting below defaults to None on purpose: None means "flag not given", which
    # is what lets cheapy.yaml win over a built-in default but lose to an explicit flag.
    parser.add_argument(
        "--w-cost", type=float, default=None, help="weight on price_score [W_COST]"
    )
    parser.add_argument(
        "--w-performance",
        type=float,
        default=None,
        help="weight on performance_score [W_PERFORMANCE]",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="conformity weighting of the capability model, w_i = prior_i ** beta [BETA]",
    )
    parser.add_argument(
        "--price-exponent",
        type=float,
        default=None,
        help="compression on price_score's cheapest/cost ratio, ratio ** price_exponent; "
        "1.0 = no compression [PRICE_EXPONENT]",
    )
    parser.add_argument(
        "--cache-hit-rate",
        type=float,
        default=None,
        help="assumed prefix-cache hit fraction, 0-1 [CACHE_HIT_RATE]",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=None,
        help="print the full scoreboard for every trajectory [VERBOSE]",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="summary only, even if VERBOSE is true in the config",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="final_score improvement a challenger must beat before a switch is taken "
        "(default: 0.0)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="route at most N trajectories (quick runs)"
    )
    args = parser.parse_args(argv)
    chunks, output_dir = resolve_dataset(args.dataset or args.data_dir or DEFAULT_DATA_DIR)

    try:
        config = load_config(args.config).resolve(
            w_cost=args.w_cost,
            w_performance=args.w_performance,
            beta=args.beta,
            price_exponent=args.price_exponent,
            cache_hit_rate=args.cache_hit_rate,
            verbose=args.verbose,
        ).validate()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"weights: cost {config.w_cost} / performance {config.w_performance}   "
        f"beta {config.beta}   price-exponent {config.price_exponent}   "
        f"cache-hit-rate {config.cache_hit_rate}   min-gain {args.min_gain}"
    )

    on_result = None
    if config.verbose:
        def on_result(decision, trajectory, _chunk_name):  # noqa: E306
            print_scoreboard(decision, trajectory)

    results = simulate(
        chunks,
        config,
        min_gain=args.min_gain,
        limit=args.limit,
        on_result=on_result,
    )
    output = write_csv(results, output_dir / OUTPUT_NAME)

    print_overview(results)
    print(f"\n  -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
