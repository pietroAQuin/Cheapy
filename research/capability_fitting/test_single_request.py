#!/usr/bin/env python3
"""One-off sanity check, not part of the offline pipeline proper: pick a real trajectory,
cut it mid-flight, send **one** query to a single model, and print enough to judge by eye
whether the response makes sense given the prefix it continued.

Nothing here is persisted to the elicitation store (`store/responses.jsonl`) — this is a
manual look, not a sampled step, and must not be picked up by `train.py` as one.

    python -m research.capability_fitting.test_single_request --model gpt-5.6-luna

Requires `.env` (see `.env.example`) and `pip install -r requirements-elicit.txt`.
"""

from __future__ import annotations

import argparse
import json
import random

from research.capability_fitting.canonical import to_canonical
from research.capability_fitting.clients import client_for
from research.capability_fitting.env import load_keys
from research.capability_fitting.parser import parse_response
from research.capability_fitting.sampler import sample_cut
from cheapy.preprocessing.trajectory_analyzer import analyze


def _prefix_summary(prefix_items: list[dict]) -> str:
    """A few human-readable lines of context — the last user-visible turn and the tool
    activity right before the cut — so the response can be sanity-checked against what
    the model was actually asked to continue."""
    lines: list[str] = []
    for item in prefix_items[-6:]:
        kind = item.get("type") or "message"
        if kind == "message":
            role = item.get("role")
            content = item.get("content")
            text = content if isinstance(content, str) else "".join(
                p.get("text", "") for p in (content or []) if isinstance(p, dict) and p.get("type") == "input_text"
            )
            lines.append(f"  [{role}] {text[:200]!r}")
        elif kind in ("function_call", "custom_tool_call"):
            lines.append(f"  [tool_call] {item.get('name')}({str(item.get('arguments') or item.get('input'))[:150]})")
        elif kind in ("function_call_output", "custom_tool_call_output"):
            lines.append(f"  [tool_output] {str(item.get('output'))[:150]!r}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--data", default="data/trajectories_v1_01.jsonl")
    parser.add_argument("--seed", type=int, default=0, help="which trajectory/cut to draw")
    args = parser.parse_args()

    anthropic_key, openai_keys = load_keys()
    if args.model.startswith("claude") and not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — copy .env.example to .env and fill it in")
    if args.model.startswith("gpt") and not openai_keys:
        raise SystemExit("no OPENAI_API_KEY[_2/_3] set — copy .env.example to .env and fill it in")

    with open(args.data, encoding="utf-8") as handle:
        lines = handle.readlines()
    rng = random.Random(args.seed)
    order = list(range(len(lines)))
    rng.shuffle(order)

    sample = None
    trajectory_id = None
    for trajectory_id in order:
        record = json.loads(lines[trajectory_id])
        sample = sample_cut(trajectory_id, record, rng)
        if sample is not None:
            break
    if sample is None:
        raise SystemExit("no eligible trajectory found — try a different --seed")

    print(f"[test] trajectory #{trajectory_id}, served_model={sample.served_model} (diagnostic only)")
    print(f"[test] cut at call {sample.step_index}, prefix ~{sample.prefix_token_count} tokens")
    print("[test] last few turns before the cut:")
    print(_prefix_summary(sample.prefix_items))

    trajectory = analyze(
        {"model": sample.served_model, "input": sample.prefix_items, "tools": sample.tools},
        id=trajectory_id,
    )
    print(f"[test] prefix Trajectory: {trajectory.total_calls} calls, toolset_size={trajectory.toolset_size}")

    canonical = to_canonical(sample.prefix_items, sample.tools)
    client = client_for(args.model, anthropic_key=anthropic_key, openai_keys=openai_keys)

    accepts_temp0 = client.probe_temperature(args.model)
    temperature = 0.0 if accepts_temp0 else None
    print(f"[test] temperature: {'0.0' if accepts_temp0 else 'omitted (model rejects it)'}")

    result = client.query(args.model, canonical, temperature)
    print("\n[test] === RAW RESPONSE ===")
    print(json.dumps(result.raw, indent=2)[:4000])

    tools = {t.name: t for t in canonical.tools}
    action = parse_response(args.model, result.raw, tools)
    print(f"\n[test] === PARSED ACTION === type={action.type.value} tool_names={sorted(action.tool_names)}")
    print("[test] this is a manual look, not a store write — nothing was persisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
