"""Cheapy — a cost/quality router for Viktor trajectories.

Everything in this package runs offline: no network, no API keys. The four stages are
`preprocessing` (JSONL -> `Trajectory` + candidate pool), `routing` (price and
performance scoring, then the HOLD / CHANGE verdict), `capability` (the fitted model
behind `performance_score`), and `cli` (the simulation over a whole export).

Design and results: docs/FULL_REPORT.md.
"""
