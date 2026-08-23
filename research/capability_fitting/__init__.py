"""Offline pipeline that fits `src/cheapy/capability/artifacts/pair_model.json`.

This runs **once**, needs provider API keys (`requirements-elicit.txt`, `.env`), and is
not on the router's runtime path — `run_pilot.py` elicits, `train.py` fits, `diagnostics.py`
reports. It imports from `cheapy.capability` (features, priors), never the other way
around: that one-directional edge is what keeps the shipped model key-free.

Design: docs/capability_model.md.
"""
