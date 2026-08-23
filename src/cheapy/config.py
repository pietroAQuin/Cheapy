"""Simulation settings — the one place `cheapy.yaml` is read.

The router itself takes every knob as a call argument (`route(..., w_cost=...)`); this
module exists so the *CLI* has a file to read defaults from instead of hardcoding them.
Nothing under `cheapy/routing/` or `cheapy/capability/` imports it, which keeps those
stages pure functions of their arguments and keeps the tests free of a config file.

Precedence, resolved by `SimulationConfig.resolve()`:

    command-line flag  >  cheapy.yaml  >  the DEFAULT_* constants below

`cheapy.yaml` lives at the repo root and uses SHOUTING keys (`W_COST`, `BETA`, ...) to
match the names the report and the CLI help use. A missing file is not an error — the
built-in defaults are the shipped operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "cheapy.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / CONFIG_NAME

#: Equal weighting: the midpoint of the cost/quality frontier, not a tuned optimum.
DEFAULT_W_COST = 0.5
DEFAULT_W_PERFORMANCE = 0.5

#: Conformity weighting of the capability model (`w_i = prior_i ** BETA`). Mirrors
#: `cheapy.capability.capability_model.DEFAULT_BETA`; kept as a literal so a missing
#: config file never depends on import order between the two modules.
DEFAULT_BETA = 3.0

#: Compression on price_score's cheapest/cost ratio (`ratio ** PRICE_EXPONENT`). Mirrors
#: `cheapy.routing.price_model.DEFAULT_PRICE_EXPONENT`, kept as a literal for the same
#: import-order reason as DEFAULT_BETA above.
DEFAULT_PRICE_EXPONENT = 0.25

#: Optimistic prefix-cache bound — see `cheapy/preprocessing/trajectory_analyzer.py`.
DEFAULT_CACHE_HIT_RATE = 1.0

DEFAULT_VERBOSE = False

#: config key -> field name. The YAML keys are the public names; the dataclass fields are
#: exactly the keyword arguments the router and the analyzer take.
_KEYS = {
    "W_COST": "w_cost",
    "W_PERFORMANCE": "w_performance",
    "BETA": "beta",
    "PRICE_EXPONENT": "price_exponent",
    "CACHE_HIT_RATE": "cache_hit_rate",
    "VERBOSE": "verbose",
}

__all__ = ["SimulationConfig", "load_config", "CONFIG_NAME", "DEFAULT_CONFIG_PATH"]


@dataclass(frozen=True)
class SimulationConfig:
    """One resolved set of simulation settings."""

    w_cost: float = DEFAULT_W_COST
    w_performance: float = DEFAULT_W_PERFORMANCE
    beta: float = DEFAULT_BETA
    price_exponent: float = DEFAULT_PRICE_EXPONENT
    cache_hit_rate: float = DEFAULT_CACHE_HIT_RATE
    verbose: bool = DEFAULT_VERBOSE

    def resolve(self, **overrides: Any) -> SimulationConfig:
        """Return a copy with every non-`None` override applied.

        `None` means "the flag was not given", which is why the CLI's argparse defaults
        are all `None` rather than the numbers — otherwise a default would silently
        outrank the config file.
        """
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"unknown setting(s): {', '.join(sorted(unknown))}")
        merged = {f.name: getattr(self, f.name) for f in fields(self)}
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return SimulationConfig(**merged)

    def validate(self) -> SimulationConfig:
        """Reject values the scoring stages cannot interpret, at the boundary."""
        if self.w_cost < 0 or self.w_performance < 0:
            raise ValueError("W_COST and W_PERFORMANCE must be >= 0")
        if self.w_cost + self.w_performance <= 0:
            raise ValueError("W_COST + W_PERFORMANCE must be > 0 (they are normalized by their sum)")
        if self.beta < 0:
            raise ValueError(f"BETA must be >= 0, got {self.beta}")
        if self.price_exponent <= 0:
            raise ValueError(f"PRICE_EXPONENT must be > 0, got {self.price_exponent}")
        if not 0.0 <= self.cache_hit_rate <= 1.0:
            raise ValueError(f"CACHE_HIT_RATE must be in [0, 1], got {self.cache_hit_rate}")
        return self


def _coerce(key: str, value: Any) -> Any:
    """YAML gives us the right types already; this only guards hand-edited strings."""
    if key == "VERBOSE":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} in {CONFIG_NAME} must be a number, got {value!r}") from exc


def load_config(path: Path | str | None = None) -> SimulationConfig:
    """Read `cheapy.yaml` (or `path`) into a `SimulationConfig`.

    A missing default config falls back to the built-in defaults; a missing *explicit*
    path is an error, because asking for a file that is not there is a typo, not a
    choice. Unknown keys are an error too — a silently ignored `W_PRICE:` would look
    like the router disregarding your settings.
    """
    explicit = path is not None
    config_path = Path(path) if explicit else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        if explicit:
            raise FileNotFoundError(f"config file not found: {config_path}")
        return SimulationConfig()

    import yaml  # local import: only the CLI path needs PyYAML installed

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must be a mapping of KEY: value")

    unknown = set(raw) - set(_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) in {config_path}: {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(_KEYS)}"
        )

    values = {_KEYS[key]: _coerce(key, value) for key, value in raw.items() if value is not None}
    return SimulationConfig(**values).validate()
