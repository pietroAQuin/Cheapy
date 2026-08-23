"""Calibrating logged actions against elicited ones.

The pivot compares two things produced under different conditions:

- **elicited** — a probe is handed a prefix cold and asked what comes next. It is
  continuing a stranger's work, at whatever sampling settings `clients.probe_temperature`
  established (in practice: provider default, because all three probes reject
  `temperature=0`).
- **logged** — the served model wrote that entire trajectory. Its next action continues its
  own plan and its own conventions, and was sampled by the Viktor harness.

So a measured `pair(opus5_logged, sol_elicited)` mixes two things: how much the two models
genuinely differ, and how much *any* logged action differs from *any* elicited one. `δ` is
the second part.

**Its sign is not predictable.** Home-field idiosyncrasy pushes agreement down; but the
probes read the served model's conventions off the prefix and get steered toward it, which
pushes agreement up. That is exactly why it is measured rather than assumed.

### How it is identified

On probe-served trajectories the same model is visible **both ways**, so the regime is the
only thing that changes:

    pair(m_elicited#1, m_elicited#2)   -> sampling noise alone          (self-pair)
    pair(m_logged,     m_elicited)     -> sampling noise + regime gap
                               delta_m  = the difference

Three independent estimates of one quantity, one per probe. **If they agree, the correction
is justified; if they scatter, that is the data refusing the assumption** — report the
spread and leave the targets uncorrected. `DeltaEstimate.is_consistent` is that test.

The self-pair term is not a formality. None of the probes accepts `temperature=0`, so a
model does not reproduce its own action reliably: measured self-agreement on a 25-step
sample was **0.74**, with an exact-match rate of 72%. That number is the ceiling on any
agreement this pipeline can measure, and it belongs in the writeup next to every reported
pair value.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class DeltaEstimate:
    """Per-probe regime offset, plus the consistency check that licenses using it."""

    per_model: dict[str, float] = field(default_factory=dict)
    noise_floor: dict[str, float] = field(default_factory=dict)
    n_observations: dict[str, int] = field(default_factory=dict)

    #: Spread across probes beyond which the "delta is a property of the regime, not of the
    #: model" assumption is not credible and no correction should be applied.
    consistency_tolerance: float = 0.10

    @property
    def mean(self) -> float:
        return statistics.fmean(self.per_model.values()) if self.per_model else 0.0

    @property
    def spread(self) -> float:
        values = list(self.per_model.values())
        return (max(values) - min(values)) if len(values) > 1 else 0.0

    @property
    def is_consistent(self) -> bool:
        """True when the per-probe estimates agree well enough to justify correcting."""
        return len(self.per_model) > 1 and self.spread <= self.consistency_tolerance

    def correction(self) -> float:
        """The offset to add to logged-vs-elicited targets — 0.0 when inconsistent.

        Refusing to correct on scattered estimates is deliberate: a correction derived from
        an assumption the data just contradicted would be worse than no correction, and it
        would hide the contradiction.
        """
        return self.mean if self.is_consistent else 0.0

    def report(self) -> str:
        lines = ["[calibration] logged-vs-elicited offset (delta):"]
        for model in sorted(self.per_model):
            lines.append(
                f"    {model:<16} delta={self.per_model[model]:+.4f}  "
                f"self-agreement={self.noise_floor.get(model, float('nan')):.4f}  "
                f"n={self.n_observations.get(model, 0)}"
            )
        lines.append(f"    mean={self.mean:+.4f}  spread={self.spread:.4f}")
        lines.append(
            f"    -> {'CONSISTENT, applying correction' if self.is_consistent else 'INCONSISTENT, NOT correcting (reported only)'}"
        )
        return "\n".join(lines)


def estimate_delta(
    logged_vs_elicited: dict[str, list[float]],
    self_pairs: dict[str, list[float]],
    consistency_tolerance: float = 0.10,
) -> DeltaEstimate:
    """Build the offset estimate from per-probe observations.

    `logged_vs_elicited[m]` holds `pair(m_logged, m_elicited)` values on trajectories that
    `m` itself served; `self_pairs[m]` holds `pair(m_elicited#1, m_elicited#2)` from the
    self-pair subsample. Both are measured on the same model, so the difference isolates
    the regime.
    """
    estimate = DeltaEstimate(consistency_tolerance=consistency_tolerance)
    for model, cross in logged_vs_elicited.items():
        same = self_pairs.get(model) or []
        if not cross or not same:
            continue
        floor = statistics.fmean(same)
        estimate.noise_floor[model] = floor
        estimate.per_model[model] = floor - statistics.fmean(cross)
        estimate.n_observations[model] = min(len(cross), len(same))
    return estimate
