"""Completing the Claude-vs-Claude agreement block.

After corpus recovery (`logged_action.py`) 21 of the 36 model pairs are measured: the 3
probe-vs-probe pairs, and the 18 logged-vs-probe pairs. The remaining 15 are
Claude-vs-Claude, and they are **structurally unobservable** — one model serves a whole
trajectory, so two Anthropic models never produce an action at the same cut point.

### Why not synthesize them from the prior

The obvious alternative — assume agreement decays with distance on the `base_capability`
axis — was tried and is degenerate. Substituting `pair = 1 - (1-A(step))*S(d)` into the
§5.1 score gives

    score(i) = 1 - (1 - A(step)) * C_i,     C_i = sum_j w_j*S(d_ij) / W

`C_i` carries no step term, so the model ranking is **identical on every step**, and `C_i`
is minimised by whichever model sits nearest the weighted centroid of the priors — i.e. it
scores *centrality*, not capability, and penalises the strongest model for being an
outlier. On the real priors it ranks `gpt-5.6-sol` above `claude-opus-5` and
`claude-fable-5` on all 957 steps. That is a benchmark lookup with extra steps, and it
fails the spec's own §4 sanity check by construction.

### What this module does instead

Every model — probe or not — ends up with a **measured** 3-vector of agreements against
the probes. That is a behavioural embedding whose coordinates are observations, not
assumptions. Nyström extension then reads the missing block off those coordinates:

    M_AA ~= M_AO @ pinv(M_OO + lam*I) @ M_AO.T

In words: *two Claude models agree with each other to the extent that they agree with the
same probes.* The inverse de-correlates the landmarks, so two probes that behave alike do
not get counted as two independent pieces of evidence.

### What it cannot do — state this, do not hide it

With 3 probes the completion can express at most **3 dimensions** of behavioural variation.
Simulated against known ground truth (hide the block, reconstruct, compare) the error is
0.000 when agreement really is rank-3, and rises to ~0.07 mean / ~0.26 worst-case by rank 4.
**From inside, we cannot tell which regime we are in** — a 4th probe would raise the cap
and was deliberately declined. So: report `condition_number` and `clamp_rate`, label every
completed cell as extrapolated (`capability_model` does), and never present these 15 cells
with the same confidence as the 21 measured ones. Leave-one-probe-out validates the
machinery on cross-provider cells; nothing available validates the Claude-vs-Claude cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Tikhonov damping on the probe Gram matrix. The probes agree with each other often, which
#: makes `M_OO` near-singular and its inverse a noise amplifier (at 95% mutual agreement the
#: condition number is ~58), so some damping is not optional here.
#:
#: 0.03 is chosen against the noise we actually expect, not against a clean problem. Swept
#: on rank-4 synthetic truth, mean abs error on the completed block:
#:
#:     ridge     noise=0.00  noise=0.02  noise=0.05  noise=0.10
#:     0.0       0.069       0.108       0.141       0.192
#:     0.01      0.078       0.101       0.130       0.186
#:     0.03      0.086       0.088       0.119       0.179
#:     0.10      0.106       0.105       0.111       0.147
#:
#: Undamped wins only in the noiseless case, which is not the case we are in: a single pair
#: observation is very noisy (measured self-agreement is ~0.74, not ~1.0), and while pooling
#: over a model's trajectories shrinks that to roughly 0.02-0.05 standard error for the
#: well-covered models, it stays far larger for the rare ones. Raise it if the per-cell
#: standard errors come in high.
DEFAULT_RIDGE = 3e-2


@dataclass(frozen=True)
class Completion:
    """The completed block plus the diagnostics that say how far to trust it."""

    matrix: np.ndarray  # (n_unmeasured, n_unmeasured), clamped to [0, 1]
    condition_number: float
    clamp_rate: float
    n_probes: int

    @property
    def rank_cap(self) -> int:
        """Ceiling on the behavioural dimensions this completion can express."""
        return self.n_probes

    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.condition_number > 30:
            notes.append(
                f"probe Gram matrix is ill-conditioned (cond={self.condition_number:.1f}): "
                f"the probes behave too much alike to span {self.n_probes} directions"
            )
        if self.clamp_rate > 0.05:
            notes.append(
                f"{self.clamp_rate:.1%} of completed cells fell outside [0,1] and were "
                f"clamped — the rank-{self.n_probes} assumption is straining"
            )
        return notes


def nystrom_complete(
    probe_block: np.ndarray, cross_block: np.ndarray, ridge: float = DEFAULT_RIDGE
) -> Completion:
    """Complete the unmeasured-vs-unmeasured block.

    `probe_block` is `M_OO` (n_probes x n_probes): agreement among the probes, **with the
    measured self-agreement on the diagonal, not 1.0**. None of the three models accepts
    `temperature=0`, so a model does not reproduce its own action reliably; the self-pair
    measurement is the honest diagonal and using 1.0 instead would overstate how much
    structure the landmarks pin down.

    `cross_block` is `M_AO` (n_unmeasured x n_probes): each unmeasured model's agreement
    against each probe.
    """
    probe_block = np.asarray(probe_block, dtype=float)
    cross_block = np.asarray(cross_block, dtype=float)
    n_probes = probe_block.shape[0]

    damped = probe_block + ridge * np.eye(n_probes)
    raw = cross_block @ np.linalg.pinv(damped) @ cross_block.T

    clamped = np.clip(raw, 0.0, 1.0)
    off_diagonal = ~np.eye(raw.shape[0], dtype=bool)
    n_off = int(off_diagonal.sum())
    clamp_rate = (
        float((raw[off_diagonal] != clamped[off_diagonal]).sum()) / n_off if n_off else 0.0
    )

    return Completion(
        matrix=clamped,
        condition_number=float(np.linalg.cond(damped)),
        clamp_rate=clamp_rate,
        n_probes=n_probes,
    )
