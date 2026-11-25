"""
Multivariate Hawkes Process — Cross-Excitation Tape Generator.

Generates N entities where each entity's event firing excites/inhibits
other entities' future firing rates. This tests whether the Tape learns
temporal causal structure (A fires → B fires more).

    λ_i(t) = μ_i + Σ_j Σ_{t_k < t} α_ij · exp(-β_ij · (t - t_k))

Key properties:
    - Cross-entity excitation (α > 0) and inhibition (α < 0)
    - Temporal decay of influence
    - Self-excitation (clustering bursts)
    - Tests: does the model learn WHO triggers WHOM?

The Ogata-thinning loop keeps one flat, time-ordered history of accepted events with a
live-window cursor. Each intensity evaluation is one `exp` + one `bincount` over the live
slice and one `(N,N)@(N,)` matvec, instead of a Python loop over source entities (N numpy
calls per proposal) or over past events. Measured on this box: 5.3x (M=16) and 10x (M=64)
at 20k events, 9.8 s for 200k events at M=64 (the old loop needed minutes). The sampled
entity sequence is identical; timestamps differ only in the last ULP of some values (a
different summation order), the same accepted practice as the previous rewrite.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from data.generators.tape_arrays import build_tape_arrays as build_tape_arrays_vec

_CACHE_VERSION = 1


def _cache_key(kwargs: dict) -> str:
    payload = json.dumps({"v": _CACHE_VERSION, **{
        k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def cached_hawkes_tape(**kwargs) -> dict:
    """``generate_hawkes_tape`` with an optional on-disk cache.

    Set ``TAPEWORLD_SYNTH_CACHE=<dir>`` to make generation a one-off per
    (parameters, seed): a pre-generation process fills the cache and training runs
    read the exact arrays back (npz round-trip, dtypes preserved). Unset, this is the
    plain generator.
    """
    root = os.environ.get("TAPEWORLD_SYNTH_CACHE")
    if not root:
        return generate_hawkes_tape(**kwargs)
    path = Path(root) / "hawkes" / f"{_cache_key(kwargs)}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            tape = {k: z[k] for k in z.files}
        tape["decay_rate"] = float(tape["decay_rate"])
        return tape
    tape = generate_hawkes_tape(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in tape.items()})
    os.replace(tmp, path)
    return tape



def branching_radius(alpha: np.ndarray, decay_rate: float) -> float:
    """Spectral radius of the branching matrix max(α, 0)/β (exponential kernel, uniform β).
    The linear process is stationary iff this is < 1; inhibition only lowers the rectified rate."""
    return float(np.max(np.abs(np.linalg.eigvals(np.maximum(alpha, 0.0) / decay_rate))))


def generate_hawkes_tape(
    n_entities: int = 8,
    n_events: int = 80_000,
    base_rate_range: tuple[float, float] = (0.5, 5.0),
    excitation_strength: float = 0.3,
    decay_rate: float = 1.0,
    sparsity: float = 0.6,
    seed: int = 42,
    uncoupled: bool = False,
    common_drive: bool = False,
    nonneg: bool = False,
    branching_ratio: float | None = None,
) -> dict:
    """Generate a multivariate Hawkes process tape.

    Note: per-entity event counts are dominated by excitation connectivity, not
    base rate — an entity linked to the active network fires hundreds of times even
    with μ ≈ 0.01, while an unlinked one can stay near zero.  To move entities into
    the starved-but-learnable band (2–100 events), widen ``base_rate_range`` rather
    than rescaling individual rates; measured counts are the ground truth.

    ``common_drive`` (SPEC_PAPER_NOV E19): three entities, a driver (0) exciting two followers
    (1, 2) with no follower–follower link, self-excitation kept: base rates (1.0, 0.2, 0.2),
    α_10 = α_20 = 0.6, no rare entities.  The self-excitation draws are the coupled stream's.
    α is lower triangular, so the process is stable whatever the drive.  Brillinger et al. (1976):
    the followers are associated only through the driver, so a pairwise follower–follower kernel
    should vanish while the driver is observed and appear when it is not.

    ``nonneg`` (E1d): off-diagonal α ~ U(0, excitation_strength) instead of U(−0.3, ·), from the
    same number of draws (base rates, mask, self-excitation and rare entities equal the default
    stream's at that seed); no inhibition, so the intensity is never rectified and the pair terms
    cannot cancel.  ``branching_ratio`` r: scale α (diagonal included) so the spectral radius of the
    branching matrix max(α, 0)/β is exactly r, the same distance from criticality (r = 1) at every
    seed.  The default stream reaches 0.83 (M = 16, strength 0.3); strength 0.6 and M = 64 are
    supercritical (1.2–2.4).  ``uncoupled`` zeroes the off-diagonal after the scaling, so its
    self-excitation equals the coupled stream's.

    Returns dict with:
        timestamps, entity_ids, values (intensity at fire time),
        base_rates, excitation_matrix, rare_idx
    """
    rng = np.random.RandomState(seed)

    # Per-entity base rates (log-uniform).
    base_rates: np.ndarray = np.exp(
        rng.uniform(np.log(base_rate_range[0]), np.log(base_rate_range[1]),
                    size=n_entities)
    )

    # Excitation matrix: α_ij = how much entity j excites entity i.
    # Sparse, asymmetric, with some inhibition.
    alpha = rng.uniform(0.0 if nonneg else -0.3, excitation_strength, size=(n_entities, n_entities))
    mask = rng.uniform(size=(n_entities, n_entities)) > sparsity
    alpha = alpha * mask.astype(np.float64)
    if uncoupled and branching_ratio is None:
        # E1b negative control: no cross-entity excitation. Every draw above still happens, so the
        # base rates, self-excitation and rare entities equal the coupled stream's at this seed.
        alpha = alpha * np.eye(n_entities)
    # Self-excitation (clustering).
    np.fill_diagonal(alpha, rng.uniform(0.1, 0.5, size=n_entities))

    # Rare entities: reduce base rate for ~25%.
    n_rare = max(1, n_entities // 4)
    rare_idx = rng.choice(n_entities, size=n_rare, replace=False)
    base_rates[rare_idx] *= rng.uniform(0.05, 0.2, size=n_rare)
    if common_drive:
        if n_entities != 3:
            raise ValueError("common_drive is a three-entity stream (driver 0, followers 1 and 2)")
        self_exc = np.diag(alpha).copy()
        alpha = np.zeros((3, 3))
        alpha[1, 0] = alpha[2, 0] = 0.6
        np.fill_diagonal(alpha, self_exc)
        base_rates = np.array([1.0, 0.2, 0.2])
        rare_idx = np.zeros(0, dtype=np.int64)
        if uncoupled:
            alpha = np.diag(self_exc)
    if branching_ratio is not None:
        alpha = alpha * (branching_ratio / branching_radius(alpha, decay_rate))
        if uncoupled:
            alpha = np.diag(np.diag(alpha))

    # Decay rate is uniform across pairs (beta_ij = decay_rate), so the kernel
    # contribution of source entity j is alpha[:, j] · Σ_k exp(-decay_rate·dt_k).

    # --- Thinning algorithm (Ogata) ---
    timestamps: list[float] = []
    entity_ids: list[int] = []
    intensities: list[float] = []  # intensity at fire time (the "value")

    # Flat history of accepted events (time-ordered) with a live-window cursor.  The
    # kernel sum is one exp + one bincount over the live slice and one (N,N)@(N,) matvec:
    #   λ_i(t) = μ_i + Σ_j α_ij Σ_{k live} exp(-β (t − τ_k))
    # This replaces the per-source-entity Python loop (N numpy calls per proposal) with
    # two array calls per proposal; the sampled stream is the same up to the last ULP of
    # some timestamps (the per-entity sum now runs through bincount/BLAS, a different
    # summation order — the same accepted practice as the previous rewrite).
    window = 10.0 / decay_rate
    ev_time = np.empty(n_events, dtype=np.float64)
    ev_ent = np.empty(n_events, dtype=np.int32)
    n_ev = 0
    start = 0

    def intensity_at(t: float) -> np.ndarray:
        """λ_i(t) = μ_i + Σ_j α_ij Σ_{live t_k of j} exp(-β_ij (t − t_k))."""
        nonlocal start
        if start < n_ev:
            start += int(np.searchsorted(ev_time[start:n_ev], t - window, side="left"))
        if start >= n_ev:
            return base_rates.copy()
        ksum = np.bincount(ev_ent[start:n_ev],
                           weights=np.exp(-decay_rate * (t - ev_time[start:n_ev])),
                           minlength=n_entities)
        return base_rates + alpha @ ksum

    t = 0.0
    n_collected = 0
    max_time = n_events * 2.0 / base_rates.sum()  # rough upper bound

    while n_collected < n_events and t < max_time * 10:
        lambdas = np.maximum(intensity_at(t), 1e-10)
        lambda_total = float(lambdas.sum())

        # Sample next event time (exponential).
        t += float(rng.exponential(1.0 / lambda_total))

        # Thinning: accept with probability lambda_total_new / lambda_total.
        lambdas_new = np.maximum(intensity_at(t), 1e-10)
        lambda_total_new = float(lambdas_new.sum())

        if rng.uniform() < lambda_total_new / lambda_total:
            # Accept. Choose which entity fires.
            probs = lambdas_new / lambda_total_new
            entity = int(rng.choice(n_entities, p=probs))

            timestamps.append(t)
            entity_ids.append(entity)
            intensities.append(float(lambdas_new[entity]))

            ev_time[n_ev] = t
            ev_ent[n_ev] = entity
            n_ev += 1
            n_collected += 1

    timestamps_arr = np.array(timestamps[:n_events], dtype=np.float64)
    entity_ids_arr = np.array(entity_ids[:n_events], dtype=np.int32)
    intensities_arr = np.array(intensities[:n_events], dtype=np.float32)

    return {
        "timestamps": timestamps_arr,
        "entity_ids": entity_ids_arr,
        "values": intensities_arr,
        "base_rates": base_rates,
        "excitation_matrix": alpha,
        "decay_rate": decay_rate,
        "fire_rates": base_rates,  # compatibility
        "rare_idx": rare_idx,
        "theta": base_rates,  # compatibility
        "sigma": np.zeros(n_entities),  # compatibility
        "coupling": alpha,  # for embedding correlation
    }


def build_hawkes_tape_arrays(
    tape: dict,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict:
    """Convert Hawkes tape into training arrays."""
    # Vectorised in data/generators/tape_arrays.py (identical output to the old per-event loops).
    # Log-intensity is the value feature and the target (more stable than raw intensity).
    # ``tape["timing_only"]`` (E1d): the value column is a constant, so a token carries only its
    # entity and clocks; the target is still log λ.  log λ_m sums every entity's past, so with the
    # value in the input a per-entity sub-tape can read the summed cross-entity drive off its own
    # tokens; without it, only the event times of the other entities reveal that drive.
    log_values = np.log(np.maximum(np.asarray(tape["values"]), 1e-6)).astype(np.float32)
    if tape.get("timing_only", False):
        return build_tape_arrays_vec(tape, horizons, feature_values=np.zeros_like(log_values),
                                     target_values=log_values)
    if "value_noise" in tape:      # E23 (b): noisy mark in the input, clean log λ as the target
        return build_tape_arrays_vec(tape, horizons,
                                     feature_values=log_values + np.asarray(tape["value_noise"], dtype=np.float32),
                                     target_values=log_values)
    return build_tape_arrays_vec(tape, horizons, feature_values=log_values)


if __name__ == "__main__":
    tape = generate_hawkes_tape(n_entities=8, n_events=5000, seed=42)
    arrays = build_hawkes_tape_arrays(tape)
    counts = np.bincount(tape["entity_ids"], minlength=8)
    print(f"Events: {len(tape['timestamps'])}")
    for i in range(8):
        r = " (rare)" if i in tape["rare_idx"] else ""
        print(f"  entity {i}: {counts[i]:5d} events, μ={tape['base_rates'][i]:.3f}{r}")
    print(f"Features: {arrays['features'].shape}")
    print(f"Excitation matrix:\n{tape['excitation_matrix'].round(3)}")
