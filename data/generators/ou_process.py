"""
Coupled Multivariate Ornstein-Uhlenbeck Process — Asynchronous Tape Generator.

Generates N entities each following an OU process:
    dX_i = θ_i (μ - X_i) dt + σ_i dW_i + Σ_j C_ij (X_j - X_i) dt

Events fire asynchronously at Poisson rates proportional to θ_i.
This creates:
    - Heterogeneous dynamics (different θ, σ per entity)
    - Cross-entity coupling (coupling matrix C)
    - Rare entities (some fire 10-100x less often)
    - An interleaved chronological tape
"""

from __future__ import annotations

import numpy as np

from data.generators.tape_arrays import build_tape_arrays as build_tape_arrays_vec


def generate_coupling_matrix(
    n_entities: int,
    coupling_strength: float = 0.3,
    sparsity: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sparse, asymmetric coupling matrix.

    C_ij > 0 means entity j pulls entity i toward it.
    Diagonal is zero (self-coupling is handled by θ).
    """
    rng = np.random.RandomState(seed + 1000)
    C = rng.uniform(0, coupling_strength, size=(n_entities, n_entities))
    mask = rng.uniform(size=(n_entities, n_entities)) > sparsity
    C = C * mask.astype(np.float64)
    np.fill_diagonal(C, 0.0)
    return C


def generate_ou_tape(
    n_entities: int = 16,
    n_events: int = 100_000,
    coupling_strength: float = 0.3,
    dt_sim: float = 0.001,
    seed: int = 42,
) -> dict:
    """Generate a coupled multivariate OU tape.

    Returns a dict with:
        timestamps:  (n_events,)  float64  — event times
        entity_ids:  (n_events,)  int32    — which entity fired
        values:      (n_events,)  float32  — entity value at event time
        theta:       (n_entities,) float64 — mean reversion speeds
        sigma:       (n_entities,) float64 — volatilities
        coupling:    (n_entities, n_entities) float64 — coupling matrix
        fire_rates:  (n_entities,) float64 — Poisson firing rates
    """
    rng = np.random.RandomState(seed)

    # --- Per-entity parameters (log-uniform for heterogeneity) ---
    # θ spans ~[0.1, 10] — fast vs slow mean reversion
    theta = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=n_entities))
    # σ spans ~[0.05, 5.0] — low vs high volatility
    sigma = np.exp(rng.uniform(np.log(0.05), np.log(5.0), size=n_entities))

    # --- Coupling matrix ---
    C = generate_coupling_matrix(n_entities, coupling_strength, seed=seed)

    # --- Firing rates: proportional to θ, with some rare entities ---
    # Base rate proportional to theta (fast entities fire more).
    # Make ~25% of entities "rare" by scaling down 10-100x.
    fire_rates = theta.copy()
    n_rare = max(1, n_entities // 4)
    rare_idx = rng.choice(n_entities, size=n_rare, replace=False)
    rare_scale = np.exp(rng.uniform(np.log(0.01), np.log(0.1), size=n_rare))
    fire_rates[rare_idx] *= rare_scale

    # Normalize so total rate produces ~n_events in reasonable time.
    total_rate = fire_rates.sum()

    # --- Simulate continuous-time OU with Euler-Maruyama ---
    # We simulate at fine dt_sim, but events fire at Poisson times.
    X = rng.normal(0, 0.1, size=n_entities)  # initial states near μ=0

    timestamps: list[float] = []
    entity_ids: list[int] = []
    values: list[float] = []

    t = 0.0
    sqrt_dt = np.sqrt(dt_sim)
    n_collected = 0

    # Per-entity next-fire times (exponential inter-arrivals).
    next_fire = np.array([
        t + rng.exponential(1.0 / max(r, 1e-12))
        for r in fire_rates
    ])

    while n_collected < n_events:
        # Advance state by dt_sim (Euler-Maruyama).
        dW = rng.normal(0, sqrt_dt, size=n_entities)
        drift = theta * (0.0 - X)  # mean reversion to μ=0
        # Add coupling: entity j pulls entity i
        coupling_drift = C @ X - C.sum(axis=1) * X
        X += (drift + coupling_drift) * dt_sim + sigma * dW
        t += dt_sim

        # Check which entities fire in this interval.
        fired = np.where(next_fire <= t)[0]
        if len(fired) > 0:
            # Sort by fire time for proper ordering.
            order = np.argsort(next_fire[fired])
            for idx in fired[order]:
                if n_collected >= n_events:
                    break
                timestamps.append(next_fire[idx])
                entity_ids.append(idx)
                values.append(float(X[idx]))
                n_collected += 1
                # Schedule next fire for this entity.
                next_fire[idx] = (
                    next_fire[idx]
                    + rng.exponential(1.0 / max(fire_rates[idx], 1e-12))
                )

    timestamps_arr = np.array(timestamps, dtype=np.float64)
    entity_ids_arr = np.array(entity_ids, dtype=np.int32)
    values_arr = np.array(values, dtype=np.float32)

    return {
        "timestamps": timestamps_arr,
        "entity_ids": entity_ids_arr,
        "values": values_arr,
        "theta": theta,
        "sigma": sigma,
        "coupling": C,
        "fire_rates": fire_rates,
        "rare_idx": rare_idx,
    }


def build_tape_arrays(
    tape: dict,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict:
    """Convert raw tape into training-ready arrays.

    Returns:
        entity_ids:  (N,)          int32
        features:    (N, 3)        float32 — [value, log(dt_entity+ε), log(dt_global+ε)]
        targets:     (N, n_h)      float32 — future values at each horizon
        masks:       (N, n_h)      float32 — 1 where target is valid
    """
    # Vectorised in data/generators/tape_arrays.py (identical output to the old per-event loops).
    return build_tape_arrays_vec(tape, horizons)


def compute_entity_weights(entity_ids: np.ndarray, n_entities: int, alpha: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights: w_i ∝ 1 / count(i)^alpha."""
    counts = np.bincount(entity_ids, minlength=n_entities).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts ** alpha
    w /= w.mean()  # normalize so mean weight = 1
    return w.astype(np.float32)


if __name__ == "__main__":
    # Quick sanity check.
    tape = generate_ou_tape(n_entities=16, n_events=10_000, seed=42)
    arrays = build_tape_arrays(tape, horizons=(1, 10, 50))

    print(f"Events: {len(tape['timestamps'])}")
    print(f"Entities: {int(tape['entity_ids'].max()) + 1}")
    print(f"Theta range: [{tape['theta'].min():.3f}, {tape['theta'].max():.3f}]")
    print(f"Sigma range: [{tape['sigma'].min():.3f}, {tape['sigma'].max():.3f}]")
    print(f"Rare entities: {tape['rare_idx']}")

    # Event counts per entity.
    counts = np.bincount(tape["entity_ids"], minlength=16)
    for i in range(16):
        marker = " (rare)" if i in tape["rare_idx"] else ""
        print(f"  entity {i:2d}: {counts[i]:5d} events, "
              f"θ={tape['theta'][i]:.3f}, σ={tape['sigma'][i]:.3f}{marker}")

    print(f"\nFeatures shape: {arrays['features'].shape}")
    print(f"Targets shape:  {arrays['targets'].shape}")
    print(f"Mask coverage:  {arrays['masks'].mean(axis=0)}")

    w = compute_entity_weights(tape["entity_ids"], 16)
    print(f"Entity weights: {w}")
