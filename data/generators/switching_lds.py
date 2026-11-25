"""
Regime-Switching Linear Dynamical System — Tape Generator.

N entities follow linear dynamics, but the dynamics switch between
K regimes at random times. Each regime has different A, B matrices.

    X_{t+1} = A_k X_t + B_k u_t + noise    (regime k active)

Regime switches are shared across entities (global regime).
This tests:
    - Whether the Tape's hidden state captures regime identity
    - Adaptation to non-stationary dynamics
    - Temporal abstraction (fast entity dynamics + slow regime switches)
"""

from __future__ import annotations

import numpy as np

from data.generators.tape_arrays import build_tape_arrays as build_tape_arrays_vec


def generate_switching_tape(
    n_entities: int = 8,
    n_events: int = 80_000,
    n_regimes: int = 3,
    regime_duration_range: tuple[float, float] = (50.0, 200.0),
    coupling_strength: float = 0.3,
    noise_scale: float = 0.1,
    dt_sim: float = 0.01,
    seed: int = 42,
) -> dict:
    """Generate a regime-switching multi-entity tape.

    Each regime has a different coupling matrix and drift.
    Regime switches happen at random intervals.
    """
    rng = np.random.RandomState(seed)
    n = n_entities

    # Per-entity base dynamics.
    theta_base = np.exp(rng.uniform(np.log(0.1), np.log(5.0), size=n))
    sigma = np.exp(rng.uniform(np.log(0.05), np.log(1.0), size=n))

    # Per-regime parameters.
    regime_mus = [rng.uniform(-2, 2, size=n) for _ in range(n_regimes)]
    regime_couplings = []
    for k in range(n_regimes):
        C = rng.uniform(-coupling_strength, coupling_strength, size=(n, n))
        mask = rng.uniform(size=(n, n)) > 0.5
        C = C * mask.astype(np.float64)
        np.fill_diagonal(C, 0.0)
        regime_couplings.append(C)

    # Firing rates.
    fire_rates = theta_base.copy()
    n_rare = max(1, n // 4)
    rare_idx = rng.choice(n, size=n_rare, replace=False)
    fire_rates[rare_idx] *= rng.uniform(0.05, 0.15, size=n_rare)

    # --- Generate regime switch times ---
    regime_switches: list[float] = [0.0]
    regime_sequence: list[int] = [0]
    t_switch = 0.0
    while t_switch < n_events * 5.0 / fire_rates.sum():
        duration = rng.uniform(*regime_duration_range)
        t_switch += duration
        regime_switches.append(t_switch)
        # Switch to a different regime.
        curr = regime_sequence[-1]
        new = rng.choice([k for k in range(n_regimes) if k != curr])
        regime_sequence.append(new)

    regime_switches_arr = np.array(regime_switches)
    regime_sequence_arr = np.array(regime_sequence)

    # --- Simulate ---
    X = rng.normal(0, 0.5, size=n)
    sqrt_dt = np.sqrt(dt_sim)

    timestamps: list[float] = []
    entity_ids: list[int] = []
    values: list[float] = []
    regimes: list[int] = []

    t = 0.0
    n_collected = 0

    next_fire = np.array([
        t + rng.exponential(1.0 / max(fr, 1e-12))
        for fr in fire_rates
    ])

    def get_regime(t: float) -> int:
        idx = int(np.searchsorted(regime_switches_arr, t, side='right')) - 1
        idx = min(idx, len(regime_sequence_arr) - 1)
        return int(regime_sequence_arr[idx])

    while n_collected < n_events:
        k = get_regime(t)
        mu_k = regime_mus[k]
        C_k = regime_couplings[k]

        dW = rng.normal(0, sqrt_dt, size=n)
        drift = theta_base * (mu_k - X) + C_k @ X - C_k.sum(axis=1) * X
        X += drift * dt_sim + sigma * dW
        # Clamp to prevent explosion.
        X = np.clip(X, -20, 20)
        t += dt_sim

        fired = np.where(next_fire <= t)[0]
        if len(fired) > 0:
            order = np.argsort(next_fire[fired])
            for idx in fired[order]:
                if n_collected >= n_events:
                    break
                timestamps.append(next_fire[idx])
                entity_ids.append(idx)
                values.append(float(X[idx]))
                regimes.append(get_regime(next_fire[idx]))
                n_collected += 1
                next_fire[idx] = (
                    next_fire[idx]
                    + rng.exponential(1.0 / max(fire_rates[idx], 1e-12))
                )

    return {
        "timestamps": np.array(timestamps[:n_events], dtype=np.float64),
        "entity_ids": np.array(entity_ids[:n_events], dtype=np.int32),
        "values": np.array(values[:n_events], dtype=np.float32),
        "regimes": np.array(regimes[:n_events], dtype=np.int32),
        "n_regimes": n_regimes,
        "regime_switches": regime_switches_arr,
        "regime_sequence": regime_sequence_arr,
        "regime_mus": regime_mus,
        "regime_couplings": regime_couplings,
        "theta": theta_base,
        "sigma": sigma,
        "fire_rates": fire_rates,
        "rare_idx": rare_idx,
        "coupling": sum(regime_couplings) / n_regimes,  # avg for embedding corr
    }


def build_switching_tape_arrays(
    tape: dict,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict:
    """Convert switching tape into training arrays."""
    # Vectorised in data/generators/tape_arrays.py (identical output to the old per-event loops).
    # Feature channel is clipped to remove residual inf/nan; targets stay raw (unchanged behaviour).
    values = np.asarray(tape["values"])
    return build_tape_arrays_vec(tape, horizons, feature_values=np.clip(values, -20, 20), target_values=values)


if __name__ == "__main__":
    tape = generate_switching_tape(n_entities=8, n_events=5000, seed=42)
    arrays = build_switching_tape_arrays(tape)
    counts = np.bincount(tape["entity_ids"], minlength=8)
    regime_counts = np.bincount(tape["regimes"], minlength=tape["n_regimes"])
    print(f"Events: {len(tape['timestamps'])}")
    print(f"Regimes: {tape['n_regimes']}, switches: {len(tape['regime_switches'])-1}")
    print(f"Regime distribution: {regime_counts}")
    for i in range(8):
        r = " (rare)" if i in tape["rare_idx"] else ""
        print(f"  entity {i}: {counts[i]:5d} events{r}")
