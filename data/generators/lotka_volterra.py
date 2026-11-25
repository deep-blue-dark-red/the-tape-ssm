"""
Multi-Species Lotka-Volterra — Predator-Prey Tape Generator.

Generates N species with predator-prey dynamics:
    dX_i/dt = X_i (r_i + Σ_j A_ij X_j) + noise

Asynchronous event firing at rates proportional to population size.
This tests:
    - Cyclic cross-entity dependencies (predator-prey oscillations)
    - Multi-scale dynamics (fast oscillations + slow trends)
    - Regime changes (population crashes/booms)
    - Non-linear cross-entity interactions
"""

from __future__ import annotations

import numpy as np

from data.generators.tape_arrays import build_tape_arrays as build_tape_arrays_vec


def generate_lv_tape(
    n_species: int = 8,
    n_events: int = 80_000,
    interaction_strength: float = 0.3,
    noise_scale: float = 0.05,
    dt_sim: float = 0.005,
    seed: int = 42,
    sampling: str = "poisson",
    state_kappa: float = 1.0,
) -> dict:
    """Generate a multi-species Lotka-Volterra tape.

    Uses a competition matrix where half the species pairs have
    predator-prey dynamics (asymmetric ±) and others have competition (-/-).

    ``sampling`` (SPEC_PAPER_NOV E8): ``"poisson"`` (default) fires entity i at its
    constant rate ``fire_rates[i]``; ``"state"`` fires it with the state-dependent hazard
    ``fire_rates[i] * exp(kappa * z_i) / c_i``, where z_i is the standardised
    log-population (Welford over the Euler steps) and c_i an EMA of exp(kappa * z_i),
    so the time-average hazard still matches the Poisson arm's rate (matched mean rates).
    Realised per-entity rates are returned in ``fire_rates_realized``.
    """
    rng = np.random.RandomState(seed)
    n = n_species

    # Intrinsic growth rates: some positive (prey), some negative (predators).
    r = rng.uniform(-0.5, 1.0, size=n)
    # Ensure at least half are prey (positive growth).
    n_prey = max(2, n // 2)
    prey_idx = rng.choice(n, size=n_prey, replace=False)
    r[prey_idx] = np.abs(r[prey_idx]) + 0.1

    # Interaction matrix A: predator-prey pairs.
    A = np.zeros((n, n))
    # Self-regulation (carrying capacity).
    np.fill_diagonal(A, -rng.uniform(0.01, 0.05, size=n))

    # Create predator-prey pairs.
    for i in range(n):
        n_interactions = rng.randint(1, min(4, n))
        targets = rng.choice(n, size=n_interactions, replace=False)
        for j in targets:
            if i == j:
                continue
            if i in prey_idx and j not in prey_idx:
                # Prey i is eaten by predator j.
                A[i, j] = -rng.uniform(0.05, interaction_strength)
                A[j, i] = rng.uniform(0.05, interaction_strength * 0.5)
            elif i not in prey_idx and j in prey_idx:
                # Predator i eats prey j.
                A[i, j] = rng.uniform(0.05, interaction_strength * 0.5)
                A[j, i] = -rng.uniform(0.05, interaction_strength)
            else:
                # Competition.
                A[i, j] = -rng.uniform(0.01, interaction_strength * 0.3)
                A[j, i] = -rng.uniform(0.01, interaction_strength * 0.3)

    # Firing rates: proportional to base growth rate.
    fire_rates = np.abs(r) + 0.5
    n_rare = max(1, n // 4)
    rare_idx = rng.choice(n, size=n_rare, replace=False)
    fire_rates[rare_idx] *= rng.uniform(0.05, 0.15, size=n_rare)

    # --- Simulate with Euler-Maruyama ---
    # Use log-populations to prevent negativity.
    log_X = rng.uniform(-1, 1, size=n)  # initial log-populations
    sqrt_dt = np.sqrt(dt_sim)

    timestamps = []
    entity_ids = []
    values = []

    t = 0.0
    n_collected = 0

    use_state = sampling == "state"
    if sampling not in ("poisson", "state"):
        raise ValueError(f"Unknown lv sampling: {sampling}")

    # Welford state for the state-dependent hazard (z-score of log X_i).
    run_mean = log_X.copy()
    run_m2 = np.zeros(n)
    n_steps = 0
    # Self-normalisation: EMA of exp(kappa * z_i), so the time-average hazard
    # converges to fire_rates (matched mean rates with the Poisson arm).
    ema_norm = np.ones(n)
    ema_decay = 0.999
    counts = np.zeros(n, dtype=np.int64)

    next_fire = np.array([
        t + rng.exponential(1.0 / max(fr, 1e-12))
        for fr in fire_rates
    ])

    while n_collected < n_events:
        X = np.exp(log_X)
        # dlog(X_i)/dt = r_i + Σ_j A_ij X_j + noise
        drift = r + A @ X
        dW = rng.normal(0, sqrt_dt, size=n)
        log_X += drift * dt_sim + noise_scale * dW
        # Clamp to prevent explosion.
        log_X = np.clip(log_X, -10, 10)
        t += dt_sim

        if use_state:
            n_steps += 1
            delta = log_X - run_mean
            run_mean += delta / n_steps
            run_m2 += delta * (log_X - run_mean)
            std = np.sqrt(np.maximum(run_m2 / max(n_steps - 1, 1), 1e-6))
            z = (log_X - run_mean) / std
            hz = np.exp(np.clip(state_kappa * z, -5.0, 5.0))
            ema_norm = ema_decay * ema_norm + (1.0 - ema_decay) * hz
            hazard = np.clip(fire_rates * hz / np.maximum(ema_norm, 1e-6), 0.0, 50.0)
            u = rng.uniform(size=n)
            fired = np.where(u < hazard * dt_sim)[0]
            for idx in fired:
                if n_collected >= n_events:
                    break
                timestamps.append(t)
                entity_ids.append(idx)
                values.append(float(log_X[idx]))
                counts[idx] += 1
                n_collected += 1
            continue

        fired = np.where(next_fire <= t)[0]
        if len(fired) > 0:
            order = np.argsort(next_fire[fired])
            for idx in fired[order]:
                if n_collected >= n_events:
                    break
                timestamps.append(next_fire[idx])
                entity_ids.append(idx)
                values.append(float(log_X[idx]))  # log-population
                n_collected += 1
                next_fire[idx] = (
                    next_fire[idx]
                    + rng.exponential(1.0 / max(fire_rates[idx], 1e-12))
                )

    out = {
        "timestamps": np.array(timestamps[:n_events], dtype=np.float64),
        "entity_ids": np.array(entity_ids[:n_events], dtype=np.int32),
        "values": np.array(values[:n_events], dtype=np.float32),
        "growth_rates": r,
        "interaction_matrix": A,
        "fire_rates": fire_rates,
        "rare_idx": rare_idx,
        "prey_idx": prey_idx,
        "theta": np.abs(r),  # compat
        "sigma": np.full(n, noise_scale),  # compat
        "coupling": A,  # for embedding corr
    }

    if use_state:
        t_total = max(timestamps[-1] - timestamps[0], 1e-9) if timestamps else 1e-9
        out["sampling"] = "state"
        out["state_kappa"] = float(state_kappa)
        out["fire_rates_realized"] = counts / t_total
    return out


def build_lv_tape_arrays(
    tape: dict,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict:
    """Convert LV tape into training arrays."""
    # Vectorised in data/generators/tape_arrays.py (identical output to the old per-event loops).
    return build_tape_arrays_vec(tape, horizons)


if __name__ == "__main__":
    tape = generate_lv_tape(n_species=8, n_events=5000, seed=42)
    arrays = build_lv_tape_arrays(tape)
    counts = np.bincount(tape["entity_ids"], minlength=8)
    print(f"Events: {len(tape['timestamps'])}")
    for i in range(8):
        role = "prey" if i in tape["prey_idx"] else "pred"
        r = " (rare)" if i in tape["rare_idx"] else ""
        print(f"  species {i} [{role}]: {counts[i]:5d} events, r={tape['growth_rates'][i]:.3f}{r}")
    print(f"Interaction matrix:\n{tape['interaction_matrix'].round(3)}")
