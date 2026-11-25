"""Interference stress harness (wide-tape branch, plan §8 S1/S2).

The decisive stress test for the interference hypothesis: hold a target process fixed
while inserting increasing volumes of irrelevant events, then measure forecast
degradation, rare-entity recall, and latency (S1).  Entity count varies independently of
total event count (S2).

The target process depends only on ``seed`` and the target parameters — distractor events
are injected around it, never mixed into its values — so the same seed reproduces the same
target tape at any distractor volume.  Output is the raw tape layout
(``entity_ids``, ``timestamps``, ``values``, time-ordered) so ``build_tape_arrays`` works
unchanged.
"""

from __future__ import annotations

import numpy as np


def make_interference_tape(target_entities: int = 4, distractor_entities: int = 8,
                           target_rate: float = 2.0, distractor_rate: float = 0.0,
                           duration: float = 100.0, seed: int = 0,
                           value_scale: float = 1.0) -> dict:
    """Fixed target process + Poisson distractor injection → raw tape dict.

    * target entities: independent OU-like AR(1) processes sampled at exponential
      (rate ``target_rate``) times — the process to be forecast.
    * distractor entities: white-noise events at ``distractor_rate`` per entity per unit
      time — irrelevant by construction (iid values, no coupling to targets).
    """
    if target_entities < 1 or distractor_entities < 0 or distractor_rate < 0:
        raise ValueError("need target_entities >= 1, distractor_* >= 0")
    rng_t = np.random.default_rng(seed)             # target process: fixed by seed
    rng_d = np.random.default_rng(10_000 + seed)    # distractors: separate stream

    # --- Target process (identical for any distractor_rate) ---
    t_eids, t_times, t_vals = [], [], []
    for a in range(target_entities):
        t = rng_t.exponential(1.0 / target_rate)
        x = float(rng_t.standard_normal()) * value_scale
        while t < duration:
            t_eids.append(a)
            t_times.append(t)
            t_vals.append(x)
            x = 0.9 * x + 0.4 * float(rng_t.standard_normal()) * value_scale   # AR(1)
            t += rng_t.exponential(1.0 / target_rate)

    # --- Distractor events (iid noise, no coupling) ---
    d_eids, d_times, d_vals = [], [], []
    n_d = rng_t.poisson(distractor_rate * duration * distractor_entities) if distractor_rate else 0
    if n_d:
        dt = rng_d.uniform(0.0, duration, size=n_d)
        de = rng_d.integers(target_entities, target_entities + distractor_entities, size=n_d)
        dv = rng_d.standard_normal(n_d).astype(np.float32) * value_scale
        d_eids, d_times, d_vals = list(de), list(dt), list(dv)

    eids = np.array(t_eids + d_eids, dtype=np.int32)
    times = np.array(t_times + d_times, dtype=np.float64)
    vals = np.array(t_vals + d_vals, dtype=np.float32)
    order = np.argsort(times, kind="stable")
    return {"entity_ids": eids[order], "timestamps": times[order], "values": vals[order]}


def split_target(tape: dict, target_entities: int) -> dict:
    """The target-only sub-tape (distractor rows removed) — for fixed-process assertions."""
    m = np.asarray(tape["entity_ids"]) < target_entities
    return {k: np.asarray(v)[m] for k, v in tape.items()}
