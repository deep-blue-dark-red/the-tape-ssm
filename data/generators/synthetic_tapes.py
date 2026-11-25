"""Synthetic tape generation for the synthetic runner: dispatch, per-process memo, parallel prefetch.

Generation is pure numpy and single-threaded (Ogata thinning / Euler loops), and a tape is a
deterministic function of (dataset, parameters, seed).  The runner asks for the same tape several
times per seed (training stream, ``--eval-thin``, A12b density variants, the CI split), and runs its
seeds one after another, so before this module a Hawkes run spent most of its wall time regenerating
on one core while the GPU idled.  ``tape`` memoises per process; ``prefetch`` fills the memo for
every seed at once in a spawn pool.  A worker runs the same code on the same inputs, so the arrays
are the ones the parent would have drawn (``tests/test_synthetic_tapes.py`` pins bit-identity).

Numpy-only on purpose: spawned workers import this module, not torch or MLX.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

from data.generators.hawkes_process import cached_hawkes_tape
from data.generators.lotka_volterra import generate_lv_tape
from data.generators.ou_process import generate_ou_tape
from data.generators.switching_lds import generate_switching_tape

Key = tuple[str, int, int, float, int, bool, bool, bool, float | None, str]

_MEMO: dict[Key, dict] = {}


def key_for(args, seed: int, n_events: int) -> Key:
    return (args.dataset, int(args.n_entities), int(n_events), float(args.coupling_strength), int(seed),
            bool(getattr(args, "uncoupled", False)), bool(getattr(args, "common_drive", False)),
            bool(getattr(args, "hawkes_nonneg", False)), getattr(args, "branching_ratio", None),
            str(getattr(args, "lv_sampling", "poisson") or "poisson"))


def make_tape(key: Key) -> dict:
    (dataset, n_entities, n_events, coupling, seed, uncoupled, common_drive, nonneg, ratio,
     lv_sampling) = key
    if dataset == "ou":
        return generate_ou_tape(n_entities=n_entities, n_events=n_events, coupling_strength=coupling, seed=seed)
    if dataset == "hawkes":
        extra = {"nonneg": True} if nonneg else {}          # only when set: old disk-cache keys unchanged
        if ratio is not None:
            extra["branching_ratio"] = float(ratio)
        return cached_hawkes_tape(n_entities=n_entities, n_events=n_events, excitation_strength=coupling,
                                  seed=seed, uncoupled=uncoupled, common_drive=common_drive, **extra)
    if dataset == "lv":
        return generate_lv_tape(n_species=n_entities, n_events=n_events, interaction_strength=coupling,
                                seed=seed, sampling=lv_sampling)
    if dataset == "switching":
        return generate_switching_tape(n_entities=n_entities, n_events=n_events, coupling_strength=coupling,
                                       seed=seed)
    raise ValueError(f"Unknown dataset: {dataset}")


def tape(args, seed: int, n_events: int) -> dict:
    """The tape for (args' generator parameters, seed, n_events); generated once per process.
    Callers get their own array copies, so an in-place edit cannot leak into a later request."""
    k = key_for(args, seed, n_events)
    if k not in _MEMO:
        _MEMO[k] = make_tape(k)
    return copy.deepcopy(_MEMO[k])


def forget(seed: int) -> None:
    """Drop every memoised tape for ``seed`` (the runner calls this when a seed is done)."""
    for k in [k for k in _MEMO if k[4] == seed]:
        del _MEMO[k]


def prefetch(args, seeds: list[int], n_events: int, workers: int) -> int:
    """Generate the tapes for ``seeds`` in a ``workers``-process spawn pool into the memo.
    Spawn, not fork: the parent may already hold CUDA/Metal state or BLAS threads.
    Returns how many tapes were generated (already-memoised ones are skipped)."""
    todo = [k for k in (key_for(args, s, n_events) for s in seeds) if k not in _MEMO]
    workers = min(workers, len(todo))
    if workers < 2:
        return 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for k, t in zip(todo, ex.map(make_tape, todo)):
            _MEMO[k] = t
    return len(todo)


def default_workers(n_seeds: int) -> int:
    return max(1, min(n_seeds, (os.cpu_count() or 1) - 1))
