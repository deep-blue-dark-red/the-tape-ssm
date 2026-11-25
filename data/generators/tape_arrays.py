"""
Vectorised conversion of a raw event tape into training arrays.

All four synthetic generators produce the same raw tape layout
(``entity_ids``, ``timestamps``, ``values`` — one event per row, in time
order) and used to convert it with per-event Python loops.  This module does
the same thing (bit-exact, see tests/test_tape_arrays.py) with a single stable
argsort by entity — 0.1 s instead of 0.5 s at 500k events.  Note that the
simulation inside ``generate_*_tape`` is the actual cost at that size (~100 s
for OU), not this conversion:

* ``dt_entity``  — time since the same entity's previous event; first events
  are imputed with that entity's median gap (1.0 if it never fires twice).
* ``dt_global``  — time since the previous event of any entity.
* ``targets``    — the entity's value at its h-th next firing, for each
  horizon h, with ``masks`` = 1 where such a firing exists.

The generators differ only in how values are transformed, so the transforms
are parameters: ``feature_values`` feeds the value channel of ``features``,
``target_values`` (defaults to the same array) feeds ``targets``.
"""

from __future__ import annotations

import numpy as np

DT_FLOOR = 1e-4
LOG_DT_CLIP = 10.0


def entity_dt(entity_ids: np.ndarray, timestamps: np.ndarray, n_entities: int | None = None) -> np.ndarray:
    """Per-event gap since the same entity's previous event (first events → entity median)."""
    n = len(entity_ids)
    if n_entities is None:
        n_entities = int(entity_ids.max()) + 1 if n else 0
    order = np.argsort(entity_ids, kind="stable")           # events grouped by entity, time order kept
    eid_s = entity_ids[order]
    t_s = timestamps[order].astype(np.float64)
    dt_s = np.empty(n, dtype=np.float32)                     # float32 like the original loop (bit-exact medians)
    dt_s[1:] = t_s[1:] - t_s[:-1]
    first = np.ones(n, dtype=bool)
    first[1:] = eid_s[1:] != eid_s[:-1]
    dt_s[first] = -1.0

    # Median of valid gaps per entity, in sorted layout: split at run boundaries.
    fill = np.ones(n_entities, dtype=np.float32)
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], n)
    for s, e in zip(starts, ends):
        valid = dt_s[s:e]
        valid = valid[valid > 0]
        if valid.size:
            fill[eid_s[s]] = np.median(valid)
    dt_s[first] = fill[eid_s[first]]

    dt = np.empty(n, dtype=np.float32)
    dt[order] = dt_s
    return dt


def future_targets(entity_ids: np.ndarray, target_values: np.ndarray, horizons: tuple[int, ...],
                   ) -> tuple[np.ndarray, np.ndarray]:
    """targets[i, k] = target_values at entity_ids[i]'s horizons[k]-th next firing; masks = validity."""
    n = len(entity_ids)
    n_h = len(horizons)
    targets = np.zeros((n, n_h), dtype=np.float32)
    masks = np.zeros((n, n_h), dtype=np.float32)
    if n == 0:
        return targets, masks
    order = np.argsort(entity_ids, kind="stable")
    eid_s = entity_ids[order]
    first = np.ones(n, dtype=bool)
    first[1:] = eid_s[1:] != eid_s[:-1]
    run_id = np.cumsum(first) - 1                            # which run each sorted position is in
    run_start = np.flatnonzero(first)
    run_len = np.diff(np.append(run_start, n))
    local = np.arange(n) - run_start[run_id]                 # index within the entity's own sequence
    remaining = run_len[run_id] - local                      # events left in the run, including self
    vals_s = target_values[order]
    for k, h in enumerate(horizons):
        ok = remaining > h
        src = np.flatnonzero(ok)
        targets[order[src], k] = vals_s[src + h]
        masks[order[src], k] = 1.0
    return targets, masks


def future_target_gaps(entity_ids: np.ndarray, timestamps: np.ndarray,
                       horizons: tuple[int, ...]) -> np.ndarray:
    """gaps[i, k] = time from event i to the same entity's horizons[k]-th next event (0 where none).
    Same indexing as :func:`future_targets`; feeds the --query-time head."""
    n = len(entity_ids)
    gaps = np.zeros((n, len(horizons)), dtype=np.float64)
    if n == 0:
        return gaps
    order = np.argsort(entity_ids, kind="stable")
    eid_s = entity_ids[order]
    first = np.ones(n, dtype=bool)
    first[1:] = eid_s[1:] != eid_s[:-1]
    run_id = np.cumsum(first) - 1
    run_start = np.flatnonzero(first)
    run_len = np.diff(np.append(run_start, n))
    local = np.arange(n) - run_start[run_id]
    remaining = run_len[run_id] - local
    t_s = np.asarray(timestamps, dtype=np.float64)[order]
    for k, h in enumerate(horizons):
        src = np.flatnonzero(remaining > h)
        gaps[order[src], k] = t_s[src + h] - t_s[src]
    return gaps


def time_targets(entity_ids: np.ndarray, timestamps: np.ndarray, target_values: np.ndarray,
                 deltas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Time-based horizons: targets[i, k] = the same entity's value at its first
    event with ``t >= t_i + deltas[k]``; masks = validity (0 when coverage ends first).

    Entity runs are made disjoint for one vectorised searchsorted by offsetting
    each run's times with ``run_id * M`` where ``M`` exceeds the tape's time span
    plus the largest delta, so a query can never leak into another entity's run.
    """
    n, n_d = len(entity_ids), len(deltas)
    targets = np.zeros((n, n_d), dtype=np.float32)
    masks = np.zeros((n, n_d), dtype=np.float32)
    if n == 0:
        return targets, masks
    d_arr = np.asarray(deltas, dtype=np.float64)
    order = np.lexsort((np.asarray(timestamps, dtype=np.float64), entity_ids))  # entity-major, time within
    eid_s = entity_ids[order]
    t_s = np.asarray(timestamps, dtype=np.float64)[order]
    vals_s = np.asarray(target_values, dtype=np.float64)[order]
    first = np.ones(n, dtype=bool)
    first[1:] = eid_s[1:] != eid_s[:-1]
    run_id = np.cumsum(first) - 1
    run_end = np.append(np.flatnonzero(first)[1:], n)        # end position (exclusive) per run
    m_off = (t_s.max() - t_s.min() if n > 1 else 0.0) + d_arr.max() + 1.0
    key = t_s + run_id * m_off
    for k, d in enumerate(d_arr):
        j = np.searchsorted(key, key + d, side="left")       # first same-run event >= t_i + d
        ok = j < run_end[run_id]
        src = np.flatnonzero(ok)
        targets[order[src], k] = vals_s[j[src]]
        masks[order[src], k] = 1.0
    return targets, masks


def dense_targets(entity_ids: np.ndarray, target_values: np.ndarray, horizons: tuple[int, ...],
                  n_entities: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """SPEC_V2 M3: all-entity targets/masks ``(n, n_entities, n_h)`` for the loss-mode arms.

    The firing entity's column is always the sparse target (:func:`future_targets`).  For every
    other entity ``e`` at position ``i``:

    * ``mode="impute"`` — the draft's A2 ("no-op / current state"): ``e``'s last observed value
      *strictly before* ``i``, the same value at every horizon, mask 1 where such an observation
      exists.
    * ``mode="next"`` — ``e``'s first observation *strictly after* ``i``, same value at every
      horizon, mask 1 where one exists.  No imputation; strictly more supervision.

    Only the loss consumes these arrays — every arm is still evaluated on the identical sparse
    task metric, and the firing column makes "dense restricted to the firing entity == sparse"
    hold by construction."""
    if mode not in ("impute", "next"):
        raise ValueError(f"unknown dense mode: {mode!r}")
    n = len(entity_ids)
    n_h = len(horizons)
    targets = np.zeros((n, n_entities, n_h), dtype=np.float32)
    masks = np.zeros((n, n_entities, n_h), dtype=np.float32)
    if n == 0:
        return targets, masks
    sparse_t, sparse_m = future_targets(entity_ids, target_values, horizons)
    pos = np.arange(n)
    for e in range(n_entities):
        idx = np.flatnonzero(entity_ids == e)                # e's event positions, ascending
        if idx.size == 0:
            continue
        if mode == "impute":
            j = np.searchsorted(idx, pos, side="left") - 1   # last event of e strictly before i
            valid = j >= 0
            src = idx[np.maximum(j, 0)]
        else:
            j = np.searchsorted(idx, pos, side="right")      # first event of e strictly after i
            valid = j < idx.size
            src = idx[np.minimum(j, idx.size - 1)]
        col = target_values[src][:, None]
        targets[:, e, :] = np.where(valid[:, None], col, 0.0)  # masked cells stay 0, like sparse
        masks[:, e, :] = valid[:, None].astype(np.float32)
    # The firing entity's column carries the real k-th-next targets (horizon-wise).
    fid = entity_ids
    targets[pos, fid, :] = sparse_t
    masks[pos, fid, :] = sparse_m
    return targets, masks


def build_tape_arrays(tape: dict, horizons: tuple[int, ...] = (1, 10, 50), *,
                      feature_values: np.ndarray | None = None,
                      target_values: np.ndarray | None = None,
                      time_deltas: tuple[float, ...] | None = None,
                      query_time: bool = False) -> dict:
    """Raw tape → ``{entity_ids, features (N,F), targets (N,H), masks (N,H)}``.

    ``query_time`` appends H columns of ``log`` elapsed time to each horizon's target observation
    (floored and clipped like the clocks; 0 where the target is masked) AFTER the clocks — the
    projection never reads them (``TapeConfig.n_features``), the gaussian head does
    (``TapeConfig.query_cols``).  k-th-next-firing targets only.

    features = [value(s) | marks | aux | log(dt_entity), log(dt_global)], both
    log-dts floored at ``DT_FLOOR`` and clipped to ±``LOG_DT_CLIP``.

    ``values`` may be ``(N,)`` (classic single-channel tape, features (N,3))
    or ``(N, C)`` (multi-channel tape).  A 2-D tape must carry a 1-D
    ``tape["target"]`` (the scored channel) unless ``target_values`` is given.
    Optional ``tape["marks"]`` ``(N, C)`` 0/1 missing-value flags and
    ``tape["aux"]`` ``(N, A)`` per-event auxiliary features (sensor/censoring)
    are appended between the values and the two clocks.  With ``time_deltas``
    the targets become time-based (value at the first event ``t >= t_i +``
    ``deltas[k]``) instead of k-th-next-firing; ``horizons`` is then ignored.
    """
    entity_ids = np.asarray(tape["entity_ids"])
    timestamps = np.asarray(tape["timestamps"])
    values = np.asarray(tape["values"])
    n = len(timestamps)
    fv = values if feature_values is None else feature_values
    if fv.ndim == 1:
        tv = fv if target_values is None else target_values
    else:
        tv = tape["target"] if target_values is None else target_values

    dt_global = np.zeros(n, dtype=np.float32)
    if n > 1:
        dt_global[1:] = np.diff(timestamps).astype(np.float32)
    dt_ent = entity_dt(entity_ids, timestamps)

    log_dt_entity = np.clip(np.log(np.maximum(dt_ent, DT_FLOOR)), -LOG_DT_CLIP, LOG_DT_CLIP).astype(np.float32)
    log_dt_global = np.clip(np.log(np.maximum(dt_global, DT_FLOOR)), -LOG_DT_CLIP, LOG_DT_CLIP).astype(np.float32)
    cols = [fv.astype(np.float32) if fv.ndim == 2 else fv.astype(np.float32)[:, None]]
    for extra_key in ("marks", "aux"):
        if extra_key in tape:
            cols.append(np.asarray(tape[extra_key], dtype=np.float32))
    cols.append(np.stack([log_dt_entity, log_dt_global], axis=1))
    features = np.concatenate(cols, axis=1)

    if time_deltas is None:
        targets, masks = future_targets(entity_ids, np.asarray(tv, dtype=np.float32), horizons)
    else:
        targets, masks = time_targets(entity_ids, timestamps, tv, time_deltas)
    if "carried_targets" in tape:
        # E21 surrogate: targets/masks travel with their token from the real tape.
        targets = np.asarray(tape["carried_targets"], dtype=np.float32)
        masks = np.asarray(tape["carried_masks"], dtype=np.float32)
        if targets.shape != (n, len(horizons)) or masks.shape != (n, len(horizons)):
            raise ValueError("carried targets/masks must be (n, H)")
    if query_time:
        if time_deltas is not None:
            raise ValueError("query_time is defined for k-th-next-firing targets only")
        gaps = future_target_gaps(entity_ids, timestamps, horizons)
        q = np.clip(np.log(np.maximum(gaps, DT_FLOOR)), -LOG_DT_CLIP, LOG_DT_CLIP).astype(np.float32)
        q = np.where(masks > 0.5, q, 0.0).astype(np.float32)
        features = np.concatenate([features, q], axis=1)
    return {"entity_ids": entity_ids, "features": features, "targets": targets, "masks": masks}



def permute_ties(tape: dict, rng: np.random.Generator) -> dict:
    """E17 (SPEC_PAPER_NOV §11.2): shuffle the entity order inside every group of equal timestamps.

    One random rank per (group, entity) — the first event's draw — then a stable sort by (group,
    rank, original index), so each entity's own events keep their order and every k-th-next target,
    gap and value is unchanged; only the interleaving inside a tie moves (index order inside a tie
    otherwise reveals the channel).  Exactly one ``rng.random(n)`` draw per call; per-event arrays
    are reordered, everything else kept."""
    t = np.asarray(tape["timestamps"])
    e = np.asarray(tape["entity_ids"]).astype(np.int64)
    n = len(t)
    u = rng.random(n)
    if n < 2:
        return tape
    grp = np.concatenate([[0], np.cumsum(t[1:] != t[:-1])])
    pair = grp * (int(e.max()) + 1) + e
    _, first, inv = np.unique(pair, return_index=True, return_inverse=True)
    order = np.lexsort((np.arange(n), u[first][inv], grp))
    return {k: (v[order] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
            for k, v in tape.items()}


def thin_tape(tape: dict, frac: float, rng: np.random.Generator) -> dict | None:
    """SPEC_V2 §8 A12 density shift: drop each observation independently with probability ``frac``.

    Per-event arrays (``shape[0] == len(timestamps)``) are sliced; everything else is kept. Gaps
    and targets are rebuilt from the thinned timestamps downstream, so the k-th-next task is
    re-posed on the sparser stream. Exactly one ``rng.random(n)`` draw per call, so a fixed rng
    gives the same thinning across arms and seeds. Returns None when fewer than two observations
    survive; ``frac <= 0`` returns the tape itself.
    """
    if frac <= 0.0:
        return tape
    n = len(tape["timestamps"])
    keep = rng.random(n) >= frac
    if int(keep.sum()) < 2:
        return None
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
            for k, v in tape.items()}


def split_channel_independent(tape: dict) -> list[dict]:
    """M5 (SPEC_V2): split an interleaved tape into per-entity sub-tapes.

    Per-event arrays (matched by ``shape[0] == len(tape["entity_ids"])``) are
    sliced per entity preserving order; everything else is copied unchanged.
    Each sub-tape carries:
    - ``pack_group``: a unique id for the source tape, so packers that bin
      several short records together never put two sub-tapes of the SAME
      source tape in one bin — backbone state must not flow from entity A's
      events into entity B's of the same timeline (the CI control's premise);
    - ``_src_idx``: the sub-tape's events' positions in the source tape, so
      callers can apply temporal splits defined on the interleaved order.

    Within a sub-tape the global Δt equals the per-entity Δt by construction,
    and per-entity k-th-next targets are identical to the interleaved tape's —
    build_tape_arrays needs no changes.
    """
    eids = np.asarray(tape["entity_ids"])
    n = len(eids)
    group = object()                      # fresh, never equal to another call's
    subs = []
    for e in np.unique(eids):
        sel = eids == e
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n
                   else v) for k, v in tape.items()}
        sub["_src_idx"] = np.asarray(sel.nonzero()[0])
        sub["pack_group"] = group
        subs.append(sub)
    return subs


def shift_surrogate(tape: dict, bounds: tuple[int, ...], rng: np.random.Generator,
                    horizons: tuple[int, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
    """E21 (SPEC_PAPER_NOV §11.8): a surrogate tape that keeps every entity's own sub-stream and
    breaks the alignment between entities (the shift-predictor control).

    ``bounds`` are the split borders as event indices; inside each segment ``[a, b)`` every
    entity's timestamps are circularly shifted by its own offset, uniform in [1/4, 3/4] of the
    segment's time span, and the segment is re-sorted by time.  No event leaves its segment, so a
    temporal split of the surrogate holds the same events as the same split of the real tape.
    One ``rng.uniform`` draw of ``n_entities`` offsets per segment, in segment order.

    Returns ``(surrogate tape, order, valid)``: per-event arrays are reordered by ``order``
    (surrogate row r is real row ``order[r]``) and ``timestamps`` replaced; ``valid`` (n, H)
    float32, in REAL row order, is 1 where a row's h-th next own event sits in the same segment
    and still follows it after the shift.  Up to 2h rows per entity per segment lose that property:
    the h before the wrap point (their real target now lies in their past) and the h at the real
    end of the segment (their real target is in the next segment, no longer the entity's h-th next
    surrogate event).  Callers carry the real targets with ``order`` and multiply the masks by
    ``valid``.
    """
    t = np.asarray(tape["timestamps"], dtype=np.float64)
    e = np.asarray(tape["entity_ids"]).astype(np.int64)
    n = len(t)
    n_ent = int(e.max()) + 1 if n else 0
    new_t = t.copy()
    seg = np.zeros(n, dtype=np.int64)
    edges = [0, *[int(b) for b in bounds], n]
    for s, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        seg[a:b] = s
        offs = rng.uniform(0.25, 0.75, size=n_ent)
        if b - a < 2:
            continue
        lo = t[a]
        hi = t[b] if b < n else t[b - 1] + (t[b - 1] - t[a]) / (b - a - 1)
        span = hi - lo
        if span <= 0:
            continue
        new_t[a:b] = lo + np.mod(t[a:b] - lo + offs[e[a:b]] * span, span)
    order = np.lexsort((np.arange(n), new_t, seg))
    # h-th next own event of every real row (−1 if none), via one stable sort by entity
    by_ent = np.argsort(e, kind="stable")
    valid = np.zeros((n, len(horizons)), dtype=np.float32)
    for k, h in enumerate(horizons):
        nxt = np.full(n, -1, dtype=np.int64)
        if n > h:
            same = e[by_ent[:-h]] == e[by_ent[h:]]
            nxt[by_ent[:-h][same]] = by_ent[h:][same]
        has = nxt >= 0
        j = nxt[has]
        i = np.flatnonzero(has)
        valid[i, k] = ((seg[j] == seg[i]) & (new_t[j] > new_t[i])).astype(np.float32)
    sur = {key: (v[order] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
           for key, v in tape.items()}
    sur["timestamps"] = new_t[order].astype(np.asarray(tape["timestamps"]).dtype)
    return sur, order, valid


def shift_surrogate_tape(tape: dict, rng: np.random.Generator,
                         horizons: tuple[int, ...]) -> tuple[dict, dict]:
    """E21 stage 2, `shift` on an irregular record: apply ``shift_surrogate`` inside the
    record's own span and carry the real targets with their tokens.

    Returns ``(surrogate tape, info)``.  ``info`` carries the scored counts and the
    draw seed offset.  A record shorter than two events is returned unchanged.
    """
    n = len(tape["timestamps"])
    if n < 2:
        return dict(tape), {"mode": "shift", "test_n_scored": [0] * len(horizons),
                            "test_n_scored_real": [0] * len(horizons)}
    real = build_tape_arrays(tape, horizons)
    sur, order, valid = shift_surrogate(tape, (), rng, horizons)
    carry_targets = real["targets"][order]
    carry_masks = (np.asarray(real["masks"], dtype=np.float32)[order]
                   * np.asarray(valid, dtype=np.float32)[order]).astype(np.float32)
    sur["carried_targets"] = carry_targets
    sur["carried_masks"] = carry_masks
    info = {"mode": "shift",
            "test_n_scored": [int(c) for c in (carry_masks > 0.5).sum(axis=0)],
            "test_n_scored_real": [int(c) for c in (real["masks"] > 0.5).sum(axis=0)]}
    return sur, info


def swap_surrogate_records(records: list[dict], rng: np.random.Generator,
                           horizons: tuple[int, ...]) -> tuple[list[dict], dict]:
    """E21 stage 2, `swap` on record sets (PhysioNet, Ldfpa).

    One derangement of records per channel: each channel of each record is replaced by
    the same channel of a donor record.  The donor's events (timestamps, values, targets)
    travel together, and the recipient's events are re-merged in time order.  Every real
    event is scored exactly once, in a different record's context.  Targets therefore come
    from the donor's real tape and the scored set is identical to the real one.
    """
    n_rec = len(records)
    if n_rec < 2:
        return [dict(t) for t in records], {"mode": "swap", "test_n_scored": [0] * len(horizons),
                                            "test_n_scored_real": [0] * len(horizons)}
    # Real arrays per record, in the record's own row order.
    real_arrays = [build_tape_arrays(t, horizons) for t in records]
    # Channels per record and global entity range.
    max_ent = max((int(np.asarray(t["entity_ids"]).max()) + 1 if len(t["entity_ids"]) else 0)
                  for t in records)
    donors: dict[int, np.ndarray] = {}
    for e in range(max_ent):
        if n_rec > 1:
            perm = np.arange(n_rec)
            while True:
                rng.shuffle(perm)
                if np.all(perm != np.arange(n_rec)):
                    break
            donors[e] = perm
    out: list[dict] = []
    for r, tape in enumerate(records):
        eids = np.asarray(tape["entity_ids"])
        parts: list[dict] = []
        for e in np.unique(eids):
            d = int(donors[int(e)][r])
            deids = np.asarray(records[d]["entity_ids"])
            sel = deids == e
            if not sel.any():
                continue
            sub = {k: (np.asarray(v)[sel] if isinstance(v, np.ndarray) and v.shape[0] == len(deids)
                       else v) for k, v in records[d].items()}
            sub["carried_targets"] = real_arrays[d]["targets"][sel]
            sub["carried_masks"] = real_arrays[d]["masks"][sel]
            parts.append(sub)
        if not parts:
            out.append(dict(tape))
            continue
        times_cat = np.concatenate([np.asarray(p["timestamps"]) for p in parts])
        idx_cat = np.concatenate([np.arange(len(p["timestamps"])) for p in parts])
        order = np.lexsort((idx_cat, times_cat))   # primary: time, stable within a part
        # Concatenate in donor order, then reindex by the merged time order.
        cat = {}
        for key in parts[0]:
            vals = [np.asarray(p[key]) for p in parts if key in p]
            if vals and all(v.ndim >= 1 and v.shape[0] == len(p["timestamps"])
                            for v, p in zip(vals, [p for p in parts if key in p])):
                cat[key] = np.concatenate(vals)[order]
            elif vals:
                cat[key] = vals[0]
        out.append(cat)
    scored = [int(sum(int((p["carried_masks"] > 0.5).sum(axis=0)[k]) for p in out
                      if "carried_masks" in p)) for k in range(len(horizons))]
    real_scored = [int(sum(int((a["masks"] > 0.5).sum(axis=0)[k]) for a in real_arrays))
                   for k in range(len(horizons))]
    return out, {"mode": "swap", "test_n_scored": scored, "test_n_scored_real": real_scored}


def anchor_columns(entity_ids: np.ndarray, timestamps: np.ndarray, values: np.ndarray, *,
                   kind: str, lookback: int = 16, source: str = "own") -> np.ndarray:
    """SPEC_V2 A4 value-anchor inputs, causal, per position.

    ``kind="last"``: the token's own value, (n, 1).
    ``kind="skip"``: (n, 2·lookback) = [values | log elapsed times]; slot 0 is the current position,
    slot j the j-th earlier one.  ``source="own"`` walks the firing entity's own observations;
    ``source="tape"`` walks the tape (every entity), the window the W5 ridge floors regress on
    (``scripts/baseline_linear.py::windows``).  Missing history is left-padded with the first value
    (the floors' convention) at the clipped maximum gap.  Only positions ≤ i enter, and every target
    lies after i, so no target leaks.
    """
    v = np.asarray(values, dtype=np.float32)
    n = len(v)
    if kind == "last":
        return v[:, None].copy()
    if kind != "skip":
        raise ValueError(f"unknown anchor kind {kind!r}")
    if source not in ("own", "tape"):
        raise ValueError(f"unknown anchor source {source!r}")
    t = np.asarray(timestamps, dtype=np.float64)
    e = np.asarray(entity_ids)
    L = int(lookback)
    vals = np.empty((n, L), dtype=np.float32)
    gaps = np.empty((n, L), dtype=np.float32)
    groups = ([np.arange(n)] if source == "tape" else
              [np.flatnonzero(e == k) for k in np.unique(e)])
    for idx in groups:
        if not len(idx):
            continue
        ve, te = v[idx], t[idx]
        wv = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(L - 1, ve[0]), ve]), L)[:, ::-1]
        wt = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(L - 1, -np.inf), te]), L)[:, ::-1]
        el = te[:, None] - wt                                       # +inf on padded slots
        vals[idx] = wv
        gaps[idx] = np.clip(np.log(np.maximum(el, DT_FLOOR)), -LOG_DT_CLIP, LOG_DT_CLIP)
    return np.concatenate([vals, gaps], axis=1)
