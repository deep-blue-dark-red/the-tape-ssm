# Fused-Transfer Tape for OpenSky — corrected architecture plan

Planned 2026-10-15. Corrects the seven defects of the v1 adapter (BIG_DATASETS.md
§"OpenSky tape adapter"; critique 2026-10-15). **"Fused transfer"** means two
things, deliberately:

1. **Fused entity representation** — the per-entity embedding is decomposed into
   *observed structure priors* (aircraft type, route) fused with a small
   per-airframe residual, instead of a monolithic learned table rediscovering
   type clusters at open-population cost.
2. **Transfer as the evaluation spine** — OpenSky gives free transfer axes
   (next day, other region, disjoint airframes). The backbone + priors carry
   over; only light components re-fit. The paper number is what structure alone
   buys zero-shot.

Design constraint: reuse existing TapeConfig machinery wherever possible
(`cross_entity`, intensity head, `loss_mode`, time-feature slots); every new
module lands in MLX + torch lockstep with parity tests.

## 0. Defect → fix map

| # | v1 defect | fix | § |
|---|---|---|---|
| 1 | icao24 table rediscovers type; no unseen-airframe generalization | fused priors: type + factorized route + gated airframe residual | 3 |
| 2 | (aircraft, hour) tapes chop flights at parquet boundaries | flight tapes stitched via `flights_data4`; scene tapes for interactions | 1 |
| 3 | censoring (receiver dropout, bbox exit) entangled with dynamics | sensor features + missing-marks retained as inputs; stale kept as flag | 2 |
| 4 | single channel; k-th-firing horizons; h=100 exceeds p5 tapes | multi-channel marks; time-based horizons {30 s, 5 min, 15 min}; dt/intensity co-head | 2 |
| 5 | no local/global structure at all | scene tapes + `cross_entity="latent"/"grouped"` + day-state broadcast | 4 |
| 6 | hour split mislabeled open-population; night/dawn confound | three explicit transfer axes incl. airframe-disjoint; daytime windows | 5 |
| 7 | dedup-to-last arbitrary; cache drops provenance | dedup to max-`serials` row; cache carries icao24/flight/block metadata | 2 |

## 1. Records: flight tapes and scene tapes

Two granularities, one loader.

- **Flight tape (spine record).** One tape = one flight. Stitch per-icao24
  events across hour files, cut at `flights_data4` boundaries (`icao24` +
  `firstSeen`/`lastSeen`, day partition). Fallback when no flight row or the
  stitch rate is poor: gap-split at >300 s (the DB's own stale-retention
  window). This removes the storage-partition leak (defect 2).
- **Scene tape (interaction record).** One tape = (spatial block, time window):
  all flights' events interleaved, `entity_ids` = global icao24 — exactly the
  synthetic-generator tape layout, so the existing bucketed cross-entity
  machinery applies without model changes. Blocks = bbox grid over the region
  (TMA-scale, ~1°; REGIONS becomes a grid, not one box). Windows = 15 min.
- E7-comparable single-entity benchmark keeps flight tapes; the local/global
  claim runs on scene tapes. Same events, two groupings — no double ingestion.

## 2. Channels, marks, censoring, targets

- **Channels** (per event): `baroAltitude`, `geoAltitude`, `velocity`,
  `vertRate`, heading as (sin, cos), position as per-tape local ENU x/y in km
  (origin = first observation; keeps scale sane per region). `n_features`
  becomes C + 2 clocks + marks instead of 3.
- **Missing ≠ absent.** Events with non-finite channel values are *kept* with a
  per-channel missing mark (0/1 features); v1 deleted the event and its
  timestamp, thinning the process exactly where quality is worst. NaN→0 the
  value, carry the mark.
- **Sensor features** (defect 3): `log(1 + #serials)`, stale flags for both
  conventions (`time − lastcontact > 15` — docs canon; `time − lastPosUpdate >
  60` — v1) as inputs; only the canon-stale rows are dropped. Dedup
  (icao24, time) keeps the row with the most `serials`, not "last".
- **Targets — time-based, not index-based** (defect 4): value at
  t_i + Δ for Δ ∈ {30 s, 5 min, 15 min}, taken at the first event with
  t ≥ t_i + Δ, masked when coverage ends first. k-th-firing horizons stay as a
  secondary spec for E7 comparability only.
- **Two heads**: point/gaussian for values + the existing `intensity` head for
  inter-event times (the *when* is half the phenomenon). New `co_head: bool`
  config field; intensity already emits per-entity λ_k(τ|h).

## 3. Entity representation: fused priors + residual

```
e(i) = LN( E_type(type_i)  +  E_route(o_i, d_i)  +  g_i ⊙ E_air(i) )
E_route(o, d) = E_arp(o) + E_arp(d)            # factorized: ~10^4 airports, not ~10^6 routes
g_i = σ(w · [n_events_i, seen_i])              # gate: unseen/starved → structure only
```

- `E_type`: from OpenSky's aircraft-metadata DB (manufacturer/model/class) — a
  few hundred types. `E_arp`: airport table. `E_air`: per-icao24 residual,
  initialized zero, trained only on seen airframes; `seen_i = 0` for holdouts.
  Hashed-bucket residual is an ablation arm, not the default.
- **Relation to the X1/C kill**: compositional sharing lost for starved
  *synthetic* entities (no regime where sharing wins). Here attributes are
  *observed metadata* and the claim is zero-shot transfer to *unseen* entities —
  a different mechanism, but the null risk is real, so E1 is pre-registered
  with a revert condition (§8).
- **M2 lesson encoded**: the identity anchor is now decomposed by construction;
  the M2 probe (permuted-A, attribution) transfers verbatim to ask *which part*
  carries the anchor — type, route, or residual.

Config surface: `entity_priors: "none"|"type"|"route"|"type,route"`,
`prior_gate: bool`; `extra["entity_type_ids"]`, `extra["entity_airport_ids"]`
(dep/arr per flight tape). Monolithic dense table remains the control arm.

## 4. Local and global context

- **Local**: scene tapes, events binned into 4 s buckets; `cross_entity="latent"`
  (K pooled latents broadcast back) as default, `"grouped"` with
  `cross_group_ids` = altitude-band × block as the structured arm.
  `cross_scope="bucket"` keeps causality (context = same-bin tokens only);
  the `st` causal joint window is an ablation.
- **Global**: a per-window day-state vector — time-of-day (sin/cos), block
  traffic count, sector counts — broadcast additively into tokens (the
  "broadcast value" pattern from BIG_DATASETS §published fits). Cheap, leak-free
  (computed from train-side windows or lagged counts).

## 5. Transfer protocol

Three axes, all leak-free by construction:

| Axis | Source → target | What moves | What's new |
|---|---|---|---|
| Temporal | day A → day A+1 (same region) | backbone, priors, seen-airframe rows | unseen airframes + new flights |
| Region | `de` → `us-east` (same day) | backbone, type priors | routes/airports, receiver geometry |
| Population | airframe-disjoint split of one day | backbone, priors | all airframes unseen (E_air idle) |

Stages per axis: **S0** source training → **S1** zero-shot (everything frozen,
residuals zeroed for unseen) → **S2** few-shot (re-fit residuals for airframes
seen in the target's first 2 h, backbone frozen) → **S3** full finetune (upper
bound). Report the S1/S3 and S2/S3 ratios — that *is* the fused-transfer claim.
Default windows are daytime hours (defect 6); night/dawn is a labeled stress
arm, not the default spec.

## 6. Trainer/config additions

- `TapeConfig`: `entity_priors`, `prior_gate`, `co_head` (§2–3); validation
  that priors imply the corresponding `extra` ids (pattern of
  `cross_group_ids`).
- Trainer (both backends): `--transfer-stage {full,s1-zeroshot,s2-fewshot}`,
  parameter-group freezing, residual-row re-fit in S2. State dict must round-
  trip S0→S1 unchanged (pinned by test).
- Adapter v2: `opensky2-<region>-<day>-<granularity>-<channels>` grammar;
  cache carries `icao24`, flight id, block id, day-state per tape (defect 7).

## 7. Build order

- **P0** flight stitching + multi-channel + marks + sensor features (+tests).
  **DONE 2026-10-15** — `data/opensky_datasets.py` v2 (`opensky2[-REGION][-Nh][-TARGET]`),
  `build_tape_arrays` generalized to 2-D values + marks + aux (features = 2C+A+2 = 21),
  `record_chunks` width-aware, dedup-to-max-serials, canon-stale (`time − lastContact >
  15 s`), flights-data4 stitching with merge + >300 s orphan gap-split, per-channel
  train-only z-score. Real-data (de, full day): 24,271 flight tapes / 5,483 airframes /
  31.5 M events, 8–13 % channel missingness marked. Tests: `tests/test_opensky.py` (9).
  Two findings: (a) the public bucket holds exactly ONE snapshot day (S3 listing
  `data-samples/trino-tables/` — 24 hour-dirs, 1 flights day), so the temporal transfer
  axis waits for Trino access or a future day; within-day axes (region,
  airframe-disjoint, flight-start chronological) carry the E-series until then.
  (b) full-day scale OOMs the chunk pipeline on the Mac (~39 k train chunks ×
  intermediates ≈ >6 GB, exit 137) — P1+ needs streamed/lazy chunk building; the
  `-Nh` knob (6 h/12 h specs) is the interim workaround, 1-epoch smoke green at 6 h
  (test MSE h1 0.025).
- **P1** time-based horizon targets (reuse `future_targets` masking pattern).
- **P2** metadata joins (aircraft DB, flights → airports) + prior tables.
- **P3** scene tapes + block grid + day-state.
- **P4** transfer stages in trainer.
- **P5** intensity co-head wiring.
P0–P1 unblock E-series arms that need no model change; P2–P5 each land with
parity tests. Data prerequisite: ≥3 snapshot days on disk (current: 1, and the bucket
offers no more); the weekly 24 h product is *not* a substitute for the spine (10 s
quantization destroys event-time structure) — use it only for a scale ablation.

## 8. Experiments (pre-registered bars)

- **E1 priors vs monolithic vs none**, airframe-disjoint: priors must beat the
  monolithic table by > 2 paired sd on ≥ 2 of 3 time horizons, else revert to
  dense-table + hashed-unseen fallback. (Guards the X1-repeat risk.)
- **E2 scene vs flight tapes** on short-horizon position error in congested
  blocks: scene context must beat flight + day-state-only by > 2 paired sd,
  else drop the scene granularity (local claim dies, global claim survives).
- **E3 transfer decay**: S1/S2/S3 curves on all three axes; headline figure.
- **E4 censoring ablation**: marks + sensor features on, off; must help most in
  the lowest `#serials` tercile (bucketed read-out, M4-style).
- **E5 event timing**: intensity co-head vs protocol-matched TPP baseline
  (EasyTPP-style) on dt NNL/MAE.
- Frequency weighting α = 0.5 default carried over (M4/A lesson — starved
  entities redistribute, aggregate flat).

## 9. Test gates

- T1 stitch correctness on synthetic gap/flight-boundary fixtures; stitch-rate
  report on real data (fallback threshold pre-set).
- T2 causality: adversarial no-leak test for scene bucketing (perturbing
  future-bin tokens cannot change past outputs) — the T5 pattern from the
  sparse-tape plan.
- T3 MLX/torch parity for prior-fusion and co-head modules.
- T4 zero-shot path: unseen airframe → finite outputs, `E_air` unused
  (zero-gradient assertion).
- T5 time-target masking: no target leaks from beyond coverage end.
- T6 cache invalidation on the new grammar; metadata survives round-trip.

## 10. Risks

- `flights_data4` is *inferred nightly* — firstSeen/lastSeen drift vs raw
  events; measure stitch rate, gap-split fallback, report both.
- `serials` arrays bloat parquet reads; read count + top-1 community hash only.
- Aircraft DB coverage gaps for general aviation → "unknown" type bucket (its
  own embedding; must not silently merge with a real type).
- License: non-profit research, no raw redistribution — publish derived
  statistics and model weights' priors, not raw pulls.
