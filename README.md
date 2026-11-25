# TapeWorld

Research implementation of **The Tape**: a single shared sequential backbone
(Mamba-3, linear attention, S5 or KDA) fed one interleaved, time-ordered stream of events
from many entities. Each entity gets a learned identity embedding, and the loss is
*sparse*: at every position only the entity that actually fired is scored.
Gradient from entity B's loss flows back through entity A's earlier tokens,
which is what lets the model learn cross-entity structure.

Two backends implement the same architecture from one `TapeConfig`:

- **MLX** on Apple Silicon (`--backend mlx`, the default on macOS).
- **PyTorch** on CUDA or CPU (`--backend torch`). On CUDA the Mamba-3 chunked
  scan runs on hand-written Triton kernels with custom backward passes.

`tests/test_backend_parity.py` keeps the two in lockstep.

**Contents:** [Setup](#setup) · [Quick start](#quick-start) · [Benchmarks](#benchmarks) ·
[Datasets](#datasets) · [Results](#results) · [Experiments workflow](#experiments-workflow) ·
[CUDA path](#cuda-path-triton-kernels) · [Architecture](#architecture) ·
[Project layout](#project-layout) · [Tests](#tests)

> **Read before quoting a number.** `AGENTS.md` lists the adopted defaults (since
> 2026-11-19: `--proj no_layernorm`, sparse-loss rows train and select on MSE) and the
> known pipeline defects found 2026-11-02, with the flags that reproduce older rows
> (`--proj mlp`, `--value-loss gauss-huber --select-on loss`, `--eval-drop-last`,
> `--no-reset-records`, `TAPEWORLD_STATE_PASSING=ref`). Every results JSON records the
> settings it ran with.

---

## Setup

Python 3.10+. The project is managed with [uv](https://docs.astral.sh/uv/), and `uv.lock`
is checked in.

```bash
# Apple Silicon (MLX backend)
uv sync                  # mlx, numpy, matplotlib, pandas, flash-linear-attention
uv sync --extra dev      # + pytest, mypy, torch (CPU) for MLX↔torch parity tests

# Linux + NVIDIA (PyTorch backend; torch resolves to CUDA 12.8 wheels, Triton comes with it)
uv sync --extra torch
uv sync --extra dev      # + pytest

# Optional extras
uv sync --extra pyrregular   # read the pyrregular .h5 files (irregular benchmarks)
uv sync --extra data         # pyopensky + traffic (OpenSky tooling)
uv sync --extra embed        # sentence-transformers, for scripts/embed_entity_text.py
```

Extras do not stack across separate `uv sync` calls. Pass them together
(`uv sync --extra dev --extra pyrregular`), and run with `.venv/bin/python` rather than
bare `uv run` so that a re-sync does not drop them.

Every command below can be run as `.venv/bin/python run.py ...`, or as plain
`python run.py ...` with the `.venv` activated. On a CUDA box, add `--backend torch`
or set `TAPEWORLD_BACKEND=torch` once.

### Shared flags

| Flag | Meaning |
|------|---------|
| `--backend mlx\|torch` | Compute backend (default: mlx on macOS, torch elsewhere) |
| `--device cuda:1` | torch device (default: cuda if available, else cpu) |
| `--no-compile` | Disable `mx.compile` / `torch.compile` of the train step |
| `--amp` | torch: bf16 autocast for the forward pass |
| `--no-triton` | torch: use the pure-PyTorch SSD reference instead of the Triton kernels |
| `--n-seeds K` | Repeat for seeds `seed..seed+K-1`, report mean ± std |
| `--schedule` | `plateau` (**default**: reload the best weights and multiply LR by `--plateau-factor` when val stalls for `--patience` epochs, at most `--plateau-max-reductions` times), `warmup_cosine`, `trapezoid`, `constant`, `epoch_decay` |
| `--lr` | Initial LR (default `1e-3`); `--skip-lr-mult` (5) and `--head-lr-mult` scale the linear-skip / head groups |
| `--chunk-size` | SSD chunk length; `0` (default) = 64 on torch, 32 on MLX. Chunking is exact up to fp rounding |
| `--compile-mode` | torch: `reduce-overhead` (default, CUDA graphs), `default`, `max-autotune` |
| `--no-tf32` | torch/CUDA: IEEE fp32 GEMMs and SSD dots instead of TF32 (TF32 is the training default; ~1e-3 relative error, several times faster) |
| `--sched-epochs` | Epoch count the cosine/trapezoid annealing is sized for (default `--epochs`) |
| `--patience` | Early-stopping patience in epochs (0 = off) |
| `--ema` | Per-step EMA decay of the weights; validation, selection and test use the averaged weights (**on by default at 0.99**) |
| `--betas`, `--lr-floor` | AdamW betas, final-LR fraction |
| `--results-json` | Where to write the results JSON (default `outputs/results/<name>_<time>.json`) |
| `--checkpoint` | Gradient checkpointing per block on both backends (`tests/test_checkpoint.py`) |
| `--resid-dropout p` | Dropout on every residual-branch output inside the blocks (`--dropout` is token-level only) |
| `--backbone` | `mamba3` (default), `mamba3-official` (wraps `state-spaces/mamba` Mamba3, CUDA only, SISO or MIMO via `--mimo-rank`), `linear_attention`, `per_entity_attn`, `s5`, `kda` |
| `--block-pattern` | Interleave block types per layer (mixed S5 / Mamba-3 stacks, E29) |
| `--scan-order g\|e` | Global time order (default) or per-entity sub-tape order (E28) |
| `--readout {last,mean,attn}` | How the pooled RUL state is read out |
| `--embed-mode {dense,compositional}` | Dense per-entity tables, or compositional embeddings over `G` shared atoms (`--n-atoms`, `--embed-k`, `--embed-assign`, `--embed-coef`, `--embed-shrink`) |
| `--backbone-routed` | MoM-style routed recurrent memory backbone (`--n-memories`, `--memory-update`, `--write-router`), optional write-error retrieval (`--retrieval we`) |

Benchmark-specific options (see `documents/improvement_ideas.md`):
`ltsf --window-norm {none,mean,last} --linear-skip --order-aug --order-ensemble K --calendar --loss-space {window,raw} --train-months N --eval-stride N`;
`cmapss --regime-norm --dense-rul --late-penalty w --select score --window-size W --val-frac f --op-settings`.
`--loss-space` defaults to **`raw`**. The plateau schedule, EMA 0.99 and `--skip-lr-mult 5`
together are the `add_train_args` default.

### Environment switches

| Variable | Effect |
|---|---|
| `TAPEWORLD_BACKEND=torch` | Default backend |
| `TAPEWORLD_STATE_PASSING` | torch inter-chunk recurrence: unset/`blocked` (two-level, block 64, **default since 2026-11-18**), `ref` (one-level form every earlier row used), `loop` (sequential) |
| `TAPEWORLD_SP_BLOCK` | Block size for `blocked` (keep 64: 128/256 were slower and close to OOM on 16 GB) |
| `TAPEWORLD_SSD=reference` | Force the pure-torch SSD instead of Triton |
| `TAPEWORLD_TF32=1` | Triton kernels default to TF32 dots |
| `TAPEWORLD_S5_SCAN=reference` | Force S5 onto its torch chunked scan |
| `TAPEWORLD_MLX_SSD=loop` | Sequential per-chunk MLX forward (the vectorised one is default and ~2× faster) |
| `TAPEWORLD_SYNTH_CACHE=<dir>` | On-disk cache for generated Hawkes tapes (`scripts/pregen_hawkes.py` fills it) |
| `TAPEWORLD_TRAFFIC_ROOT`, `TAPEWORLD_OPENSKY_ROOT` | Override the METR-LA/PEMS-BAY and OpenSky data roots |

Results JSONs record every `TAPEWORLD_*` switch under `tapeworld_env`.
`scripts/profile_step.py` / `scripts/profile_step_mlx.py` time one training step.

---

## Quick start

`run.py` is the single entry point, and each subcommand has its own `--help`.

```bash
python run.py --help
python run.py synthetic --help
```

| Command          | What it does |
|------------------|--------------|
| `synthetic`      | Train on simulated multi-entity event streams (OU, Hawkes, Lotka-Volterra, switching LDS) |
| `ltsf`           | Long-term time-series forecasting (ETT, Weather, Exchange, Electricity, Traffic) |
| `cmapss`         | NASA C-MAPSS turbofan remaining-useful-life benchmark |
| `irregular`      | Irregular / asynchronous benchmarks (t-PatchGNN trio, Physiome-ODE, EasyTPP, Time-IMM, pyrregular, OpenSky, thinned LTSF panels) |
| `tpp`            | Marked temporal point processes (EasyTPP): intensity head, per-event log-likelihood, next-event RMSE |
| `stformer`       | STformer / NSTformer reproduction (arXiv 2408.09158) on METR-LA / PEMS-BAY |
| `plot-exchange`  | Train and plot 1-step exchange-rate predictions |
| `plot-multistep` | Train and plot autoregressive multi-step exchange-rate predictions |

```bash
python run.py synthetic --dataset hawkes --epochs 30
python run.py ltsf --dataset ETTh1 --all-horizons --n-seeds 3
python run.py cmapss --all-subsets --backend torch
python run.py irregular --dataset physionet --dt-feature 2
python run.py tpp --dataset taxi --backend mlx
python run.py stformer --dataset metr-la --model nstformer --backend torch
```

---

## Benchmarks

### Synthetic (simulated) streams

```bash
python run.py synthetic                       # 16 coupled OU entities, 100k events, 30 epochs
python run.py synthetic --dataset hawkes      # multivariate Hawkes, cross-excitation
python run.py synthetic --dataset lv          # Lotka-Volterra predator-prey
python run.py synthetic --dataset switching   # switching linear dynamical system

# Ablations
python run.py synthetic --no-entity-embed     # A1: zero the entity embeddings
python run.py synthetic --dense-loss          # A2: uniform entity weight (see documents/dense_sparse_confound.md)
python run.py synthetic --no-freq-reweight    # A3: no per-entity frequency reweighting

# Backbone and scale
python run.py synthetic --backbone linear_attention
python run.py synthetic --n-entities 32 --n-events 500000 --d-model 128 --n-layers 6
```

Before any Hawkes run, read the E1c OUTCOME in `SPEC_PAPER_NOV.md` (2026-11-06). Every
M=64 Hawkes config has branching radius ρ = 2.14–2.39, and coupling strength 0.6 gives
1.16–1.73. Both are above the stationarity bound, so those rows are flagged uncertified. The
M=16 default (ρ 0.64–0.83) is stationary in practice.
`data/generators/hawkes_process.py::branching_radius` computes ρ.

### LTSF

```bash
bash scripts/download_data.sh          # ETT + Exchange + Weather + Electricity + Traffic

python run.py ltsf --dataset ETTh1 --horizon 96
python run.py ltsf --dataset ETTh1 --all-horizons --n-seeds 3   # 96, 192, 336, 720
python run.py ltsf --dataset traffic --max-vars 64              # subset the wide panels
python run.py ltsf --dataset electricity --max-vars 32 --backend torch
python run.py ltsf --dataset exchange_rate --log-transform
```

Dataset names are case-sensitive and match the file stems in `data/ltsf_raw/`:
`ETTh1 ETTh2 ETTm1 ETTm2 weather traffic exchange_rate electricity`.

Protocol: the model is trained with the sparse next-step loss at every position **plus a
direct multi-horizon head**, which predicts each entity's next H values from its last
position. Reported MSE/MAE come from that head over every test window (stride 1), at the
epoch with the best validation MSE. That is the published LTSF convention.
`--no-direct-head` restores the older evaluation, which repeats the 1-step prediction
H times. The split is the corrected TSLib border
(`data/ltsf_datasets.py::standard_split`, 2026-09-07); every number measured before that
date is void. `--train-months N` keeps only the last *N* 30-day months of train
(L-MTSF protocol, normaliser refitted).

### C-MAPSS

```bash
bash scripts/download_cmapss.sh

python run.py cmapss --subset FD001
python run.py cmapss --all-subsets --n-seeds 3
python run.py cmapss --subset FD002 --rul-only          # drop the world-model loss
python run.py cmapss --alpha-world 0.5 --alpha-rul 0.5  # reweight the two losses
```

Protocol: 20 % of the *training engines* (`--val-frac 0.2`) are held out, each truncated
at five random cycles. The epoch with the best validation RMSE is the one evaluated on the
test set. (Early versions of `documents/cmapss_results.md` selected on the test set; treat
those figures accordingly.)

### Irregular / asynchronous

```bash
bash scripts/download_irregular.sh   # → data/irregular/ (~6.8 GB, git-ignored)

python run.py irregular --dataset physionet --dt-feature 2
python run.py irregular --dataset physiome --physiome-model dupont_1991a --fold 0
python run.py irregular --dataset GDELT --horizons 1,10
python run.py irregular --dataset pyrregular-taxi
python run.py irregular --dataset opensky-de-6h-alt          # opensky[-REGION][-Nh][-CHANNEL]
python run.py irregular --dataset ltsf-thin-electricity --thin-frac 0.8
```

Every observed `(time, channel, value)` triple of a record becomes one tape event, with
entity = channel (event type for EasyTPP). The model predicts each entity's value at its
k-th next observation (`--horizons`). Short records are packed into bins of `--chunk-len`
events, and `--reset-records` (default on) zeroes the scan state at each record boundary.
This is not the t-PatchGNN windowed-forecast protocol, so MSE is not comparable to their
Table 2. See `documents/irregular_benchmarks.md`.

### Marked temporal point processes (EasyTPP)

```bash
python run.py tpp --dataset taxi
python run.py tpp --dataset retweet --backend mlx
```

Datasets: `taxi`, `stackoverflow`, `retweet`, `taobao`, plus the E14 tie-broken copies
`stackoverflow_jitter` and `retweet_jitter`. The command trains the Tape as an intensity
head and reports per-event log-likelihood and next-event RMSE against the S2P2 control.
E14 (2026-11-05) found that the StackOverflow and Retweet log-likelihoods are inflated by
same-timestamp ties. Quote the `_jitter` rows for those two sets.

### STformer / NSTformer reproduction

```bash
python run.py stformer --dataset metr-la --model nstformer --backend torch
python run.py stformer --dataset pems-bay --model stformer --backend torch
```

Trains on the authors' released windows, split indices, scaler and adjacency. The first
run downloads them; see `documents/stformer_reproduction.md`.

### Cross-entity mixing and sparsity flags

| Flag | Meaning |
|------|---------|
| `--cross-entity {none,mean,latent,full,grouped,linattn}` | Entity-mixing module after each block: global pool, K latent pools, O(M²) pairwise, per-group pools, or kernel linear attention |
| `--cross-latents K`, `--cross-groups G`, `--cross-scope {bucket,st}` | Mixing sizes and linattn scope |
| `--diagnose-cross-entity` | Record the half-mask ablation and learned-step-size diagnostics |
| `--thin-frac p`, `--thin-seed S` | Drop a fraction *p* of (entity, timestep) cells (same draw per split); masked inputs imputed to the observed mean |
| `--dt-gaps`, `--dt-normalise` | Feature 2 becomes log gap-to-last-observation, or log(1/M) |
| `--dt-feature i` | Add the log Δt feature to the step-size pre-activation (continuous-time bias) |

Mixing helps at M=64 (`mean`). It does not transfer to width (862 entities), so the claim
is scoped to M=64. Keep the default `none` unless you are running that program.

### Exchange-rate plots

```bash
python run.py plot-exchange                       # → outputs/exchange_rate_predictions.png
python run.py plot-multistep --n-steps 10         # → outputs/exchange_rate_multistep.png
```

---

## Datasets

Everything under `data/` except the ETT, Weather and Exchange CSVs (and the entity-text
priors) is git-ignored and has to be fetched. This section lists, for every dataset the
runners read, where it comes from, how to get it, where it lands, and the SHA-256 of the
exact bytes the published rows were trained on.

### Verifying the files

The manifests are in `data/checksums/<collection>.sha256`, in the standard `sha256sum`
format with paths relative to the repo root:

```bash
python3 scripts/hash_datasets.py --check              # every collection
python3 scripts/hash_datasets.py --check ltsf cmapss  # a subset
shasum -a 256 -c data/checksums/ltsf.sha256           # or without the script (repo root)
python3 scripts/hash_datasets.py                      # rewrite the manifests after adding data
python3 scripts/hash_datasets.py --list               # what each collection covers
```

The **collection digest** in each heading below is the SHA-256 of that manifest file. One
hash therefore pins every file name and byte in the collection. Hashed 2026-11-25 on the
Mac data tree (985 files, 21.8 GB).

| Collection | Files | Size | Collection digest (SHA-256 of `data/checksums/<name>.sha256`) |
|---|---:|---:|---|
| `ltsf` | 10 | 0.42 GB | `4874a1607e6fa01499c985203db0efaaca3fe78fcc0ccad77d2d06324a649756` |
| `cmapss` | 14 | 0.06 GB | `2373093d2996ce5ac7b189d2d83b9481fa51a01fcfebe4a444912fa0134db53b` |
| `stformer-traffic` | 10 | 1.05 GB | `ed75fb4959168252370a88ceb18c4f7a3dffacdbad0b81a9336f6e019e755fb5` |
| `tpatchgnn` | 8 | 0.41 GB | `13279fff3221b0770f73c0961faed46a7cd544cede9e294b8a0607b65004cc0a` |
| `physiome-ode` | 750 | 4.73 GB | `b1ec20cd7d01cbe8ab90bd414317760175fa7cacd2d51faec9a3968add323827` |
| `easytpp` | 12 | 0.06 GB | `44985f672b4689a6ec8508c2d3167c814593412656f2e4e70bc8c6b8c22b8edd` |
| `easytpp-jitter` (derived) | 8 | 0.12 GB | `d541d98d07dfd3d22f8bafc5605e69a64649081f3f1c57932ecfd295ff90bb0a` |
| `timeimm` | 135 | 0.06 GB | `70daf55f5ed2080e0c23f22f7c9b82a53d8ee0ba5e88443b7a920fd2ccdc2ff8` |
| `pyrregular` | 10 | 1.52 GB | `f58a037ce99196c283156407500fbdac6b7bed43d4d23a83fe927614227f2a44` |
| `opensky` | 28 | 13.42 GB | `8abe32ea638f0db0a01869ccad0558e7939ef63ee4f1867c778c87bbd2e4577c` |

Not hashed: the simulated datasets (they are produced by code; see
[Simulated datasets](#simulated-datasets-not-hashed)), and the derived caches
`data/traffic/processed/*.npz` and `data/opensky/processed/*.npz`, which are rebuilt from
the hashed raw files on first load.

Licences are those of the upstream sources. Check each source before redistributing.

### Overview

| Benchmark | Dataset | Entities × length | Source | Get it with |
|---|---|---|---|---|
| `ltsf` | ETTh1, ETTh2 | 7 vars × 17,420 h | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) | committed; `scripts/download_data.sh` |
| `ltsf` | ETTm1, ETTm2 | 7 vars × 69,680 (15 min) | same | committed; `scripts/download_data.sh` |
| `ltsf` | Weather | 21 vars × 52,696 (10 min, 2020) | Autoformer bundle (MPI-BGC Jena) | committed; `scripts/download_weather.sh` |
| `ltsf` | Exchange | 8 currencies × 7,588 days | Autoformer bundle (Lai et al. 2018) | committed; `scripts/download_data.sh` |
| `ltsf` | Electricity | 321 clients × 26,304 h | Autoformer bundle (UCI) | `scripts/download_electricity.sh` |
| `ltsf` | Traffic | 862 sensors × 17,544 h | LSTNet / Autoformer bundle (PeMS) | `scripts/download_traffic.sh` |
| `cmapss` | C-MAPSS FD001–FD004 | 100/260/100/249 train engines | NASA PCoE | `scripts/download_cmapss.sh` |
| `stformer` | METR-LA | 207 sensors × 34,272 (5 min) | STformer authors (BasicTS) | automatic on first run |
| `stformer` | PEMS-BAY | 325 sensors × 52,116 (5 min) | STformer authors (BasicTS) | automatic on first run |
| `irregular` | PhysioNet 2012, Activity, USHCN | 41 / 12 / 5 channels | [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) | `scripts/download_irregular.sh` |
| `irregular` | Physiome-ODE | 50 ODE models × 5 folds | [Zenodo 11492058](https://zenodo.org/records/11492058) | `scripts/download_irregular.sh` |
| `irregular`, `tpp` | EasyTPP Taxi, StackOverflow, Retweet, Taobao | event types as entities | [HF `easytpp/*`](https://huggingface.co/easytpp) | `scripts/download_irregular.sh` (Taobao by hand) |
| `irregular` | Time-IMM (8 sets) | 1–30 entities each | [Time-IMM](https://github.com/blacksnail789521/Time-IMM) | `scripts/download_irregular.sh` |
| `irregular` | pyrregular (10 sets) | 57–121,312 records | [HF `splandi/pyrregular`](https://huggingface.co/datasets/splandi/pyrregular) | `scripts/download_irregular.sh` |
| `irregular` | OpenSky 2026-03-01 | open aircraft population | [OpenSky Network](https://opensky-network.org/) | `scripts/download_opensky_*.sh` (**snapshot withdrawn**, see below) |
| `synthetic` | OU, Hawkes, LV, switching LDS, interference | configurable | `data/generators/` | generated, not downloaded |

### LTSF (`data/ltsf_raw/`) — collection `ltsf`

These are the standard long-term forecasting panels, in the versions distributed with
Autoformer (Wu et al., NeurIPS 2021) as `all_six_datasets.zip`.

- **Autoformer bundle:** [Google Drive folder](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)
  or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/), linked from
  [thuml/Autoformer](https://github.com/thuml/Autoformer) and
  [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library). Unzip the
  CSVs into `data/ltsf_raw/`.
- **ETT:** [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (Zhou et al.,
  Informer, AAAI 2021), raw files at
  `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/<name>.csv`.
- **Primary sources:** Electricity is
  [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
  (aggregated hourly, 321 clients, 2012–2014). Weather is the
  [MPI-BGC Jena weather station](https://www.bgc-jena.mpg.de/wetter/) (2020, 10-minute).
  Exchange and Traffic come from Lai et al. (LSTNet, SIGIR 2018),
  [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).
  Traffic is Caltrans PeMS San Francisco Bay Area road occupancy for 2015–2016, also
  published as [Zenodo 4656132](https://zenodo.org/records/4656132) in Monash `.tsf` format.

| File | SHA-256 | Notes |
|---|---|---|
| `data/ltsf_raw/ETTh1.csv` | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` | byte-identical to the bundle |
| `data/ltsf_raw/ETTh2.csv` | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` | byte-identical to the bundle |
| `data/ltsf_raw/ETTm1.csv` | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` | byte-identical to the bundle |
| `data/ltsf_raw/ETTm2.csv` | `db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd` | byte-identical to the bundle |
| `data/ltsf_raw/weather.csv` | `9b2b19b13e342d9c9b9c51620d42b4be232df7fba30e7f5c4968b7c79ab138ad` | bundle copy with CRLF → LF (`tr -d '\r'`) |
| `data/ltsf_raw/exchange_rate.csv` | `d55e7aa2641009814a18ba3279431b13f6d413b0eab195b9ff21988d8cf94e97` | bundle copy with CRLF → LF |
| `data/ltsf_raw/electricity.csv` | `7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad` | byte-identical to the bundle |
| `data/ltsf_raw/traffic.csv` | `f39d0a6066d98206ce07227801e72ca381add78c683e858239f6ee7c398eae57` | Mac copy: header `sensor_0…861`, dates 2015-01-01…2016-12-31 |
| `data/traffic.csv` | `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16` | bundle copy (header `0…860,OT`); bigd symlinks `data/ltsf_raw/traffic.csv` to it |
| `data/ltsf_raw/all_six_datasets.zip` | `b9a0ecc12c38bcb7d531b00e56052ecf1d4fd046e1ad4a4fc237dd88ad19783d` | the Autoformer bundle itself |

The two Traffic files differ only in their header and date column. Both carry exactly the
values of LSTNet's `traffic/traffic.txt.gz` (17,544 × 862, max |Δ| = 0; checked
2026-11-25), so rows trained on either are comparable. The loader drops the date column.
The bundle's weather and exchange CSVs have CRLF line endings. Stripping the `\r` gives
the committed files byte for byte, and the parsed values are identical.
`data/national_illness.csv` (ILI, from the same bundle) is committed but no runner reads it.

Known script issues: the Exchange URL in `scripts/download_data.sh`
(`…/exchange_rate/exchange_rate.txt`) now returns 404 upstream (only `exchange_rate.txt.gz`
exists), and the Autoformer GitHub raw paths tried by `download_electricity.sh` /
`download_weather.sh` no longer exist. The Autoformer bundle above is the reliable source.

### C-MAPSS (`data/cmapss/`) — collection `cmapss`

The NASA Commercial Modular Aero-Propulsion System Simulation turbofan run-to-failure data
(Saxena et al., PHM 2008), four sub-datasets FD001–FD004 (train / test / true-RUL files,
26 columns: unit, cycle, 3 operating settings, 21 sensors).

- [NASA PCoE data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/),
  entry 6 "Turbofan Engine Degradation Simulation".
- Direct zip (PHM Society mirror):
  `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`.
- Kaggle mirror (the copy used here, `nasa-cmaps.zip`):
  [behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).

| File | SHA-256 |
|---|---|
| `train_FD001.txt` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `train_FD002.txt` | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `train_FD003.txt` | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` |
| `train_FD004.txt` | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| `test_FD001.txt` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `test_FD002.txt` | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `test_FD003.txt` | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` |
| `test_FD004.txt` | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| `RUL_FD001.txt` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `RUL_FD002.txt` | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| `RUL_FD003.txt` | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` |
| `RUL_FD004.txt` | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |
| `readme.txt` | `4f5270554b775c67e73aff383c5436fd329d6e4cc3d3a116913276fae511269b` |
| `nasa-cmaps.zip` (Kaggle archive) | `bb97bc5e283c38b869b227fcfda17772148113dd90b4d6fe0277864fd23e6c1a` |

### METR-LA / PEMS-BAY (`data/traffic/`) — collection `stformer-traffic`

These are road-speed panels from Los Angeles loop detectors (METR-LA, 207 sensors, March to
June 2012) and the Bay Area (PEMS-BAY, 325 sensors, January to May 2017), originally released
with DCRNN (Li et al., ICLR 2018). The reproduction uses the
**BasicTS-processed artifacts released by the STformer authors**:
windows of 12 in / 12 out, a 7:1:2 split by window position, a global train z-score, and the
DCRNN weighted adjacency. Using them makes the windows, split indices and scaler identical
to theirs.

- Source: [XinyuNing/STformer-and-NSTformer](https://github.com/XinyuNing/STformer-and-NSTformer),
  `datasets/{METR-LA,PEMS-BAY}/` (repo HEAD `384979f` on 2026-11-25).
- `data/traffic_datasets.py::load` downloads the zip and the pickles from
  `https://raw.githubusercontent.com/XinyuNing/STformer-and-NSTformer/HEAD/datasets/<NAME>/`
  on first use and caches `data/traffic/processed/<name>.npz` (not hashed; it is rebuilt from the raw files).
  `python data/traffic_datasets.py` fetches both.

| File | SHA-256 |
|---|---|
| `METR-LA/data.zip` | `fd36fd439561c2496df7a0f03b298c4af88dd960bc37570bc84659c67d24517f` |
| `METR-LA/data_in_12_out_12_rescale_True.pkl` | `590988182d2dcccf2303ebbcd27a93b342342e4afcce27de494d2663fce9c521` |
| `METR-LA/index_in_12_out_12_rescale_True.pkl` | `4a62a9d73e094a4b7ed9595f1c9c47c53aea74d746b05224f5eae118f5ec25ab` |
| `METR-LA/scaler_in_12_out_12_rescale_True.pkl` | `dcb863c95f88998f6c210f7b0d7ee136d129b3f253c8ce0f3b17c63801a8a80f` |
| `METR-LA/adj_mx.pkl` | `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3` |
| `PEMS-BAY/data_in_12_out_12_rescale_True.zip` | `9c8ecc4495f0228dbe0278612cb17c570f66238f1c1a69b27fb0095fafa393d1` |
| `PEMS-BAY/data_in_12_out_12_rescale_True.pkl` | `cb73d24be30d62fb3dbcda114111061de3fd6c21b80aea668bbfa7c908bb0aed` |
| `PEMS-BAY/index_in_12_out_12_rescale_True.pkl` | `cc2777b7d1347a467f01f17050e0273bf8377508544e3fc10f11119e447d04df` |
| `PEMS-BAY/scaler_in_12_out_12_rescale_True.pkl` | `e11b0a00a36e620c1221ab79ba7cc84f6b4a864cd84298f89709f2d9f93cfa45` |
| `PEMS-BAY/adj_mx.pkl` | `116275f5704d0d492018e14c047f7ae5004385b450aa907f88424055a4a97370` |

Paths are relative to `data/traffic/raw/`.

### t-PatchGNN trio (`data/irregular/tpatchgnn/`) — collection `tpatchgnn`

These are the processed irregular multivariate time-series (IMTS) sets shipped in the
t-PatchGNN repository (Zhang et al., ICML 2024):
[usail-hkust/t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN), `data/`
(cloned at HEAD; HEAD was `00c94e7` on 2026-11-25). `download_irregular.sh` shallow-clones
it and copies `activity/`, `ushcn/` and `physionet/*.zip`.

- **PhysioNet 2012**: ICU stays, first 48 h, 41 channels, sets a/b/c of 4,000 records each
  ([PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/), ODC-By).
- **Human Activity**: 12 channels from wearable localisation tags
  ([UCI 196](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity)).
- **USHCN**: daily climate, 1,114 stations, 5 channels, the De Brouwer et al. (GRU-ODE-Bayes)
  5 % subsample ([USHCN](https://cdiac.ess-dive.lbl.gov/epubs/ndp/ushcn/ushcn.html)).
- **MIMIC-III** (the fourth member) needs PhysioNet credentialed access and is not used:
  [mimic-iii-ext-tpatchgnn](https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/).

| File | SHA-256 |
|---|---|
| `physionet/processed/set-a_0.0.pt` | `07614a34127c0a8cdcf137d53de79ee99c6dc12ba7743a158bae43d7dc766629` |
| `physionet/processed/set-b_0.0.pt` | `195794998c1d5c5fc4eb9eb30e1cbbe34271f8c2cd3a4d15c60719a46bf25690` |
| `physionet/processed/set-c_0.0.pt` | `b3285c6fee890359271c081d031b26ed3e1f8a7e0fab0297b10df006ec7b7a42` |
| `physionet/raw.zip` | `77d277787425a23d26cd26f3d46f8d980c3882c840b5bf49b1d2aa9616a9bff8` |
| `activity/processed/data.pt` | `99131e95e612cc354cab3466e12793eb6c7b68460a386157db02509905a160aa` |
| `activity/raw/ConfLongDemo_JSI.txt` | `1fbc965b8d7faf499ea854428b2138132d94e6a26404bc8e4b0c62144f513e8a` |
| `ushcn/processed/ushcn.pt` | `06049d363c8bf7eabfe207ef76d73417a0e9b97467e1f060517f94f7eedf4d9a` |
| `ushcn/raw/small_chunked_sporadic.csv` | `671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf` |

Paths are relative to `data/irregular/tpatchgnn/`.

### Physiome-ODE (`data/irregular/physiome_ode/`) — collection `physiome-ode`

These are 50 irregular multivariate datasets generated from biological ODE models in the
Physiome repository, with observation dropping. Each has 2,000 instances and 5 fixed folds
(Klötergens et al. 2025, [arXiv 2502.07489](https://arxiv.org/abs/2502.07489), code
[kloetergensc/Physiome-ODE](https://github.com/kloetergensc/Physiome-ODE)). The values are
simulated, but the dataset is downloaded rather than generated here, so it is hashed.

- Data: [Zenodo record 11492058](https://zenodo.org/records/11492058) (CC-BY-4.0), file
  `final.zip`, 715,161,107 bytes, Zenodo `md5:f609a80851345874b8221d02ac458418`.
  The script deletes the zip after unpacking. Its SHA-256 was not recorded, so the
  manifest pins the 750 unpacked files instead.
- Layout: `final/<model>/<fold 0-4>/{train,valid,test}.pt` (50 models × 5 folds × 3 files).
  Per-file hashes are in `data/checksums/physiome-ode.sha256`.

### EasyTPP (`data/irregular/easytpp/`) — collections `easytpp`, `easytpp-jitter`

These are marked event sequences from Ant Research's EasyTPP benchmark
([ant-research/EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess)),
hosted on the Hugging Face hub as `easytpp/<name>` with `train.json`, `dev.json` and
`test.json`. Each event has `time_since_start`, `time_since_last_event` and `type_event`.

```bash
# fetched by download_irregular.sh for taxi, stackoverflow, retweet; taobao by hand:
for f in train dev test; do
  curl -L --fail -o data/irregular/easytpp/taobao/$f.json \
    https://huggingface.co/datasets/easytpp/taobao/resolve/main/$f.json
done
```

| File | SHA-256 |
|---|---|
| `taxi/train.json` | `0c54b53482e792f8cd813195e76e08466a68761b35c34cd503a012b41525d989` |
| `taxi/dev.json` | `9ae91594723cf9cba7333efa18f85d309f5ee76336cc25d5811d0364f05a3ddf` |
| `taxi/test.json` | `dc1e765b6626c2a231ab026606e29864c082807354066e0307494fb738bcc059` |
| `stackoverflow/train.json` | `809e68fbaba02f657b3dc2cb02ecb4df1f58c4fc8f9fbccdfa83defe71ef5413` |
| `stackoverflow/dev.json` | `4c4e595342c4c2624a77502695654ce3bec157242be30b6f3654575791741c6a` |
| `stackoverflow/test.json` | `e531fafc1cc2c109fcfe0a312e43beddda16725ec7c2a98b651609550f9ffb51` |
| `retweet/train.json` | `20f2358e346997e228c058ecaf76c09a3353863e0099816472a8bf80655b4e3c` |
| `retweet/dev.json` | `686a297e163486cc164083a6acc63badb332d41f782e198a01048869298eb419` |
| `retweet/test.json` | `3ea0967389716870966d2d32e996d8f5e784a6a5fa0c8b2c259ddfef41780e3a` |
| `taobao/train.json` | `aea30eea6f18e85e1cf0789fcf6ba748116fdbe05df77733779fe288bed8cd8f` |
| `taobao/dev.json` | `b5c4a2ed781c3c3cc4fc5118dd2a6eeb1ead93d342ed8ab4dcaeb8aa73528fe5` |
| `taobao/test.json` | `1efdd46d7bdf3efd00fb260e4b239ba2278dde03f455f71d30320aed48b3bb84` |

**Jittered copies (derived).** `stackoverflow_jitter/` and `retweet_jitter/` are built
locally by `scripts/make_jittered_easytpp.py <name> --seed 20261105`. The script adds
U(0, g_min) jitter per event and does a stable re-sort, which breaks same-timestamp ties
(E14). It is deterministic: the same seed reproduces the files byte for byte, and each
copy's `jitter.json` records the seed and the SHA-256 of every source and output file.

| File | SHA-256 |
|---|---|
| `stackoverflow_jitter/train.json` | `e37f21a917ba8b2f43a693b11a45d5fd9d23fcca58470c6cf72fddf0e70ee6d4` |
| `stackoverflow_jitter/dev.json` | `00a5744a3f667fdedcd4636046df3357117e7175b6b9951042ac84a55a2130b0` |
| `stackoverflow_jitter/test.json` | `bad889fe65326c5fd10cc52f38728c74461654a5d019308f2f25da306029e8d9` |
| `stackoverflow_jitter/jitter.json` | `5813aa1d07ea08819acdbf0273aa6a37ef95bb879eafdbdd5882e528f256a15d` |
| `retweet_jitter/train.json` | `5a86ae701a154c16402b4e51d6fca2456d6d47453734788f75827ee7b06e38ff` |
| `retweet_jitter/dev.json` | `afc58db50a59832bf856026b7ccf7ffed3297fdf0125786fdac4f10676d98ab8` |
| `retweet_jitter/test.json` | `f1b3482a178635284e4e7b3c53b48b5b5bab4e046e1af528e8cab073dc768f57` |
| `retweet_jitter/jitter.json` | `6d5db1c1d230e24cf5fd2cb4601af486b280ba1122c79630cf9d75af28359b0e` |

### Time-IMM (`data/irregular/timeimm/`) — collection `timeimm`

These are eight redistributable datasets covering nine cause-driven irregularity mechanisms
(Chang et al., NeurIPS 2025 Datasets & Benchmarks,
[arXiv 2506.10412](https://arxiv.org/abs/2506.10412)). Each ships
`data/<NAME>/processed/<entity>/time_series.csv` (+ `text.csv`, which the unimodal Tape
ignores).

- Source: [blacksnail789521/Time-IMM](https://github.com/blacksnail789521/Time-IMM), a git
  clone. The local checkout is commit **`e934cef23d526817dcf0d1a66c6f82274c96c04b`**
  (2026-01-05). Pin it with `git -C data/irregular/timeimm checkout e934cef`.
- Datasets: `GDELT` (8 entities), `RepoHealth` (4), `FNSPID` (10), `ClusterTrace` (3),
  `StudentLife` (20), `ILINet` (1), `CESNET` (11 shipped), `EPA-Air` (8). MIMIC-IV needs
  credentialed access and is absent (only its preprocessing notebook is in the repo).
- 135 files; per-file hashes are in `data/checksums/timeimm.sha256`.
- Defect 2 in `AGENTS.md`: Time-IMM timestamps are in seconds and log Δt is clipped at ±10.
  On FNSPID and StudentLife the entity gap is therefore a constant.

### pyrregular (`data/irregular/pyrregular/`) — collection `pyrregular`

This is a collection of 34 standardized, naturally irregular datasets (Spinnato & Landi,
*PYRREGULAR*, ICLR 2026, [arXiv 2505.06047](https://arxiv.org/abs/2505.06047)), stored as
xarray/HDF5 on the Hugging Face hub at
[splandi/pyrregular](https://huggingface.co/datasets/splandi/pyrregular), revision
`data-v1`. TapeWorld uses ten of them, fetched one file at a time from
`https://huggingface.co/datasets/splandi/pyrregular/resolve/data-v1/data_final/<Name>.h5`.
Reading them needs `uv sync --extra pyrregular` (verified against pyrregular 0.4.0).
Licences vary per dataset (CC-BY-4.0, CC0, CC-BY-NC-SA-4.0, unknown for the MSR GPS sets);
see the pyrregular metadata.

| `--dataset` | File | Bytes | SHA-256 |
|---|---|---:|---|
| `pyrregular-taxi` | `Taxi.h5` | 87,734,232 | `21463a6c138a0ea45558849ba872784fed6d770c2cb941b920408c31e31d330a` |
| `pyrregular-geolife` | `GeolifeSupervised.h5` | 92,618,387 | `eb11b2fbfc61d4aedfc027cac3e03887328e6163c7cb0cace8afdfcb331de3b8` |
| `pyrregular-vehicles` | `Vehicles.h5` | 2,751,833 | `d14c8f61784ed0a75d87fed31b60ef58e83f9f7b77155a242e576145ce225e5f` |
| `pyrregular-ldfpa` | `Ldfpa.h5` | 4,277,366 | `876fa3295ae08004de41b11d95a060d104a9966bd7ec90de1029398c03f583cc` |
| `pyrregular-mimic3` | `Mimic3.h5` | 264,871 | `2050a6b459af5c6eb0f4211e26b3a0ec525451d59dac6fc12ad52b1770714953` |
| `pyrregular-physionet2012` | `Physionet2012.h5` | 10,894,302 | `3a236e007c0a296eac610c671d308b72a97aeed6865c7f7407537fb8f824107f` |
| `pyrregular-physionet2019` | `Physionet2019.h5` | 31,620,387 | `57210ee1da9c2ee6cca3f4bf183f5209e35035d55c3c62cc678f53c180e6f923` |
| `pyrregular-seabirds` | `Seabirds.h5` | 8,542,499 | `6b9d9f82cf205c73efcb88b4164a2437c9503032c463f8b7a6fa0f12dc831348` |
| `pyrregular-pamap2` | `Pamap2.h5` | 889,201,664 | `ea28c7682d51e9f43ce673d5b15375cff111e295f7370fcf315dd51be15cad27` |
| `pyrregular-insectwingbeat` | `InsectWingbeat.h5` | 387,017,571 | `6f677dbfa53092271714eb8adb45ab2d05140e5e47ad88524b9aca58d9f2e768` |

`Mimic3.h5` is pyrregular's open 57-patient MIMIC-III demo, not the credentialed dataset.
Per-dataset statistics are in `documents/irregular_datasets_survey.md`.

### OpenSky (`data/opensky/`) — collection `opensky`

This is ADS-B state vectors from the [OpenSky Network](https://opensky-network.org/), used as
an open-population stress test (one tape per aircraft, global `icao24` entity ids). The
loader is `data/opensky_datasets.py`, and the plan is in `documents/opensky_fused_tape_plan.md`.

- **One-day Trino tables snapshot, 2026-03-01**: `state_vectors` (24 hourly parquet
  partitions) and `flights` (2 files), 13.3 GB. It was fetched with
  `scripts/download_opensky_trino_snapshot.sh` from
  `https://s3.opensky-network.org/data-samples/trino-tables/` using the key/size manifest
  `data/opensky/trino_snapshot.keys`.
  **As of 2026-11-25 the snapshot is no longer public:** that prefix holds only a
  `placeholder.txt` dated 2026-11-04, and the object keys return 404. The local copy and
  `data/checksums/opensky.sha256` are now the record of what was used. Access to the live
  data goes through OpenSky's Trino interface, which needs an approved research account
  (see [opensky-network.org](https://opensky-network.org/)). OpenSky data is for non-commercial research use
  under its terms.
- **Aircraft metadata database** (icao24 → registration, manufacturer, typecode), used for
  type priors: `scripts/download_opensky_aircraft_db.sh` from
  `https://opensky-network.org/datasets/metadata/aircraftDatabase.csv`. This URL is live but
  serves a moving file, so the hash below pins the copy used.

| File | SHA-256 |
|---|---|
| `aircraftDatabase.csv` (94,509,542 bytes) | `d40c8a7b3825dfca896fe234de56040a70e7a215ee34a895468116e51e1afcae` |
| `trino_snapshot.keys` | `fe0a9b77cbe75905c13da6edcf9c1f8eef78957bd71b83c5ad389b05c225c1de` |

<details>
<summary>The 26 snapshot parquet files (paths under <code>data/opensky/trino_snapshot_2026-03-01/</code>)</summary>

| File | SHA-256 |
|---|---|
| `flights/day=1772323200/part-00001-7a04c6cd-e438-4096-b0d5-2fc085af90b1.c000.snappy.parquet` | `05a1566cbeeda2ff4d66ac7d87d2369bf9710c5430ec0767b986ad09cbbad4bc` |
| `flights/day=1772323200/part-00001-a1c89589-f077-4615-bd84-d86ac2fc1497.c000.snappy.parquet` | `a7f37d6e43055bbf92ae6c6eedae5b7f987383241b593a09bb49354b6eaf8b55` |
| `state_vectors/hour=1772323200/part-00000-3d627b04-2f58-406c-970c-ea26251b2b70.c000.snappy.parquet` | `c0f3ce9064988c99e62abb6ae8d8ab205c0be2457c29dcb86c1b52f44a617928` |
| `state_vectors/hour=1772326800/part-00000-f71083bd-2b5d-4fdf-b35c-a1623c0d6d3d.c000.snappy.parquet` | `3408176777771c7e8999846eae526e965799f2c4580dacb7d4186220191369d8` |
| `state_vectors/hour=1772330400/part-00000-7d18074b-dd4c-4d1f-aa45-125e6d3117f9.c000.snappy.parquet` | `20926e6ae8a7009f4d0703de67d7983e50b3a29a6dd30271bf9c37f000e013e1` |
| `state_vectors/hour=1772334000/part-00000-aa1cd872-0cd9-44e4-9577-e802e2611940.c000.snappy.parquet` | `5999982d41dc5bd8eccaa2990af4e8a139457cc952ddf64ecc882b185947f117` |
| `state_vectors/hour=1772337600/part-00000-db749a09-f28e-47ad-aa07-1f6556a69d8f.c000.snappy.parquet` | `bd4da8b5c3aa2cdcd86d507351e107fcf2fbad23beff3872f6e7a8f09c7f12e1` |
| `state_vectors/hour=1772341200/part-00000-468a81ff-9d66-480a-a616-01f36ddd946a.c000.snappy.parquet` | `25869c1248439d08e42ba496a8b86af48e9429e7d541fecf9d5f47dcfebe5170` |
| `state_vectors/hour=1772344800/part-00000-36e1108c-5af1-4388-968b-f3efdaeb42d0.c000.snappy.parquet` | `cd189b57a092d0ff0d071545b0e4abad092cb7462d0a66623f97ca8f630160cd` |
| `state_vectors/hour=1772348400/part-00000-43971b16-6645-46db-9edb-5b4ea96dbe40.c000.snappy.parquet` | `276690130bf0efbcc00ab4c28d1b34fff91247d5c358ce7694473b6333ac5fba` |
| `state_vectors/hour=1772352000/part-00000-470be778-92a8-433d-ae70-499d043d9eba.c000.snappy.parquet` | `877c734dadbd6ea3d31861399b078d2cc950407938a0757ac6be26083401870d` |
| `state_vectors/hour=1772355600/part-00000-917b92a1-f0cd-44aa-aa9c-5731d9e83e25.c000.snappy.parquet` | `714a03743178a53061692ffb26c456b0fc71ecce5ad0508d5d4ec8e9d5236046` |
| `state_vectors/hour=1772359200/part-00000-6b73b13d-6de0-4353-8763-7afe2a23b8c3.c000.snappy.parquet` | `6ce2d5de8d43c3f3bad54713f57670b46968eeb2c50a6dffdbe949f3852e4e9d` |
| `state_vectors/hour=1772362800/part-00000-c0292d21-4fd4-442e-a9fd-952dca50fdf9.c000.snappy.parquet` | `f186ac7a5db8dd1f4cbc94cbb068462f2c22ee8c0f8de16219b3555065171ae8` |
| `state_vectors/hour=1772366400/part-00000-53827dd6-11f7-4e9a-9bb6-71883da82c7d.c000.snappy.parquet` | `cf8ae7bcd8633b46401718266e1067ec1edc84a5577183d1dd0712ee9c050aac` |
| `state_vectors/hour=1772370000/part-00000-e8c37e9b-969d-4e07-a04d-c4808da9c66a.c000.snappy.parquet` | `1d6cdae5395979419fe87e3ca8352a6d3f4f8d8a1a965b7a1c6051acc1b82fdc` |
| `state_vectors/hour=1772373600/part-00000-69d63471-e1a0-469d-8b11-dac6ee7c33eb.c000.snappy.parquet` | `f1028811e01b42feed18c5d0b84cc3f718aa811d22e5929dfd3b9dcf74da0c2d` |
| `state_vectors/hour=1772377200/part-00000-229792d1-b779-4746-85cb-41070aa37023.c000.snappy.parquet` | `0c1279bddc86e122a2a2e3493b711c38aec32c42228d79d9a8083b04e7cda35b` |
| `state_vectors/hour=1772380800/part-00000-b46105b8-2aa6-469b-a839-0986535ad36c.c000.snappy.parquet` | `251450f8d113ea1851cf81d5e1c92b6a70f87af9dd1f0d6ae79f8a99c137cd1b` |
| `state_vectors/hour=1772384400/part-00000-d8f13854-3596-4133-940a-895e50f1e643.c000.snappy.parquet` | `8569e5e88bcd16a02cb7733c08c0dbd1243da5d644d64131d270f3f6ddbdf8ec` |
| `state_vectors/hour=1772388000/part-00000-3fb30eb6-6374-4a94-8b5f-76b15d96885f.c000.snappy.parquet` | `b395974bd65c2369e5658080eb54936b38be55d8b1d981607511c3ad78ce2d2d` |
| `state_vectors/hour=1772391600/part-00000-88860fb1-d01d-457b-b71d-3b3aa1fe5d00.c000.snappy.parquet` | `599fcd3e46bec72a00aeaa56cff8ec1c1a48bc3f810e7e6904261c74b3473b5e` |
| `state_vectors/hour=1772395200/part-00000-6a08b11b-ae86-4ef6-ba92-01c8a9d06375.c000.snappy.parquet` | `9f2176e6f4bd2b1c1ad1807b484815d18df57f53f15525982cd7728da183f679` |
| `state_vectors/hour=1772398800/part-00000-29422461-ba4e-40f7-bba0-5653f1a09f7f.c000.snappy.parquet` | `54a0e391215d45eaea49512a81cf42953d709e4a00f638b4dc87dbf262b65854` |
| `state_vectors/hour=1772402400/part-00000-4295aede-bf5a-44fe-ba7f-7f5522b16a1e.c000.snappy.parquet` | `dd33e6829038f3a6d01ce106c2722d86c51d9197ecb38bc096bdb64cd1a210ba` |
| `state_vectors/hour=1772406000/part-00000-06db1353-b5db-4296-8677-619388e22c12.c000.snappy.parquet` | `2348e14ec3d75825ee7b9d95a071e89dcebe502bf5088a149f5ef07b282072e1` |

</details>

### Entity-text priors (`data/entity_text/`, committed)

These are hand-written channel descriptions (`<dataset>.json`) and their offline embeddings
(`<dataset>.<model>.npy`, built by `scripts/embed_entity_text.py`), used by
`--entity-priors` / the G2 entity-table init. They are tracked in git, so git is their hash;
they are not external data.

### Simulated datasets (not hashed)

The synthetic benchmarks are generated on the fly. A tape is a deterministic function of the
generator code, its parameters and the seed, so the code is the reference and nothing is
hashed. Results JSONs record the git commit and the full argument set, which reproduces any
synthetic row.

| `--dataset` | Process | Code |
|---|---|---|
| `ou` (default) | Coupled multivariate Ornstein–Uhlenbeck, dX_i = θ_i(μ − X_i)dt + σ_i dW_i + Σ_j C_ij(X_j − X_i)dt; Poisson firing at rates ∝ θ_i, so some entities are 10–100× rarer | [`data/generators/ou_process.py`](data/generators/ou_process.py) (`generate_ou_tape`, `generate_coupling_matrix`) |
| `hawkes` | Multivariate Hawkes, λ_i(t) = μ_i + Σ_j Σ_{t_k<t} α_ij e^{−β(t−t_k)}, excitation and inhibition, sampled by Ogata thinning | [`data/generators/hawkes_process.py`](data/generators/hawkes_process.py) (`generate_hawkes_tape`, `cached_hawkes_tape`, `branching_radius`) |
| `lv` | Multi-species Lotka–Volterra, dX_i/dt = X_i(r_i + Σ_j A_ij X_j) + noise; firing ∝ population (`--lv-sampling poisson\|state`) | [`data/generators/lotka_volterra.py`](data/generators/lotka_volterra.py) (`generate_lv_tape`) |
| `switching` | Regime-switching LDS, X_{t+1} = A_k X_t + B_k u_t + noise, with global regime switches | [`data/generators/switching_lds.py`](data/generators/switching_lds.py) (`generate_switching_tape`) |
| interference harness | Fixed AR(1) target process plus Poisson white-noise distractor entities (wide-tape S1/S2 stress) | [`data/generators/interference.py`](data/generators/interference.py) (`make_interference_tape`) |

Supporting code: `data/generators/synthetic_tapes.py` handles dispatch, per-process
memoisation and spawn-pool prefetch, with bit-identity pinned by
`tests/test_synthetic_tapes.py`. `data/generators/tape_arrays.py` turns a raw tape into
`[value, log Δt_entity, log Δt_global]` features, targets and masks. With
`TAPEWORLD_SYNTH_CACHE` set, Hawkes tapes are cached as `.npz` keyed by the SHA-1 of their
parameters; `scripts/pregen_hawkes.py` pre-fills the cache. The `ltsf-thin-*` irregular
sets are not simulated: they are Bernoulli-thinned views of the hashed LTSF panels
(`load_ltsf_thinned`, seed `--thin-seed`).

---

## Results

Every run writes a JSON to `outputs/results/` with the config, git hash, host,
`TAPEWORLD_*` environment, per-epoch history, final metrics and per-seed runs.

```bash
python scripts/summarize_results.py                      # markdown tables from outputs/results/**.json
python scripts/summarize_results.py --docs documents     # rewrite documents/{ltsf,cmapss,synthetic,irregular}_results.md
python scripts/sweep.py configs/f7_data_scale.json       # run a grid, resumable, skips finished points
python scripts/sweep.py configs/ltsf_v3_raw.json --dry-run
python scripts/val_curve_gate.py outputs/results/...     # flag runs whose argmin sat on a flat curve
python scripts/paper_harness.py check                    # paper cells vs results JSONs
python scripts/paper_harness.py numbers                  # recompute every bound paper cell
```

Where things stand:

- **The numbers** are the result JSONs under `outputs/results/`. `scripts/paper_harness.py numbers`
  recomputes every bound cell of the paper from them.
- **The paper** is `paper/the_tape_v3.tex` (canonical since 2026-11-19). `the_tape_nov.tex`
  and `the_tape.tex` are archived drafts.
- **Claims** are in the claim ledgers (`SPEC_V2.md` §1.5, `SPEC_PAPER_NOV.md` §3), the dated
  OUTCOME blocks under each item, and the newest `documents/readout_*.md`.
- `documents/{ltsf,cmapss,synthetic,irregular}_results.md` are machine-generated
  per-benchmark tables. Sweeps are excluded unless passed explicitly; read those from the
  sweep's own `summary.md`.
- `WORK_QUEUE.md` is the frozen v1 work log (cited as `WQ §<id>`).
- `ALL_RESULTS_SUMMARY.md` was deleted on 2026-11-19. Its last version is
  `git show df3dead:ALL_RESULTS_SUMMARY.md`, which is what older documents citing it mean.

---

## Experiments workflow

```bash
python3 scripts/supervise.py --brief     # what is owed: relaunch / read / launch / outcome; failed jobs; hygiene
```

`registry.toml` holds one entry per registered item that is in flight or still owes work.
Supervision reads it and never writes. New experiments go through the `experiment` CLI
(`experiment new TRACK SLUG`, then `freeze`, `submit -H HOST`, `collect`, `judge`). Plans,
gates and predictions are written and frozen before the run. Jobs run through TinyShed
(`tshed submit --git ...`), from a frozen commit that never touches `main`. Hosts and paths
are in `experiments/config.toml`. Details are in `AGENTS.md`.

---

## CUDA path (Triton kernels)

`model/torch/` is a line-for-line port of the MLX model. `tests/test_backend_parity.py`
copies MLX weights into the torch model and checks that the outputs agree to 2e-4.
The Mamba-3 chunked scan is split into the classic SSD stages:

1. `chunk_state`: per-chunk state increment `Σ_s exp(cs_T − cs_s)·scale_s·k_s⊗v_s`
2. state passing: the recurrence over chunks (two-level blocked form by default, plain torch, autograd)
3. `chunk_scan`: intra-chunk decayed attention plus the inter-chunk state read

Stages 1 and 3 are Triton kernels (`model/torch/ops/ssd_triton.py`) with hand-written
backward kernels (flash-attention-style row/column split, no atomics), wrapped in
`torch.autograd.Function`. `model/torch/ops/ssd_reference.py` is the pure-torch reference
used on CPU and by `tests/test_ssd_triton.py`. Kernel changes must pass that test on CUDA
under both IEEE and `TAPEWORLD_TF32=1`.

Measured on an RTX 4070 Ti Super (torch 2.11 + cu128, Triton 3.6): Mamba-3 block
forward+backward at B=16, L=1024, d_model=128.

| headdim | d_state | reference | Triton (IEEE) | Triton (TF32) |
|--------:|--------:|----------:|--------------:|--------------:|
| 32 | 32 | 15.6 ms | 10.0 ms (1.56×) | 9.8 ms |
| 32 | 64 | 24.0 ms | 18.2 ms (1.32×) | 16.9 ms |
| 64 | 32 | 10.6 ms | 8.7 ms (1.22×) | 7.8 ms |
| 64 | 64 | 15.2 ms | 15.8 ms (0.96×) | 11.9 ms |

Full 4-layer train step (point head): 65 ms → 44 ms, peak memory 1.9 GB → 1.2 GB.
Reproduce with `.venv/bin/python scripts/bench_ssd.py`. Constraints: chunk size, `d_state`
and `headdim` must be powers of two in [16, 128].

---

## Architecture

```
events → [E_entity(id) + Proj(value, Δt_entity, Δt_global)] → RMSNorm
       → [Mamba3 | LinearAttention | S5 | KDA] × n_layers   (optional cross-entity mix per block)
       → RMSNorm → head
            gaussian: (μ, log σ²) per horizon      (synthetic / irregular)
            point:    next step [+ H direct steps] (LTSF, MSE)
            + rul:    pooled final states → scalar (C-MAPSS)
            intensity head                          (TPP)
```

One model class (`TapeModel`), built from `TapeConfig`, serves all experiments. The head and
token projection are config options. Experiments talk to a backend module
(`train/mlx_backend.py`, `train/torch_backend.py`) that exposes `build_model`, a `Trainer`
(compiled step, AdamW, grad clipping), `predict`, state snapshot/restore and the loss
factories, so an experiment is backend-agnostic numpy code. Model forward returns a dict:
`mu`/`log_var` (gaussian head), `pred` (point head), `rul`, `hidden`.

- **Entity embeddings**: a learned vector per entity, added to the projected features (dense
  tables or compositional atom mixtures, `--embed-mode`).
- **Sparse loss**: evaluated only for the firing entity at each position, reweighted by entity
  frequency. `--dense-loss` drops the frequency weight; the dense head is entity-conditioned,
  so with matched targets it is the *same* loss (`documents/dense_sparse_confound.md`).
- **Token projection**: `--proj no_layernorm` (default since 2026-11-19) is injective on
  `[x, log Δt_ent, log Δt_glob]`. The old `mlp` projection's LayerNorm over three features
  left one degree of freedom (defect 1).
- **Record resets**: packed records do not share scan state (`--reset-records`, default on).
- **Cross-entity learning**: because all entities share one sequence, entity B's loss
  back-propagates through entity A's tokens. `--cross-entity` adds explicit mixing.

Extra per-position inputs (record-reset flag, query-time columns, anchor window, physical-clock
column) ride in `features` after the projected columns, in that order.

---

## Project layout

```
TapeWorld/
├── run.py                         CLI entry point (dispatches to experiments/)
├── AGENTS.md                      Orientation: conventions, adopted defaults, known defects, hosts
├── SPEC_V2.md, SPEC_PAPER_NOV.md  Research agenda and claim ledgers, pre-registered items, OUTCOME blocks
├── registry.toml                  Registered items in flight (read by scripts/supervise.py)
├── WORK_QUEUE.md                  Frozen v1 work log
├── model/
│   ├── config.py                  TapeConfig: one dataclass for both backends
│   ├── stformer_common.py         STCS clustering for the STformer reproduction
│   ├── mlx/                       MLX: tape_world, mamba3, ssd, linear_attention, s5, per_entity_attention,
│   │                              sparse_group, routed_memory, retrieval, stformer, mamba2
│   ├── torch/                     PyTorch: same modules + kda, mamba3_official, mamba3_q8, quant
│   │   └── ops/                   ssd_reference.py, ssd_triton.py, s5_scan_triton.py, dispatch
│   └── _archive/                  Superseded Mamba-3 variants (kept for the scan-equivalence test)
├── train/
│   ├── common.py                  CLI arg groups, schedules, epoch loop, adopted defaults, results JSON
│   ├── mlx_backend.py             Trainer + losses (mx.compile)
│   └── torch_backend.py           Trainer + losses (torch.compile, fused AdamW, AMP)
├── experiments/
│   ├── synthetic.py, ltsf.py, cmapss.py, irregular.py, tpp.py, stformer.py
│   ├── entity_holdout.py, entity_prior.py, dlinear.py, robust_metrics.py
│   ├── plot_exchange.py, plot_exchange_multistep.py, utils.py
│   ├── config.toml                Hosts and paths for the experiment CLI
│   └── <track>-<date>-<slug>/     Frozen experiments (plan.md, exp.toml, results)
├── data/
│   ├── ltsf_datasets.py           LTSF loader (corrected standard_split)
│   ├── irregular_datasets.py      t-PatchGNN, Physiome-ODE, EasyTPP, Time-IMM, pyrregular, thinned LTSF
│   ├── traffic_datasets.py        METR-LA / PEMS-BAY (STformer artifacts)
│   ├── opensky_datasets.py        OpenSky snapshot → tapes
│   ├── generators/                Simulators (ou, hawkes, lotka_volterra, switching_lds, interference),
│   │                              synthetic_tapes.py, tape_arrays.py, cmapss_dataset.py
│   ├── checksums/                 SHA-256 manifests, one per dataset collection
│   ├── entity_text/               Entity-text priors (committed)
│   └── ltsf_raw/, cmapss/, irregular/, traffic/, opensky/   Data (mostly git-ignored; see Datasets)
├── scripts/                       Downloads, hash_datasets.py, sweep.py, summarize_results.py,
│                                  supervise.py, paper_harness.py, benchmarks, read-outs, audit/
├── configs/                       Sweep grids
├── tests/                         pytest suite
├── documents/                     Write-ups, read-outs, audits
├── paper/                         the_tape_v3.tex (canonical)
└── outputs/                       Results JSONs and plots
```

---

## Tests

```bash
uv sync --extra dev
bash scripts/run_tests.sh                  # compileall + pytest + mypy
.venv/bin/python -m pytest tests -q        # pytest only
```

Tests skip cleanly when a backend is absent, so the same suite runs on a Mac (MLX + CPU torch)
and on a CUDA box (torch + Triton). `pyproject.toml` sets `-x`, so a run stops at the first
failure; pass `--maxfail=100` to see them all. Selected files:

- `test_backend_parity.py`: MLX and torch agree with copied weights; chunked SSD equals the sequential recurrence.
- `test_ssd_triton.py`, `test_s5_triton.py`, `test_triton_modes.py`: Triton forward/backward vs the torch reference (CUDA only).
- `test_state_passing.py`, `test_mlx_ssd.py`: state-passing forms and the vectorised MLX SSD (the Metal repeatability pin, `test_ssd_state_passing_repeatable`, is in `test_backend_parity.py`).
- `test_token_injective.py`, `test_proj_no_layernorm.py`, `test_reset_query_loss.py`: the defect-1 and defect-4 fixes.
- `test_cross_entity.py`, `test_trainers.py`, `test_loss.py`, `test_optim_harness.py`: mixing arms, compiled == eager, loss factories.
- `test_irregular_datasets.py`, `test_tpp.py`, `test_traffic_datasets.py`, `test_opensky.py`: loaders (fixtures where the data is absent).
- `test_synthetic_tapes.py`, `test_hawkes_cache.py`, `test_tape_arrays.py`: simulator determinism and feature layout.
- `test_paper_harness.py`, `test_supervise.py`, `test_sweep.py`, `test_results_json.py`: tooling.
