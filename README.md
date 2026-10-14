# the-tape-ssm

The Tape SSM is an architecture for multi-entity time series built on a single,
deliberately simple idea: treat every observation — whichever entity it comes
from, whenever it arrives — as one token in a single shared sequence. The
backbone is a state-space model (Mamba-3 or linear attention) with no
per-entity parameters; entity identity enters only through a learned per-entity
embedding, and irregular timing enters as the inter-observation gap Δt fed into
the SSM step size. The training objective is sparse: loss is charged only at
actually-observed (entity, time) pairs, so asynchronous, heterogeneous, and
irregularly sampled data need no imputation or per-entity modelling.

One `TapeConfig` builds the same model on two backends — MLX
(Apple Silicon) and PyTorch (CUDA with Triton SSD kernels, or CPU).

## Contents

- `findings_report_2026.pdf` — findings report

## Related

Main repository: [TapeWorld](../)

## Quick start

```bash
python run.py synthetic --dataset hawkes --epochs 30
python run.py ltsf --dataset ETTh1 --all-horizons --n-seeds 3
uv run python -m pytest tests -q
```
