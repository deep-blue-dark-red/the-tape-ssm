# the-tape-ssm

The Tape architecture: per-entity learned embeddings ("tapes") trained with a
sparse loss over a shared sequential state-space backbone (Mamba3 / linear
attention). One `TapeConfig` builds the same model on two backends — MLX
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
