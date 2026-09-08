# Pretrained sampling examples

[Project overview](../README.md) · [Experiment guide](../experiments/README.md)

These five bundles are included directly in Git (about 4.8 MiB total). No external
download, account or training is needed. Run commands from the repository root
after `pixi install --locked -e dev`.

| Bundle | Target | Weights | Training budget |
| --- | --- | --- | --- |
| [ising_baseline_4x4](ising_baseline_4x4/manifest.json) | Unconstrained 4×4, exact critical coupling | Final raw, seed 42 | 10,000 steps |
| [ising_soft_4x4](ising_soft_4x4/manifest.json) | Soft 4×4, exact critical coupling, composition 0.5, penalty 50 | Final raw, seed 42 | 10,000 steps |
| [ising_hard_4x4](ising_hard_4x4/manifest.json) | Hard 4×4, coupling 0.1, exactly 8 sites of each species | Final raw, seed 42 | 2,000 steps |
| [ising_hard_16x16](ising_hard_16x16/manifest.json) | Hard 16×16, exact critical coupling, exactly 128 sites of each species; patch radius 2 | Final EMA, seed 42 | 100,000 steps, coupling curriculum |
| [ising_hard_24x24](ising_hard_24x24/manifest.json) | Hard 24×24, exact critical coupling, exactly 288 sites of each species; patch radius 3 | Final EMA, seed 42 | 100,000 steps, coupling curriculum, bf16 training |

The exact critical coupling used here is `0.22034339675488573`. The small hard
example is subcritical. This set illustrates inference across the three regimes;
it is not a matched comparison across methods or the full dissertation archive.
Each `config.json` and checkpoint is copied byte-for-byte from a retained local
run. The manifests record the source run, selected raw/EMA filename and hashes.
These are inference weights without optimizer or RNG state for training resume.
They are distributed under the repository's [MIT License](../LICENSE).

## Draw samples

```bash
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_baseline_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-baseline-4x4
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_soft_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-soft-4x4
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_4x4 \
  --n-samples 64 --batch-size 8 --seed 0 --out results/demo-hard-4x4
pixi run -e dev python -m scripts.sample_checkpoint checkpoints/ising_hard_24x24 \
  --n-samples 8 --batch-size 1 --seed 0 --out results/demo-hard-24x24
```

The helper verifies both hashes before loading, uses the production model/target
constructors and runs eager fp32 inference. Hard configurations also pass the
production registry drift check. CPU and one PyTorch thread are the defaults;
on a CUDA machine, use `pixi run -e cuda` and append `--device cuda`. The 24×24
head is more expensive, so start with the small batch shown above.

Every output directory must be new. To repeat a draw, choose another `--out`.
The files are:

- `samples.pt`: tensor of shape `(n_samples, n_sites)`, spins in `{-1, +1}`.
- `log_weights.pt`: one unnormalised log importance weight per proposal.
- `metadata.json`: checkpoint/config hashes, seed, batch size, device, precision,
  PyTorch version, integration intervals and importance-sampling ESS.

The grid follows each trainer: the bundled flip configs use 50 grid points
(49 intervals); hard 4×4 uses 100 intervals and hard 24×24 uses 128 matching
intervals. Seeds are repeatable within a fixed software/device/batch setup;
changing the batch size changes RNG consumption. These small demonstration draws
do not reproduce the report's evaluation budgets or estimates.

## Use the importance weights

The saved states are raw proposals from the learned process. Hard swaps guarantee
composition at every step; importance weighting corrects target expectations
within the sampler's finite-grid approximation. For example, estimate the
expected fraction of neighbouring sites with equal spins:

```python
from pathlib import Path
import torch

run = Path("results/demo-hard-4x4")
x = torch.load(run / "samples.pt", weights_only=True).reshape(-1, 4, 4)
log_w = torch.load(run / "log_weights.pt", weights_only=True)
w = log_w.double().softmax(dim=0)
agreement = 0.5 * (
    (x == x.roll(1, dims=1)).double().mean(dim=(1, 2))
    + (x == x.roll(1, dims=2)).double().mean(dim=(1, 2))
)
print("Weighted neighbour agreement:", (w * agreement).sum().item())
print("Importance-sampling ESS:", (1 / w.square().sum()).item())
```

## Recreate the README animation

Rendering the committed recording does not load a model:

```bash
pixi run -e dev python -m scripts.animate_recorded_swap --out /tmp/recorded_swap.gif
```

To record a fresh trajectory from the bundled 24×24 EMA model and animate it:

```bash
pixi run -e dev python -m scripts.record_swap_trajectory checkpoints/ising_hard_24x24 \
  --seed 20260907 --out results/demo-trajectory-24x24.npz
pixi run -e dev python -m scripts.animate_recorded_swap \
  --recorded results/demo-trajectory-24x24.npz --out /tmp/fresh_swap.gif
```

The recorder uses one raw draw, stores all 129 grid states and evaluates the
marked site's rates at each time. It does not select a trajectory or resample
terminal states. See the [visual provenance](../assets/readme/README.md).
