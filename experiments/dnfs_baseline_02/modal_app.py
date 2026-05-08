"""Modal wrapper for DNFS Ising baseline training. Delegates the actual
training to `run.train` so the remote and local code paths share a single
implementation.

The container image is built by installing pixi inside the container and
running `pixi install --environment dev --locked` against the project's
`pixi.lock`, so the remote runtime stack (pytorch, numpy, ...) matches
local dev byte-for-byte. One source of truth is `pixi.lock` -- there is
no second dep list living in this file.

The `CONDA_OVERRIDE_CUDA=12.4` env var is required because the build
container has no GPU and pixi's `__cuda` virtual package check would
otherwise fail; runtime containers get a real GPU from Modal.

Usage (after `modal token new` and `modal secret create wandb-secret ...`):
    # Single config:
    pixi run -e dev modal run -m \\
        experiments.dnfs_baseline_02.modal_app::main \\
        --cfg-name stage_1_d4 --seed 42

    # All 8 post-redo configs in parallel. `--detach` is REQUIRED:
    # without it, the ephemeral app stops when the entrypoint returns and
    # all spawned FunctionCalls are cancelled before any container runs.
    pixi run -e dev modal run --detach -m \\
        experiments.dnfs_baseline_02.modal_app::batch --scale all
"""
import modal

PROJECT_DIR = "/repo"
PIXI_ENV_BIN = f"{PROJECT_DIR}/.pixi/envs/cuda/bin"

# Build the container image from `pixi.lock`. The repo is added at /repo;
# pixi installs the dev env in-place; PATH points at the pixi env's bin.
#
# Local-only directories (`.pixi/` env tree, past `results/`, `wandb/` run
# logs, `.git`, caches) are excluded from the upload. They are large,
# irrelevant on the remote, and `.pixi/` in particular causes "modified
# during build" failures when any local pixi-run command touches conda
# metadata while the Modal uploader is still streaming bytes.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://pixi.sh/install.sh | bash",
        "ln -s /root/.pixi/bin/pixi /usr/local/bin/pixi",
    )
    .workdir(PROJECT_DIR)
    .add_local_dir(
        ".",
        PROJECT_DIR,
        copy=True,
        ignore=[
            ".pixi/**",
            "results/**",
            "wandb/**",
            ".git/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            ".venv/**",
            "*.pdf",
        ],
    )
    .run_commands(
        f"cd {PROJECT_DIR} && CONDA_OVERRIDE_CUDA=12.4 "
        "pixi install --environment cuda --locked"
    )
    .env({"PATH": f"{PIXI_ENV_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)

# Persistent volume for run artefacts (training_log.csv, checkpoints, eval
# tensors). One volume across runs lets us re-pull anything later via
# `modal volume get`.
volume = modal.Volume.from_name("dnfs-results", create_if_missing=True)

# wandb API key lives in a Modal secret, NOT in the image. `modal secret
# create wandb-secret WANDB_API_KEY=...` makes it available as an env var.
wandb_secret = modal.Secret.from_name("wandb-secret")

app = modal.App("dnfs-baseline", image=image)


@app.function(
    # L4 is roughly the same wall-clock as A100 for Stage 1's MLP-bound
    # workload, at about half the cost. Reconsider for Stage 3 (LeT) where
    # attention layers benefit more from A100/H100-class compute.
    gpu="L4",
    volumes={"/results": volume},
    secrets=[wandb_secret],
    # 24h is generous for D=10; tighten if cost matters.
    timeout=24 * 60 * 60,
    # Note: `nonpreemptible=True` is not supported for GPU workloads on Modal
    # (rejected at deploy time). Long runs may preempt-and-auto-retry; clean up
    # truncated wandb runs (`_step < n_steps - 1`) post-completion.
)
def train_remote(cfg_name: str, seed: int = 42):
    """Run a single training config on Modal and persist artefacts to the
    volume. Importable but typically launched via `modal run`."""
    import sys

    # /repo is the mount point of `add_local_dir`. Inserting it onto sys.path
    # lets `experiments.dnfs_baseline_02.run` import resolve correctly.
    sys.path.insert(0, "/repo")
    from experiments.dnfs_baseline_02.run import train

    train(cfg_name, seed=seed, output_dir="/results")
    # commit() makes the artefacts visible to subsequent `modal volume get`
    # calls. Without this, the volume is reverted on container shutdown.
    volume.commit()


@app.local_entrypoint()
def main(cfg_name: str, seed: int = 42):
    """Local CLI entry: spawns `train_remote` as a remote Modal call."""
    train_remote.remote(cfg_name=cfg_name, seed=seed)


@app.local_entrypoint()
def batch(scale: str = "all", seed: int = 42):
    """Fire off post-redo configs in parallel. scale: 'all' | 'd4' | 'd10'."""
    d4 = ["stage_0_d4", "stage_0_d4_cv", "stage_1_d4", "stage_2_d4"]
    d10 = ["stage_0_d10", "stage_0_d10_cv", "stage_1_d10", "stage_2_d10"]
    configs = {"all": d4 + d10, "d4": d4, "d10": d10}[scale]
    for cfg in configs:
        train_remote.spawn(cfg_name=cfg, seed=seed)
    print(f"spawned {len(configs)} jobs: {configs}")


@app.local_entrypoint()
def batch_configs(configs: str = "", seed: int = 42):
    """Spawn arbitrary comma-separated configs in parallel.

    Example: `--configs stage_1_d10,stage_2_d10` to relaunch only the leMLP D=10 runs.
    """
    cfg_list = [c.strip() for c in configs.split(",") if c.strip()]
    for cfg in cfg_list:
        train_remote.spawn(cfg_name=cfg, seed=seed)
    print(f"spawned {len(cfg_list)} jobs: {cfg_list}")


@app.function(gpu="L4", timeout=10 * 60)
def profile_lemlp_remote(D: int = 100, B: int = 256, h: int = 256, K: int = 3,
                          n_iters: int = 50, warmup: int = 5):
    """Benchmark four LeMLP forward variants on a Modal GPU.

    Variants:
      current — production code in models/lemlp.py
      A       — Step 5 readout without `diff` materialisation
      B       — Step 3 hollow-MLP via broadcast matmul instead of einsum
      AB      — Both A and B
    """
    import sys
    sys.path.insert(0, "/repo")
    import time
    import torch
    from discrete_flow_sampler.models.lemlp import LeMLPRateMatrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    print(f"Device: {device.type} ({name})")

    torch.manual_seed(0)
    model = LeMLPRateMatrix(d=D, vocab_size=2, hidden_dim=h, n_summands=K).to(device)
    model.eval()
    x = torch.randint(0, 2, (B, D), device=device).float() * 2 - 1
    t = torch.rand(B, device=device)

    def fwd_A():
        x_idx = ((x + 1) / 2).long()
        x_emb = model.token_embedder(x_idx)
        W = model.W_raw * model.hollow_mask
        pre = torch.einsum("kdj,bjh->kbdh", W, x_emb) + model.b[:, None, None, :]
        H = model.activation(pre).sum(dim=0) + model.time_embedder(t)[:, None, :]
        omega_all = model.omega.weight
        omega_xi = model.omega(x_idx)
        G_all = torch.einsum("bdh,sh->bds", H, omega_all)
        G_self = torch.einsum("bdh,bdh->bd", H, omega_xi).unsqueeze(-1)
        return (G_all - G_self).scatter(-1, x_idx.unsqueeze(-1), 0.0)

    def fwd_B():
        x_idx = ((x + 1) / 2).long()
        x_emb = model.token_embedder(x_idx)
        W = model.W_raw * model.hollow_mask
        pre = torch.matmul(W.unsqueeze(1), x_emb.unsqueeze(0)) + model.b[:, None, None, :]
        H = model.activation(pre).sum(dim=0) + model.time_embedder(t)[:, None, :]
        omega_all = model.omega.weight
        omega_xi = model.omega(x_idx)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        G = torch.einsum("bdh,bdsh->bds", H, diff)
        return G.scatter(-1, x_idx.unsqueeze(-1), 0.0)

    def fwd_AB():
        x_idx = ((x + 1) / 2).long()
        x_emb = model.token_embedder(x_idx)
        W = model.W_raw * model.hollow_mask
        pre = torch.matmul(W.unsqueeze(1), x_emb.unsqueeze(0)) + model.b[:, None, None, :]
        H = model.activation(pre).sum(dim=0) + model.time_embedder(t)[:, None, :]
        omega_all = model.omega.weight
        omega_xi = model.omega(x_idx)
        G_all = torch.einsum("bdh,sh->bds", H, omega_all)
        G_self = torch.einsum("bdh,bdh->bd", H, omega_xi).unsqueeze(-1)
        return (G_all - G_self).scatter(-1, x_idx.unsqueeze(-1), 0.0)

    variants = {"current": lambda: model(x, t), "A": fwd_A, "B": fwd_B, "AB": fwd_AB}

    print("\nNumerical equivalence (vs current):")
    with torch.no_grad():
        G_curr = variants["current"]()
        for vname in ("A", "B", "AB"):
            G = variants[vname]()
            ok = torch.allclose(G_curr, G, rtol=1e-4, atol=1e-5)
            print(f"  {vname:<3}  allclose={ok}  max_abs_diff={(G_curr - G).abs().max().item():.2e}")

    def bench(fn):
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1000

    print(f"\nForward timings (mean over {n_iters} iters, {warmup} warmup):")
    times = {n: bench(fn) for n, fn in variants.items()}
    for vname, ms in times.items():
        suffix = "" if vname == "current" else f"  ({times['current'] / ms:.2f}x)"
        print(f"  {vname:<8} {ms:.2f} ms{suffix}")
    return times


@app.local_entrypoint()
def profile():
    """Run leMLP forward profiling on a remote Modal GPU."""
    times = profile_lemlp_remote.remote()
    print(f"\nDone. timings (ms): {times}")


@app.function(gpu="L4", timeout=15 * 60)
def profile_step_remote(cfg_name: str = "stage_1_d10", n_outer: int = 2, warmup_outer: int = 1):
    """Profile outer/inner training loop on Modal GPU, mirroring samplers/training.py.

    Outer step (1 per `inner_steps_per_outer` gradient updates):
      sample_x0, sample_traj (no_grad CTMC, T model fwds), compute_c_t
      (T fwds for control_variate, 0 for naive_mc), var_bookkeeping, buffer_flatten.
    Inner step (× inner_steps_per_outer):
      buffer_draw, kolmogorov_loss (1 fwd LE / 2 non-LE, grad-live), backward, optim_step.

    Reports amortised per-gradient-step = outer_total / inner_steps_per_outer + inner_total.
    Pre-rewrite leMLP D=10 baseline was ~363 ms/step; memory predicts ~15 ms/step here.
    """
    import sys
    sys.path.insert(0, "/repo")
    import time
    import torch

    from discrete_flow_sampler.targets.ising import IsingTarget
    from discrete_flow_sampler.samplers.ctmc import sample_ctmc
    from discrete_flow_sampler.samplers.kolmogorov import loss as kolmogorov_loss
    from discrete_flow_sampler.samplers.log_z_estimators import compute_c_t_grid
    from experiments.dnfs_baseline_02.configs import CONFIGS
    from experiments.dnfs_baseline_02.run import _build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    print(f"Device: {device.type} ({name})")

    cfg = CONFIGS[cfg_name]
    target = IsingTarget(D=cfg.ising.D, sigma=cfg.ising.sigma, bias=cfg.ising.bias, device=device)
    model = _build_model(cfg, target)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    n_grid = cfg.ctmc.n_euler_steps
    inner_batch = cfg.train.batch_size
    outer_batch = cfg.train.outer_batch_size or cfg.train.batch_size
    inner_steps_per_outer = cfg.train.inner_steps_per_outer
    is_le = model.is_locally_equivariant
    inner_warmup_first = 10  # discard first N inner of first measured outer

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    outer_sections = ["sample_x0", "sample_traj", "compute_c_t",
                      "var_bookkeeping", "buffer_flatten"]
    inner_sections = ["buffer_draw", "kolmogorov_loss", "backward", "optim_step"]
    outer_times = {s: [] for s in outer_sections}
    inner_times = {s: [] for s in inner_sections}

    torch.manual_seed(0)

    for outer in range(warmup_outer + n_outer):
        outer_kept = outer >= warmup_outer
        ot = {}

        sync(); t0 = time.perf_counter()
        x_initial = torch.randint(0, 2, (outer_batch, target.d), device=device).float() * 2 - 1
        sync(); ot["sample_x0"] = (time.perf_counter() - t0) * 1000

        t_grid = torch.linspace(0.0, 1.0, n_grid, device=device)

        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            x_traj = sample_ctmc(model, x_initial, t_grid, return_all_states=True)
        sync(); ot["sample_traj"] = (time.perf_counter() - t0) * 1000

        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            c_t_grid, integrand_per_t = compute_c_t_grid(t_grid, x_traj, target, model, mode=cfg.estimator)
        sync(); ot["compute_c_t"] = (time.perf_counter() - t0) * 1000

        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            t_grid_per_state = t_grid.repeat_interleave(outer_batch)
            x_traj_flat = x_traj.reshape(n_grid * outer_batch, target.d)
            naive_per_t = target.dt_log_p_tilde_t(x_traj_flat, t_grid_per_state).reshape(n_grid, outer_batch)
            _ = naive_per_t.var(dim=-1).mean().item()
            _ = integrand_per_t.var(dim=-1).mean().item()
        sync(); ot["var_bookkeeping"] = (time.perf_counter() - t0) * 1000

        sync(); t0 = time.perf_counter()
        x_buffer = x_traj.reshape(n_grid * outer_batch, target.d)
        t_idx_buffer = torch.arange(n_grid, device=device).repeat_interleave(outer_batch)
        buffer_size = n_grid * outer_batch
        sync(); ot["buffer_flatten"] = (time.perf_counter() - t0) * 1000

        if outer_kept:
            for s, v in ot.items():
                outer_times[s].append(v)

        for inner in range(inner_steps_per_outer):
            it = {}

            sync(); t0 = time.perf_counter()
            sample_idx = torch.randint(buffer_size, (inner_batch,), device=device)
            x_sample = x_buffer[sample_idx]
            t_idx_sample = t_idx_buffer[sample_idx]
            t_sample = t_grid[t_idx_sample]
            c_t_sample = c_t_grid[t_idx_sample]
            sync(); it["buffer_draw"] = (time.perf_counter() - t0) * 1000

            sync(); t0 = time.perf_counter()
            loss_value = kolmogorov_loss(x_sample, t_sample, c_t_sample, model, target)
            sync(); it["kolmogorov_loss"] = (time.perf_counter() - t0) * 1000

            sync(); t0 = time.perf_counter()
            optimiser.zero_grad()
            loss_value.backward()
            sync(); it["backward"] = (time.perf_counter() - t0) * 1000

            sync(); t0 = time.perf_counter()
            optimiser.step()
            sync(); it["optim_step"] = (time.perf_counter() - t0) * 1000

            inner_kept = outer_kept and (outer > warmup_outer or inner >= inner_warmup_first)
            if inner_kept:
                for s, v in it.items():
                    inner_times[s].append(v)

    outer_means = {s: sum(v) / len(v) for s, v in outer_times.items()}
    inner_means = {s: sum(v) / len(v) for s, v in inner_times.items()}
    outer_total = sum(outer_means.values())
    inner_total = sum(inner_means.values())
    amortised = outer_total / inner_steps_per_outer + inner_total

    n_outer_kept = len(outer_times["sample_x0"])
    n_inner_kept = len(inner_times["backward"])

    print(f"\n=== {cfg_name} (B_inner={inner_batch}, M_outer={outer_batch}, "
          f"d={target.d}, LE={is_le}, T={n_grid}, est={cfg.estimator}, "
          f"IPO={inner_steps_per_outer}) ===")

    print(f"\n[outer] {n_outer_kept} iters")
    print(f"{'section':<20} {'mean (ms)':<12} {'std':<8}")
    print("-" * 44)
    for s in outer_sections:
        m = outer_means[s]
        std = (sum((v - m) ** 2 for v in outer_times[s]) / len(outer_times[s])) ** 0.5
        print(f"{s:<20} {m:<12.3f} {std:<8.3f}")
    print(f"{'OUTER_TOTAL':<20} {outer_total:<12.3f}")

    print(f"\n[inner] {n_inner_kept} iters")
    print(f"{'section':<20} {'mean (ms)':<12} {'std':<8}")
    print("-" * 44)
    for s in inner_sections:
        m = inner_means[s]
        std = (sum((v - m) ** 2 for v in inner_times[s]) / len(inner_times[s])) ** 0.5
        print(f"{s:<20} {m:<12.3f} {std:<8.3f}")
    print(f"{'INNER_TOTAL':<20} {inner_total:<12.3f}")

    print(f"\n[amortised per-gradient-step]")
    print(f"  outer_total / IPO = {outer_total / inner_steps_per_outer:7.3f} ms")
    print(f"  inner_total       = {inner_total:7.3f} ms")
    print(f"  TOTAL             = {amortised:7.3f} ms")
    print(f"  Implied steps/sec = {1000 / amortised:.1f}")

    return {
        "cfg": cfg_name,
        "is_le": is_le,
        "outer_total_ms": outer_total,
        "inner_total_ms": inner_total,
        "amortised_per_step_ms": amortised,
        "outer_means": outer_means,
        "inner_means": inner_means,
    }


@app.local_entrypoint()
def profile_step():
    """Profile leMLP and vanilla MLP at D=10 (outer/inner amortised)."""
    le_call = profile_step_remote.spawn(cfg_name="stage_1_d10")
    mlp_call = profile_step_remote.spawn(cfg_name="stage_0_d10")
    le = le_call.get()
    mlp = mlp_call.get()
    print(f"\n=== Comparison (amortised per gradient step) ===")
    print(f"leMLP   amortised: {le['amortised_per_step_ms']:.2f} ms  ({1000/le['amortised_per_step_ms']:.1f} steps/sec)")
    print(f"vanilla amortised: {mlp['amortised_per_step_ms']:.2f} ms  ({1000/mlp['amortised_per_step_ms']:.1f} steps/sec)")
    print(f"leMLP / vanilla:   {le['amortised_per_step_ms'] / mlp['amortised_per_step_ms']:.2f}x slower")
