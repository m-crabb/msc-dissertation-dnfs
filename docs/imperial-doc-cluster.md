# Imperial DoC GPU cluster (Slurm) — operations guide

**Purpose:** everything needed to run this repo's GPU jobs on the Imperial
Department of Computing Slurm cluster, written so a future session (or model)
can operate it cold. All facts below were verified first-hand on 2026-07-24
unless marked otherwise. Official guide:
<https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/>.

**Why we're here:** Modal on-demand A100s were too expensive for eval-only
passes; the DoC cluster is already paid for. mars-node (also Slurm) may join
later — keep job scripts portable (paths in variables at the top).

---

## 1. Access

- Submission hosts: `gpucluster2.doc.ic.ac.uk` / `gpucluster3.doc.ic.ac.uk`.
  Direct external SSH is **disabled** (since 2025-12-17); reach them via
  College VPN or a jump through `shell[1-5].doc.ic.ac.uk`.
- `~/.ssh/config` on the user's Mac already has working aliases (key auth,
  connection multiplexing):
  - `ssh doc-shell` → shell1.doc.ic.ac.uk (user `mc625`)
  - `ssh gpucluster` → gpucluster2 via ProxyJump doc-shell
- The login node is a thin cloud VM (`cloud-vm-*.doc.ic.ac.uk`). Do not run
  anything heavy on it — including big NFS reads (see §5).

## 2. Storage

| Path | What | Rules |
|---|---|---|
| `/homes/mc625` | home | **Tiny quota.** Never build envs or caches here. |
| `/vol/gpudata/mc625-dnfs` | **CephFS workspace — primary** | Repo + env + results + pixi cache live here. **7× faster reads than bitbucket** (§5). Allocated via `ws_allocate`; 365-day expiry (extendable ×3). |
| `/vol/bitbucket/mc625` | dept NFS share | Legacy. Only the pixi *binary* (`.pixi/bin/pixi`) still lives here; slow under load, do not put envs here. |

**Canonical repo location on the cluster: `/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs`.**
The workspace was created with `ws_allocate -r 14 -m mc625@ic.ac.uk dnfs 365`
(`ws_list` to inspect, `ws_allocate -x ... dnfs 365` to extend before expiry).

## 3. Environment (pixi, locked)

No conda, no module system on the cluster. We install pixi and reuse the
repo's locked `cuda` environment — the **same solve as the Modal image**
(pytorch 2.10.0 + CUDA 12.9), so numerics stay comparable with the frozen
baselines. Do **not** substitute CSG's prebuilt starter env
(`/vol/bitbucket/starter`, torch 2.4.1+cu121) for real runs: six torch minor
versions of drift against the frozen `eval/` numbers. It remains useful as a
scheduling/GPU smoke test that is independent of our env.

- pixi binary: `/vol/bitbucket/mc625/.pixi/bin/pixi` (installed via
  `curl -fsSL https://pixi.sh/install.sh | bash` with pixi home moved to
  bitbucket).
- Install/refresh the env (login node has internet; conda-forge resolves).
  Install **into the gpudata workspace** so the ~4 GB env sits on fast storage:

  ```bash
  cd /vol/gpudata/mc625-dnfs/msc-dissertation-dnfs
  CONDA_OVERRIDE_CUDA=12.9 PIXI_CACHE_DIR=/vol/gpudata/mc625-dnfs/.pixi-cache \
      /vol/bitbucket/mc625/.pixi/bin/pixi install --environment cuda --locked
  ```

  Both env vars are load-bearing:
  - `CONDA_OVERRIDE_CUDA=12.9` — the login node has no GPU driver, so pixi
    sees no `__cuda` virtual package and refuses the env without this mock.
  - `PIXI_CACHE_DIR` — pixi's package cache defaults to `~/.cache/rattler`
    on /homes and **dies with "Quota exceeded (os error 122)"** mid-unpack.
    If that ever happens, `rm -rf ~/.cache/rattler` to reclaim the quota.

- Run anything with:

  ```bash
  /vol/bitbucket/mc625/.pixi/bin/pixi run --frozen -e cuda python ...
  ```

  `--frozen` activates the installed env without re-solving (compute nodes
  shouldn't depend on network).

## 4. Slurm essentials

- Partitions (GPU / count): `a100` 80 GB ×12 (contended, often queued),
  `a40` 48 GB ×7, `a30` 24 GB ×20 (best availability in practice), `a16`,
  `t4`. Per-user cap: 3 GPUs, 32 cores, 200 GB RAM. Walltime cap: 3 days.
- **`sbatch` does not source `~/.bashrc`** — export everything explicitly in
  the script and use absolute paths (this is why job scripts hardcode the
  pixi path).
- Useful: `squeue --me`, `squeue --me --start` (ETA), `sacct -j <id>`,
  `scancel <id>`, `sshare -U $USER` (fairshare), `sinfo`.
- Job scripts live in `slurm/` (repo), logs in `slurm/logs/` (must exist
  before submission — Slurm won't create the output directory).
- Template: `slurm/smc_tau_sweep.sbatch` — a 3-task array (one τ per task)
  showing the whole pattern: partition/gres/mem/time headers, the two env
  var exports, absolute pixi path, `cd` to the repo, module-style invocation
  of `experiments.*.run`.
- wandb: eval-only runs never touch it. Training does. `~/.bashrc` isn't
  sourced and wandb's `~/.netrc` lookup proved unreliable, so the **verified**
  path is: `wandb login <KEY>` once on the cluster (writes `~/.wandb`/netrc
  properly), OR export `WANDB_API_KEY` in the sbatch body. See §5a for the
  gotcha that bit us. To prove the pipeline without a key at all, set
  `WANDB_MODE=offline` (logs locally, `wandb sync` later).

## 5. Storage speed — why the env lives on gpudata (measured 2026-07-24)

`/vol/bitbucket` is one department-wide NFS server at ~98 % capacity and is
**slow**: measured sequential read **2.5 MB/s on the login VM, 10.3 MB/s on a
compute node**. Because a Python env is thousands of small files plus ~2–3 GB
of CUDA `.so`s, a *cold* `import torch` off bitbucket took **~16 min for the
eval path and >40 min for the training path** (extra wandb import tree) — the
process sits in `D` state on wchan `rpc_wait_bit_killable`, ~5 s CPU, pure NFS
wait. The starter env only *feels* fast because the whole department keeps it
permanently warm in cache.

**Fix (done): the env + repo live on `/vol/gpudata/mc625-dnfs` (CephFS).**
Measured there: **read 69.6 MB/s (7×), write 13.4 MB/s.** Result: the same
cold training start dropped from ~40 min to **~3 min**. This is now the
default; bitbucket keeps only the pixi binary.

Residual guidance:
- Still budget a few minutes of cold-import headroom in `#SBATCH --time`;
  it's small now but nonzero on a never-touched node.
- **Never** run imports or bulk reads on the login VM to "check" things —
  it is the slowest path and shared. Smoke-test via a tiny
  sbatch job instead (pattern: `slurm/env_probe.sbatch` — dd throughput +
  timed import + `torch.cuda.is_available()` on the target partition).
- If it ever matters again (many short jobs), the remaining escape hatch is
  staging a squashfs/tarball of the env to node-local disk at job start.
  Not needed at gpudata speeds.

## 5a. wandb-on-Slurm gotcha (cost us a cycle, 2026-07-24)

`~/.netrc`-based wandb auth is fragile here and bit us twice:
- `sbatch` never sources a shell profile, so a `WANDB_API_KEY` in `.bashrc`
  is invisible to the job.
- An indented/append-built `machine api.wandb.ai` block in `~/.netrc` did
  **not** populate — the cluster file ended up 0 bytes despite the append
  command printing success, so `awk`-extracting the key at runtime yielded
  an empty string and training died at `wandb.init()` with
  `UsageError: No API key configured`.

**Do this instead:** `wandb login <KEY>` once on the cluster (writes auth the
way wandb itself expects), then training sbatch jobs need nothing extra. To
prove a training pipeline *without* any credential, run with
`WANDB_MODE=offline` (logs to a local `wandb/` dir; `wandb sync <dir>` later).

## 6. Standard workflow (Mac → cluster → Mac)

All cluster paths are under the gpudata workspace
`/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs` (abbreviated `$W` below).

```bash
W=/vol/gpudata/mc625-dnfs/msc-dissertation-dnfs

# 1. Sync code (from the Mac repo root; excludes envs/wandb only — results
#    are synced so staged checkpoints ride along):
rsync -az --exclude '.pixi' --exclude 'wandb' --exclude '__pycache__' \
    --exclude '.pytest_cache' --exclude '*.egg-info' \
    ./ gpucluster:$W/

# 2. Stage a run dir the job needs (mkdir parents first — the Mac's old
#    rsync 2.6.9 silently fails on missing nested destination dirs):
ssh gpucluster "mkdir -p $W/results/03_hard"
rsync -az results/03_hard/<run_dir> gpucluster:$W/results/03_hard/

# 3. Submit and watch:
ssh gpucluster "cd $W && sbatch slurm/<script>.sbatch"
ssh gpucluster squeue --me

# 4. Retrieve artefacts back into the local results tree:
rsync -az gpucluster:$W/results/03_hard/<run_dir>/ results/03_hard/<run_dir>/
```

Checkpoints that only exist on the Modal volume are pulled locally first
(`pixi run -e default modal volume get dnfs-results <path> <local>`), then
staged with step 2.

## 7. Known history / campaign log

- 2026-07-09: starter-env GPU smoke test (job 258416, A16 node gpuvm36) —
  scheduling + CUDA verified. Leftovers in `/vol/bitbucket/mc625/`
  (`smoke_gpu.sbatch`, `dnfs-env/` abandoned venv) are ignorable.
- 2026-07-24: pixi cuda env verified on a30 (job 265686: torch 2.10.0, CUDA,
  `run.py` imports). SMC τ sweep ran as array 265690 (τ = 0.3/0.5/0.7) — all
  τ gave zero resample events (SMC ≡ plain IS on that checkpoint).
- 2026-07-24 (same evening): hit the >40 min cold-import wall on the training
  path off bitbucket → migrated env+repo to gpudata workspace
  `/vol/gpudata/mc625-dnfs`; cold training start dropped to ~3 min. wandb auth
  gotcha diagnosed (§5a). Full training path proven via a `WANDB_MODE=offline`
  smoke; online wandb pending a one-time `wandb login` after key rotation.
