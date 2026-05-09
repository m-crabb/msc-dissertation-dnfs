# discrete-flow-sampler

Code repository for MSc dissertation on discrete neural flow samplers at Imperial College London.

## Setup

Environment and dependencies are managed with [pixi](https://pixi.sh). Install pixi first:

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # macOS / Linux
# or see https://pixi.sh/latest/#installation for other platforms
```

The lockfile (`pixi.lock`) is committed, so installs resolve against the
same dependency set used for the Modal runs. Supported platforms are
`linux-64` and `osx-arm64`; other platforms fail at the resolve step.

```bash
pixi install            # resolve and install the default environment
pixi run -e dev test    # smoke-test the dev environment
```

## Tasks

- `pixi run -e dev test` — run the test suite
- `pixi run -e dev lint` — ruff lint `src/`, `experiments/`, `tests/`
- `pixi run -e dev format` — ruff format the same directories
- `pixi run jupyter` — launch JupyterLab

## Layout

- `src/discrete_flow_sampler/` — library code: `targets/`, `mcmc/`, `models/`, `samplers/`, `diagnostics/`, `constraints/`
- `experiments/` — numbered experiment dirs in `<name>_NN` form (Python module identifiers can't begin with a digit)
- `notebooks/` — exploratory and tutorial notebooks
- `tests/` — pytest suite (run via `pixi run -e dev test`)
- `data/`, `results/` — project assets (gitignored; per-run artefacts written under `results/`)

## Acknowledgements

This work is heavily inspired by that of Zijing Ou, whose original code can be found at https://github.com/J-zin/DNFS.
