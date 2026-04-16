# discrete-flow-sampler

Code repository for MSc dissertation on discrete neural flow samplers at Imperial College London.

## Setup

Environment and dependencies are managed with [pixi](https://pixi.sh).

```bash
pixi install
pixi run -e dev test
```

## Tasks

- `pixi run -e dev test` — run the test suite
- `pixi run -e dev lint` — ruff lint `src/`, `experiments/`, `tests/`
- `pixi run -e dev format` — ruff format the same directories
- `pixi run jupyter` — launch JupyterLab

## Layout

- `src/discrete_flow_sampler/` — library code (`models/`, `samplers/`, `constraints/`, `utils/`)
- `experiments/` — numbered experiment runs
- `tests/` — unit tests
- `configs/`, `data/`, `results/`, `notebooks/`, `docs/` — project assets
