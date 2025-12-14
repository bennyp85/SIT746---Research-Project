# SIT746 Honours Research Project

Python code + notebooks for the SIT746 research project. The repository currently contains:
- Time-series preprocessing utilities and exploratory work on symbolic encodings (SAX + simple trend features / “TFSAX”-style).
- Early scaffolding for quantum-ML experiments (module stubs; quantum dependencies are not yet wired in).

## Current State

- **Packaged code lives in `src/`** (setuptools “src layout”).
- **Implemented utilities:** `src/data/preprocessing.py` (TSF parsing, resampling helpers, PAA + trend direction extraction).
- **Exploratory notebooks:** `experiments/m4_eda.ipynb`, `experiments/tfsax_encoding.ipynb`, plus additional notebooks in `notebooks/`.
- **Quantum code:** `src/quantum_ml/` exists but is currently minimal (placeholders).

## Project Layout

```
.
├── config/
│   └── default.yml
├── docs/
│   ├── design_docs/
│   ├── experiments/
│   └── literature/
├── experiments/                # notebook-led experiments
├── notebooks/                  # general exploration notebooks
├── results/                    # outputs (figures/logs/models/etc.)
└── src/
    ├── data/
    │   ├── preprocessing.py
    │   └── datasets/
    │       └── M4/             # bundled .tsf files used by notebooks
    ├── experiments/            # (currently mostly scaffolding)
    └── quantum_ml/             # (currently mostly scaffolding)
```

## Setup

### Requirements (what’s in `pyproject.toml`)

- Python `>=3.9`
- Core deps: `numpy`, `matplotlib`

Some notebooks/utilities also import packages that are not listed as core deps yet (e.g. `pandas`, `scipy`, Jupyter). Install them as needed for notebook work.

### Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Notebooks

```bash
pip install jupyter pandas scipy
jupyter lab
```

## Documentation

- Experiment logging template: `docs/experiments/README.md`
- Literature review notes: `docs/literature/README.md`
- Design notes: `docs/design_docs/`

## Project Info

- Institution: Deakin University
- Unit: SIT746 – Research Project

Note: this repository is under active development; modules and dependencies will change as experiments evolve.
