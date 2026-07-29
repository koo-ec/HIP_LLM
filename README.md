# HIP-LLM

[![CI](https://github.com/koo-ec/HIP_LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/koo-ec/HIP_LLM/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ress.2026.112615-blue)](https://doi.org/10.1016/j.ress.2026.112615)

A tested implementation and replication package for:

> R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang and X. Zhao,<br>
> **“A hierarchical imprecise probability approach to reliability assessment of large language models”**,<br>
> *Reliability Engineering & System Safety* **272** (2026), 112615.

HIP-LLM models large-language-model reliability using hierarchical Bayesian inference, imprecise priors and operational profiles. This repository includes the reusable Python implementation, tests, source-provenance records, a generated replication notebook, and reproducible result artefacts.

<p align="center">
  <img src="docs/figures/General_Structure_2.PNG" alt="HIP-LLM hierarchy" width="80%">
</p>

## Scope and reproducibility

The package reproduces the disclosed statistical inference from published measurements. It is not an exact historical replay of the original API experiments because model snapshots, prompts, generation settings, item subsets and random seeds were not published.

Source discrepancies and reconstruction assumptions are recorded in [`data/provenance_manifest.yaml`](data/provenance_manifest.yaml) and discussed in [`results/reproduction_report.md`](results/reproduction_report.md). The implementation does not silently substitute missing source information.

## Quick start

### Requirements

- Python 3.10 or later
- Git

### Installation

```bash
git clone https://github.com/koo-ec/HIP_LLM.git
cd HIP_LLM
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test,profile]"
```

For the exact recorded dependency set:

```bash
python -m pip install -r requirements-lock.txt
```

A Conda environment is also available:

```bash
conda env create -f environment.yml
```

## Run the replication

Published-numerics mode requires no API keys or network calls:

```bash
python scripts/run_notebook.py --mode published_numerics --scalability-profile quick
```

Run the complete reproducibility pipeline:

```bash
bash scripts/reproduce_all.sh
```

Run the test suite without live, slow or notebook tests:

```bash
python -m pytest -m "not live and not slow and not notebook"
```

Strict-exact mode stops when a required source setting is unresolved:

```bash
python scripts/run_notebook.py --strict-exact
```

> [!CAUTION]
The `configs/live_api.yaml` file is a design specification for a future paid-provider evaluation pipeline; live execution is **not implemented** in this release. Never commit API keys or cached provider responses.

## Repository layout

```text
HIP_LLM/
├── src/hip_llm/                    reusable Python package
├── tests/                         unit and reproducibility tests
├── configs/                       published, scalability and optional live modes
├── data/                          reference data and provenance manifest
├── scripts/                       notebook, test and provenance tooling
├── results/                       generated figures, tables and report
├── docs/figures/                  paper and explanatory image assets
├── HIP_LLM_exact_replication.ipynb
└── pyproject.toml
```

## Documentation and results

- [Paper-oriented overview and visualisations](docs/paper-overview.md)
- [Replication report](results/reproduction_report.md)
- [Generated figures](results/figures/)
- [Generated tables](results/tables/)
- [Data provenance](data/provenance_manifest.yaml)
- [Repository documentation index](docs/README.md)

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md), follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and use the issue or pull-request templates. Please report security concerns according to [`SECURITY.md`](SECURITY.md), not in a public issue.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). A BibTeX entry is also available in the [paper overview](docs/paper-overview.md#-citation).

## Licence

The software and repository documentation are available under the [MIT Licence](LICENSE). Third-party publications and datasets retain their respective rights and licences.
