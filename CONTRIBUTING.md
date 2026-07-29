# Contributing to HIP-LLM

Thank you for helping improve HIP-LLM. Contributions should preserve the repository's reproducibility, provenance and safety guarantees.

## Before opening a change

1. Search existing issues and pull requests.
2. Open an issue for substantial behavioural or methodological changes.
3. Do not include API keys, provider responses containing sensitive data, or proprietary datasets.
4. Distinguish faithfully reproduced results from reconstructions and new experiments.

## Development setup

```bash
git clone https://github.com/koo-ec/HIP_LLM.git
cd HIP_LLM
python -m venv .venv
python -m pip install -e ".[test,profile,dev]"
```

Create a focused branch, for example `fix/envelope-orientation` or `docs/provenance-note`.

## Quality checks

Run these before submitting a pull request:

```bash
python -m ruff check src tests scripts
python -m pytest -m "not live and not slow and not notebook"
python -m build
```

Changes to inference code should include tests against an analytical, quadrature or independently computed reference. Changes to published inputs must update `data/provenance_manifest.yaml` and preserve the original source artefact rather than overwriting it.

Generated figures, tables and reports are tracked because they are part of the replication deliverable. Regenerate affected artefacts and explain any numerical difference in the pull request.

## Pull requests

- Keep each pull request focused.
- Explain the scientific or engineering motivation.
- State whether outputs changed and why.
- Include commands used for verification.
- Do not run live API experiments unless the change explicitly requires them and the cost has been approved.

By contributing, you agree that your contribution will be licensed under the repository's MIT Licence.
