# `data/processed/` — derived from `data/raw/`, regenerable

Everything here is reproducible from `data/raw/` plus the recorded configuration
hash and seeds; nothing here is a primary source.

Mode B writes per-task evaluation records here (task id, benchmark, model
snapshot, binary outcome, failure kind, token usage), from which
`BenchmarkResult` builds the `(C, N)` counts consumed by the inference pipeline.

Mode A needs nothing here: it reads the authors' published numerics directly
from `data/reference/official_figure_numerics.csv`, whose SHA-256 is verified
against `data/provenance_manifest.yaml` on every run.
