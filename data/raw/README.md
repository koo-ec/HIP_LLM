# `data/raw/` — immutable inputs, never edited in place

Raw data is kept strictly separate from processed data.

* `api_cache/` — created by Mode B (`live_api`). Holds **completed provider
  responses only**, keyed by `(provider, snapshot, generation fingerprint,
  rendered prompt, generation index)`, each with its timestamp, resolved model
  id and token usage. `ResponseCache.put` refuses any record without a real
  `raw_response`, so a fabricated example cannot enter this directory.
  Git-ignored: it may contain benchmark content and is per-account.

* Benchmark exports — if you supply `local_path` to
  `hip_llm.benchmark_eval.load_benchmark_split`, put the JSONL export here.

Nothing in this directory is modified after it is written. Derived artifacts go
to `data/processed/`; published reference datasets live in `data/reference/`
and are checksummed in `data/provenance_manifest.yaml`.
