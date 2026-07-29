#!/usr/bin/env python
"""Execute the replication notebook from a clean kernel via Papermill.

Executing with Papermill (rather than opening the notebook) is what proves the
notebook has no hidden state and no out-of-order dependencies: the kernel starts
empty and every cell runs top to bottom in source order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "HIP_LLM_exact_replication.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(ROOT / "results" / "diagnostics" / "executed_notebook.ipynb"))
    ap.add_argument("--mode", default="published_numerics", choices=["published_numerics"])
    ap.add_argument("--scalability-profile", default="full", choices=["full", "quick", "off"])
    ap.add_argument("--strict-exact", action="store_true", help="abort on any unresolved source setting")
    ap.add_argument("--results-root", default="results",
                    help="where artifacts are written (use a scratch dir for trial runs)")
    ap.add_argument("--timeout", type=int, default=7200, help="per-cell timeout in seconds")
    args = ap.parse_args()

    try:
        import papermill as pm
    except ImportError:
        print("papermill is not installed:  pip install papermill ipykernel", file=sys.stderr)
        return 2

    if not NOTEBOOK.is_file():
        print(f"notebook not found: {NOTEBOOK}", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"executing {NOTEBOOK.name} -> {out}")
    pm.execute_notebook(
        str(NOTEBOOK),
        str(out),
        parameters={
            "RUN_MODE": args.mode,
            "SCALABILITY_PROFILE": args.scalability_profile,
            "STRICT_EXACT": bool(args.strict_exact),
            "RESULTS_ROOT": args.results_root,
        },
        cwd=str(ROOT),
        kernel_name="python3",
        execution_timeout=args.timeout,
        progress_bar=False,
    )
    print("notebook executed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
