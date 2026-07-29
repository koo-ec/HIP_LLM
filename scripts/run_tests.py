#!/usr/bin/env python
"""Run the test suite.

By default the ``live`` marker is deselected, so no provider API is contacted
and no money is spent.  ``--live`` opts in; in that mode a missing API key or an
unavailable model snapshot is a FAILURE, never a skip.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="include tests that hit real provider APIs")
    ap.add_argument("--slow", action="store_true", help="include slow convergence/scalability tests")
    ap.add_argument("--notebook", action="store_true", help="include the papermill notebook test")
    args, pytest_args = ap.parse_known_args()

    marker_parts: list[str] = []
    if not args.live:
        marker_parts.append("not live")
    if not args.slow:
        marker_parts.append("not slow")
    if not args.notebook:
        marker_parts.append("not notebook")

    cmd = [sys.executable, "-m", "pytest", "-q"]
    if marker_parts:
        cmd += ["-m", " and ".join(marker_parts)]
    cmd += pytest_args

    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
