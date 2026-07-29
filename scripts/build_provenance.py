#!/usr/bin/env python
"""Recompute the checksums recorded in ``data/provenance_manifest.yaml``.

Rewrites only the ``sha256`` fields of entries that carry a ``local_copy``,
leaving every narrative field untouched.  Run this after refreshing a reference
dataset so that the manifest can never drift from the bytes on disk.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "provenance_manifest.yaml"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"manifest not found: {MANIFEST}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT / "src"))
    from hip_llm.schemas import sha256_file

    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    changed = 0
    for entry in data.get("sources", []):
        local = entry.get("local_copy")
        if not local:
            continue
        p = Path(local)
        if not p.is_absolute():
            p = ROOT / local
        if not p.is_file():
            p = ROOT.parent / local
        if not p.is_file():
            print(f"  ! missing local copy for {entry['source_id']}: {local}")
            continue
        actual = sha256_file(p)
        if entry.get("sha256") != actual:
            print(f"  updated {entry['source_id']}: {entry.get('sha256')} -> {actual}")
            entry["sha256"] = actual
            changed += 1
        else:
            print(f"  ok      {entry['source_id']}")

    data["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    print(f"{changed} checksum(s) updated; manifest rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
