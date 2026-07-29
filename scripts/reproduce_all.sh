#!/usr/bin/env bash
# Clean-room reproduction of the whole package, from a fresh environment.
#
#   bash scripts/reproduce_all.sh            # published numerics + tests + report
#   bash scripts/reproduce_all.sh --quick    # skip the slow n_bar scalability sweep
#
# The supplied live-API configuration is a design specification only; a complete
# paid-provider evaluation pipeline is not implemented in this release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="full"
for arg in "$@"; do
  case "$arg" in
    --quick) PROFILE="quick" ;;
    --live) echo "live API execution is not implemented; configs/live_api.yaml is a design specification" >&2; exit 2 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

echo "==> [1/6] environment"
python --version
python -c "import numpy, scipy, pandas, matplotlib, yaml; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)"

echo "==> [2/6] install (editable)"
python -m pip install -q -e ".[test,profile]"

echo "==> [3/6] verify source checksums"
python scripts/build_provenance.py

echo "==> [4/6] tests (live tests deselected)"
python scripts/run_tests.py --slow

echo "==> [5/6] execute the replication notebook (mode=published_numerics, scalability=$PROFILE)"
python scripts/run_notebook.py --mode published_numerics --scalability-profile "$PROFILE"

echo "==> [6/6] artifacts"
ls -1 results/figures | head -50
ls -1 results/tables  | head -50
echo
echo "reproduction report: results/reproduction_report.md"
