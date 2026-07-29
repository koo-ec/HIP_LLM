"""High-level public API for HIP-LLM."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .failure_probability import FailureProb, LogprobsUnavailableError
from .results import FailureProbResult

try:
    __version__ = version("HIPLLM")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"

__all__ = [
    "FailureProb",
    "FailureProbResult",
    "LogprobsUnavailableError",
    "__version__",
]
