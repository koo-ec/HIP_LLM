"""High-level public API for HIP-LLM."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .failure_probability import FailureProb, LogprobsUnavailableError
from .operational_failure import (
    OperationalFailureProb,
    OperationalFailureResult,
    paper_inference_settings,
    quick_inference_settings,
)
from .results import FailureProbResult

try:
    __version__ = version("HIPLLM")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "unknown"

__all__ = [
    "FailureProb",
    "FailureProbResult",
    "LogprobsUnavailableError",
    "OperationalFailureProb",
    "OperationalFailureResult",
    "paper_inference_settings",
    "quick_inference_settings",
    "__version__",
]
