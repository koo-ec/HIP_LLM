"""Sphinx configuration for the HIPLLM documentation."""

from __future__ import annotations

from importlib.metadata import version

project = "HIPLLM"
author = "HIP-LLM contributors"
release = version("HIPLLM")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
autosummary_generate = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = f"HIPLLM {release}"
