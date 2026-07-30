"""Sphinx configuration for the HIP-LLM documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version

project = "HIP-LLM"
author = "HIP-LLM contributors"
copyright = "2026, HIP-LLM contributors"

try:
    release = package_version("HIPLLM")
except PackageNotFoundError:
    release = "1.1.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_sitemap",
]

autosummary_generate = True
autodoc_typehints = "description"
myst_heading_anchors = 3
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

html_theme = "furo"
html_title = "HIP-LLM: Reliability Assessment for Large Language Models"
html_short_title = "HIP-LLM"
html_baseurl = "https://koo-ec.github.io/HIP_LLM/"
html_logo = "_static/hip-llm-logo.svg"
html_favicon = "_static/hip-llm-mark.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["robots.txt", "llms.txt", "llms-full.txt"]
html_copy_source = False
html_show_sourcelink = False
html_last_updated_fmt = "%d %B %Y"
html_use_opensearch = html_baseurl

html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/koo-ec/HIP_LLM/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#008f86",
        "color-brand-content": "#007f78",
        "color-api-name": "#007f78",
        "color-api-pre-name": "#596575",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--headings": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#40d8ce",
        "color-brand-content": "#55e1d8",
        "color-api-name": "#55e1d8",
        "color-api-pre-name": "#a8b3c2",
    },
    "footer_icons": [
        {
            "name": "GitHub repository",
            "url": "https://github.com/koo-ec/HIP_LLM",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16" aria-hidden="true">
                  <path fill-rule="evenodd"
                    d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8z"/>
                </svg>
            """,
            "class": "",
        }
    ],
}

html_context = {
    "site_url": html_baseurl,
    "repository_url": "https://github.com/koo-ec/HIP_LLM",
    "default_description": (
        "HIP-LLM is an open-source Python implementation for reliability assessment "
        "of large language models using hierarchical Bayesian inference, imprecise "
        "probability and explicit operational profiles."
    ),
    "social_image": (
        "https://raw.githubusercontent.com/koo-ec/HIP_LLM/main/"
        "docs/figures/General_Structure_2.PNG"
    ),
    "page_descriptions": {
        "index": (
            "HIP-LLM documentation: estimate large language model failure under explicit "
            "operational profiles using hierarchical Bayesian and imprecise-probability methods."
        ),
        "quickstart": (
            "Install HIPLLM, estimate operational-profile failure probability, and run the "
            "StrategyQA OpenAI workflow in Google Colab."
        ),
        "api": (
            "HIPLLM Python API reference for operational-profile failure inference, StrategyQA "
            "utilities and token-confidence diagnostics."
        ),
        "paper": (
            "Paper, method and reproducibility resources for the hierarchical imprecise "
            "probability approach to large language model reliability assessment."
        ),
    },
    "page_keywords": {
        "index": "LLM reliability, large language model safety, imprecise probability, operational profile, Bayesian reliability",
        "quickstart": "HIPLLM tutorial, LLM failure probability, StrategyQA, Google Colab, Python",
        "api": "HIPLLM API, OperationalFailureProb, FailureProb, Python documentation",
        "paper": "LLM reliability paper, hierarchical Bayesian inference, imprecise probability, reproducibility",
    },
}

sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"
