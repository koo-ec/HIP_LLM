"""Static validation for the StrategyQA operational-profile Colab notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


NOTEBOOK = Path("notebooks/strategyqa_openai_operational_profile_colab.ipynb")


def test_strategyqa_colab_is_a_valid_notebook() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells


def test_strategyqa_colab_code_cells_are_syntactically_valid() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        python_lines = [
            line
            for line in cell.source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        source = "\n".join(python_lines).strip()
        if not source:
            continue
        ast.parse(source, filename=f"{NOTEBOOK}:cell-{index}")


def test_strategyqa_colab_pins_inputs_and_uses_explicit_op() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "gpt-4.1-mini-2025-04-14" in text
    assert "1ba1e97452e189569357876f2854b01357ffbe37" in text
    assert "OperationalFailureProb" in text
    assert "operational_profile" in text
    assert "parse_strategyqa_answer" in text
    assert "OPENAI_API_KEY" in text
    assert "google.colab" in text
    assert "decomposition_steps" in text
    assert "selected_qids" in text
    assert "api_error_policy" in text
