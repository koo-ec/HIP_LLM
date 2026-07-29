"""Notebook tests (specification items 41-43).

The structural checks run everywhere and are cheap.  The full Papermill
execution is marked ``notebook`` because it re-runs the whole pipeline.
"""

from __future__ import annotations

import json
import re

import pytest

NOTEBOOK_NAME = "HIP_LLM_exact_replication.ipynb"

REQUIRED_SECTIONS = [
    "0. Title, claim boundary, and run mode",
    "1. Environment and provenance",
    "2. Paper audit",
    "3. Source discrepancy report",
    "4. Coin-flip imprecise-probability sanity example",
    "5. Minimal executable HIP-LLM example",
    "6. Full published-numerics model",
    "7. RQ1",
    "8. RQ2",
    "9. RQ3",
    "10. RQ4",
    "11. RQ5",
    "12. RQ6",
    "13. RQ7",
    "14. RQ8",
    "15. Conclusions and reproducibility status",
]


@pytest.fixture(scope="module")
def notebook(pytestconfig):
    path = pytestconfig.rootpath / NOTEBOOK_NAME
    assert path.is_file(), f"notebook not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def notebook_text(notebook):
    return "\n".join("".join(c["source"]) for c in notebook["cells"])


def test_notebook_is_valid_and_non_trivial(notebook):
    import nbformat

    nbformat.validate(nbformat.from_dict(notebook))
    assert len(notebook["cells"]) > 40


def test_every_required_section_is_present(notebook_text):
    missing = [s for s in REQUIRED_SECTIONS if s not in notebook_text]
    assert not missing, f"missing notebook section(s): {missing}"


def test_notebook_ships_without_stale_outputs(notebook):
    """Committed outputs would let a stale result masquerade as a fresh one."""
    with_output = [
        i for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "code" and c.get("outputs")
    ]
    assert not with_output, f"code cells carry committed output: {with_output[:10]}"


def test_execution_counts_are_cleared(notebook):
    counts = [
        c.get("execution_count") for c in notebook["cells"] if c["cell_type"] == "code"
    ]
    assert all(c is None for c in counts), "notebook was saved with execution counts"


def test_parameters_cell_is_tagged_for_papermill(notebook):
    tagged = [c for c in notebook["cells"] if "parameters" in c.get("metadata", {}).get("tags", [])]
    assert len(tagged) == 1, "exactly one cell must carry the 'parameters' tag"
    src = "".join(tagged[0]["source"])
    for name in ("RUN_MODE", "SCALABILITY_PROFILE", "STRICT_EXACT", "RESULTS_ROOT"):
        assert name in src


def test_notebook_defines_no_giant_functions(notebook):
    """Core logic must live in importable modules, not in notebook-only blobs."""
    offenders = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for match in re.finditer(r"^def\s+(\w+)", src, re.MULTILINE):
            start = src[: match.start()].count("\n")
            body = src.splitlines()[start:]
            length = 0
            for line in body[1:]:
                if line.strip() and not line.startswith((" ", "\t")):
                    break
                length += 1
            if length > 45:
                offenders.append((i, match.group(1), length))
    assert not offenders, f"move these into hip_llm/: {offenders}"


def test_notebook_imports_the_package(notebook_text):
    assert "from hip_llm" in notebook_text or "import hip_llm" in notebook_text


def test_notebook_contains_no_hard_coded_published_results(notebook_text):
    """Published values may appear only as clearly-labelled reference artifacts."""
    # The paper's HIP-LLM Table 5 medians must never be assigned as computed output.
    for forbidden in ("0.5713", "0.5751", "0.5860, 0.5866"):
        for match in re.finditer(re.escape(forbidden), notebook_text):
            window = notebook_text[max(0, match.start() - 300) : match.start() + 120].lower()
            assert any(
                marker in window
                for marker in ("printed", "reference", "paper_table", "blocked", "p_gt",
                               "ground_truth", "p_l_ground_truth")
            ), f"{forbidden!r} appears without a reference/printed label"


def test_notebook_states_the_claim_class(notebook_text):
    lowered = notebook_text.lower()
    assert "exact statistical reproduction from published measurements" in lowered
    assert "claim" in lowered


# --- 43. every figure/table has a machine-readable counterpart -------------- #
@pytest.mark.notebook
def test_reported_artifacts_have_machine_readable_files(pytestconfig):
    root = pytestconfig.rootpath
    figures = root / "results" / "figures"
    tables = root / "results" / "tables"
    assert figures.is_dir() and tables.is_dir(), "run the notebook first"

    pngs = sorted(figures.glob("*.png"))
    assert pngs, "no figures were produced"
    for png in pngs:
        for ext in (".pdf", ".svg", ".meta.json"):
            companion = png.with_suffix("") .with_suffix(ext) if ext == ".meta.json" else png.with_suffix(ext)
            assert companion.exists(), f"missing {companion.name}"

    csvs = {p.stem for p in tables.glob("*.csv")}
    jsons = {p.stem for p in tables.glob("*.json")}
    assert csvs, "no tables were produced"
    assert csvs <= jsons, f"tables without a JSON twin: {sorted(csvs - jsons)}"


@pytest.mark.notebook
def test_figure_metadata_records_provenance(pytestconfig):
    figures = pytestconfig.rootpath / "results" / "figures"
    metas = sorted(figures.glob("*.meta.json"))
    assert metas, "run the notebook first"
    for m in metas:
        payload = json.loads(m.read_text(encoding="utf-8"))
        for key in ("figure_id", "caption", "config_hash", "data_source", "seeds", "generated_utc"):
            assert key in payload and payload[key] not in (None, ""), (m.name, key)
        assert len(payload["config_hash"]) == 64


@pytest.mark.notebook
def test_reproduction_report_exists_and_states_one_claim_class(pytestconfig):
    report = pytestconfig.rootpath / "results" / "reproduction_report.md"
    assert report.is_file(), "run the notebook first"
    text = report.read_text(encoding="utf-8")
    for section in ("Executive summary", "Source audit", "Results by research question",
                    "Source conflicts", "Numerical diagnostics", "Final claim"):
        assert section in text, f"missing report section: {section}"
    claims = [
        "Exact historical end-to-end reproduction",
        "Exact statistical reproduction from published measurements",
        "Faithful contemporary end-to-end rerun",
        "Partial reproduction with documented gaps",
        "Reproduction blocked by missing information",
    ]
    tail = text.split("Final claim")[-1]
    chosen = [c for c in claims if c in tail]
    assert len(chosen) == 1, f"the final claim must name exactly one class, found {chosen}"


# --- 41, 42. clean-kernel execution ----------------------------------------- #
@pytest.mark.notebook
@pytest.mark.slow
def test_notebook_executes_from_a_clean_kernel(pytestconfig, tmp_path):
    papermill = pytest.importorskip("papermill")
    root = pytestconfig.rootpath
    out = tmp_path / "executed.ipynb"
    # Artifacts go to a temp directory: a reduced-profile test run must never
    # overwrite the recorded full-profile results that ship with the package.
    papermill.execute_notebook(
        str(root / NOTEBOOK_NAME),
        str(out),
        parameters={
            "RUN_MODE": "published_numerics",
            "SCALABILITY_PROFILE": "quick",
            "STRICT_EXACT": False,
            "RESULTS_ROOT": str(tmp_path / "results"),
        },
        cwd=str(root),
        kernel_name="python3",
        progress_bar=False,
    )
    assert (tmp_path / "results" / "reproduction_report.md").is_file()
    assert (root / "results" / "reproduction_report.md").is_file()
    executed = json.loads(out.read_text(encoding="utf-8"))
    errors = [
        o for c in executed["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    assert not errors, f"notebook raised: {errors[0].get('evalue') if errors else ''}"

    counts = [
        c.get("execution_count")
        for c in executed["cells"]
        if c["cell_type"] == "code" and c.get("execution_count") is not None
    ]
    assert counts == sorted(counts), "cells executed out of source order (hidden state)"
