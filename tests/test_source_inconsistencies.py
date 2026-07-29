"""Source-integrity tests (specification items 31-35).

These assert facts about the PUBLISHED artifacts.  Several of them assert that a
discrepancy IS present: if one of those ever fails, the source has changed and
the reproduction report must be revisited.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hip_llm.benchmark_eval import accuracy_to_counts, load_accuracy_table
from hip_llm.schemas import load_yaml, sha256_file
from hip_llm.validation import (
    assert_no_overwrite,
    check_fig9_against_sources,
    check_table5_internal_consistency,
    compare_accuracy_sources,
)

REPO_CSV_SHA256 = "ccbabdf8fe4d0a83f53f0d1ad88a74256a0b49af3b0854f35431bad73001085f"


@pytest.fixture(scope="module")
def manifest(pytestconfig):
    root = pytestconfig.rootpath
    return load_yaml(root / "data" / "provenance_manifest.yaml")


@pytest.fixture
def table3(reference_dir):
    return load_accuracy_table(reference_dir / "paper_table3.csv")


@pytest.fixture
def repo(reference_dir):
    return load_accuracy_table(reference_dir / "official_figure_numerics.csv")


# --- 31. checksums ---------------------------------------------------------- #
def test_official_numerics_checksum_is_byte_identical_to_the_repository(reference_dir):
    """Our reference copy must be the repository file byte for byte."""
    actual = sha256_file(reference_dir / "official_figure_numerics.csv")
    assert actual == REPO_CSV_SHA256


def test_every_manifest_checksum_still_matches(manifest, pytestconfig):
    root = pytestconfig.rootpath
    checked = 0
    for entry in manifest["sources"]:
        local = entry.get("local_copy")
        expected = entry.get("sha256")
        if not local or not expected:
            continue
        path = root / local
        if not path.is_file():
            continue
        result = assert_no_overwrite(path, expected)
        assert result.passed, f"{entry['source_id']}: {result.detail}"
        checked += 1
    assert checked >= 5


# --- 32. Table 3 vs repository numerics ------------------------------------- #
def test_table3_and_repository_numerics_disagree(table3, repo):
    """Discrepancy D1 must be detected, quantified, and NOT silently reconciled."""
    cmp = compare_accuracy_sources(table3, repo)
    assert len(cmp) == 16
    mismatches = cmp[~cmp["match"]]
    assert len(mismatches) == 12, f"expected 12 differing cells, got {len(mismatches)}"

    worst = cmp.loc[cmp["abs_diff"].idxmax()]
    assert worst["model"] == "GPT-4o" and worst["subdomain"] == "RACE-H"
    assert worst["abs_diff"] == pytest.approx(0.368, abs=1e-9)

    # Only the Haiku 3.5 column agrees on all four subdomains.
    agreeing = cmp[cmp["match"]]
    assert set(agreeing["model"]) == {"Haiku 3.5"}
    assert len(agreeing) == 4


def test_the_two_accuracy_sources_are_never_merged(reference_dir):
    """The datasets must remain distinct files with distinct content."""
    a = sha256_file(reference_dir / "paper_table3.csv")
    b = sha256_file(reference_dir / "official_figure_numerics.csv")
    assert a != b


def test_config_assigns_each_figure_to_exactly_one_source(pytestconfig):
    cfg = load_yaml(pytestconfig.rootpath / "configs" / "paper_published_numerics.yaml")
    assignment = cfg["figure_source_assignment"]
    valid = {"official_figure_numerics", "paper_table3", "fig9_caption"}
    assert set(assignment.values()) <= valid
    # Discrepancy D4: Fig. 7 is driven by Table 3, every other main figure by the repo CSV.
    assert assignment["Fig7"] == "paper_table3"
    assert assignment["Fig3"] == "official_figure_numerics"
    assert assignment["Fig9"] == "fig9_caption"


def test_fig7_is_only_reproducible_from_table3(table3, repo):
    """Evidence for D4: the paper's own Fig. 7 narrative quotes Table 3 values.

    Paper Section 4.3.3 states theta_BoolQ ~ 0.91 vs theta_RACE-H ~ 0.55 for
    GPT-4o.  Only Table 3 supplies those; the repository gives 0.910 / 0.920.
    """
    def acc(df, model, sub):
        return float(df[(df["model"] == model) & (df["subdomain"] == sub)]["theta_hat"].iloc[0])

    assert acc(table3, "GPT-4o", "BoolQ") == pytest.approx(0.909, abs=1e-9)
    assert acc(table3, "GPT-4o", "RACE-H") == pytest.approx(0.552, abs=1e-9)
    assert acc(repo, "GPT-4o", "RACE-H") == pytest.approx(0.920, abs=1e-9)

    # Under Table 3 the domain aggregate falls as Omega_RACE rises (as plotted);
    # under the repository numerics it would RISE, contradicting the panel.
    def p2(df, omega_race):
        return (1 - omega_race) * acc(df, "GPT-4o", "BoolQ") + omega_race * acc(df, "GPT-4o", "RACE-H")

    assert p2(table3, 0.10) > p2(table3, 0.517) > p2(table3, 0.90)
    assert p2(repo, 0.10) < p2(repo, 0.517) < p2(repo, 0.90)

    # Reproduce the published Fig. 7b band centres from Table 3.
    assert p2(table3, 0.10) == pytest.approx(0.8733, abs=5e-4)
    assert p2(table3, 0.517) == pytest.approx(0.7244, abs=5e-4)
    assert p2(table3, 0.90) == pytest.approx(0.5877, abs=5e-4)


# --- 33. the Fig. 9 aggregate discrepancy ----------------------------------- #
def test_fig9_caption_conflicts_with_both_accuracy_sources(reference_dir, table3, repo):
    fig9 = pd.read_csv(reference_dir / "fig9_caption_numerics.csv")
    row = fig9[fig9["success_definition"] == "pass@1"].iloc[0]
    results = check_fig9_against_sources(
        {"pass1_accuracy": float(row["accuracy"]), "N": int(row["N"])}, table3, repo
    )
    assert not any(r.passed for r in results[:2]), "Fig. 9 should agree with neither source"
    assert results[0].data["abs_diff"] == pytest.approx(abs(0.471 - 0.450), abs=1e-9)
    assert results[1].data["abs_diff"] == pytest.approx(abs(0.471 - 0.486), abs=1e-9)
    assert not results[2].passed  # N = 257 differs from the baseline N = 80


def test_fig9_counts_round_consistently(reference_dir):
    fig9 = pd.read_csv(reference_dir / "fig9_caption_numerics.csv")
    for _, row in fig9.iterrows():
        assert accuracy_to_counts(float(row["accuracy"]), int(row["N"])) == int(row["successes"])


def test_fig9_pass3_exceeds_pass1(reference_dir):
    """Pass@3 is strictly more permissive than Pass@1 on identical tasks."""
    fig9 = pd.read_csv(reference_dir / "fig9_caption_numerics.csv")
    p1 = float(fig9[fig9["success_definition"] == "pass@1"]["accuracy"].iloc[0])
    p3 = float(fig9[fig9["success_definition"] == "pass@3"]["accuracy"].iloc[0])
    assert p3 > p1


# --- 34. Table 5's mathematically impossible rows --------------------------- #
def test_table5_has_exactly_two_inconsistent_hip_llm_rows(reference_dir):
    t5 = pd.read_csv(reference_dir / "paper_table5_printed.csv")
    results = check_table5_internal_consistency(t5)
    failed = [r for r in results if not r.passed]
    assert len(failed) == 2, [r.name for r in failed]
    names = " | ".join(r.name for r in failed)
    assert "OPapprox" in names and "HIP-LLM" in names
    assert "OPGT" in names
    # Each check must name the table it came from, not a hard-coded one.
    assert all(r.name.startswith("Table5") or r.name.startswith("Table 5") for r in results)

    for r in failed:
        med = r.data["median"]
        lo, hi = r.data["interval"]
        assert med[1] < lo, f"{r.name}: expected the median to fall BELOW the interval"


def test_table4_is_internally_consistent(reference_dir):
    """The same invariant passes on Table 4 -- the defect is specific to Table 5."""
    t4 = pd.read_csv(reference_dir / "paper_table4_printed.csv")
    results = check_table5_internal_consistency(t4)
    assert all(r.passed for r in results), [r.name for r in results if not r.passed]
    # Rows must be labelled with their own table, not with a hard-coded "Table 5".
    assert all("Table4" in r.name or "Table 4" in r.name for r in results), results[0].name


def test_printed_errors_are_consistent_with_printed_medians(reference_dir):
    """|median - p_GT| must reproduce the printed error column in both tables."""
    for name in ("paper_table4_printed.csv", "paper_table5_printed.csv"):
        df = pd.read_csv(reference_dir / name)
        for _, r in df.iterrows():
            errs = sorted(
                (abs(r["median_lower"] - r["p_gt"]), abs(r["median_upper"] - r["p_gt"]))
            )
            assert errs[0] == pytest.approx(r["error_lower"], abs=5e-5), (name, r["method"])
            assert errs[1] == pytest.approx(r["error_upper"], abs=5e-5), (name, r["method"])


def test_a_correct_implementation_is_not_forced_to_emit_the_invalid_values(
    toy_domain, fast_settings, wide_interval
):
    """Our own output must satisfy the invariant that Table 5 violates."""
    from hip_llm.envelopes import quantile_envelope
    from hip_llm.posterior import run_domain

    ds = run_domain(toy_domain, wide_interval, fast_settings, domain_index=0)
    m_lo, m_hi = quantile_envelope(ds.p, 0.50)
    c_lo, _ = quantile_envelope(ds.p, 0.05)
    _, c_hi = quantile_envelope(ds.p, 0.95)
    assert c_lo <= m_lo <= m_hi <= c_hi


# --- 35. sources cannot be silently overwritten ----------------------------- #
def test_overwrite_guard_detects_modification(tmp_path):
    p = tmp_path / "ref.csv"
    p.write_text("model,domain,subdomain,theta_hat\nX,D1,A,0.5\n", encoding="utf-8")
    good = sha256_file(p)
    assert assert_no_overwrite(p, good).passed
    p.write_text("model,domain,subdomain,theta_hat\nX,D1,A,0.6\n", encoding="utf-8")
    result = assert_no_overwrite(p, good)
    assert not result.passed and "expected" in result.detail


def test_loader_refuses_a_checksum_mismatch(reference_dir):
    from hip_llm.benchmark_eval import BenchmarkLoadError

    with pytest.raises(BenchmarkLoadError, match="checksum mismatch"):
        load_accuracy_table(reference_dir / "official_figure_numerics.csv", expected_sha256="0" * 64)


# --- D5: Table 3's own printed mean is internally inconsistent -------------- #
def test_table3_printed_llm_mean_for_gpt4o_disagrees_with_its_own_cells(table3, reference_dir):
    """D5: three columns agree to a rounding unit; GPT-4o is off by 3 units."""
    means = pd.read_csv(reference_dir / "paper_table3_printed_means.csv", comment="#")
    printed = means[means["statistic"] == "llm_mean"].iloc[0]
    deviations = {}
    for model in ("GPT-4o-mini", "GPT-4o", "Sonnet 4.5", "Haiku 3.5"):
        recomputed = float(table3[table3["model"] == model]["theta_hat"].mean())
        deviations[model] = abs(recomputed - float(printed[model]))

    # One rounding unit at three decimals is 0.001.
    assert deviations["GPT-4o"] == pytest.approx(0.003, abs=5e-4)
    for model in ("GPT-4o-mini", "Sonnet 4.5", "Haiku 3.5"):
        assert deviations[model] <= 0.001 + 1e-9, (model, deviations[model])
    assert deviations["GPT-4o"] > 2 * max(
        deviations[m] for m in ("GPT-4o-mini", "Sonnet 4.5", "Haiku 3.5")
    )


# --- D6: the perturbation conflict is recorded, not resolved ---------------- #
def test_perturbation_conflict_is_recorded_and_unused(pytestconfig):
    cfg = load_yaml(pytestconfig.rootpath / "configs" / "paper_published_numerics.yaml")
    assert cfg["sampling"]["perturbation_setting_from_repository"] == 0.07
    assert cfg["sampling"]["perturbation_semantics"] == "UNVERIFIED"
    rq5 = load_yaml(pytestconfig.rootpath / "configs" / "synthetic_rq5.yaml")
    assert rq5["operational_profiles"]["perturbation_magnitude"] is None


def test_manifest_catalogues_every_known_discrepancy(manifest):
    ids = {d["id"] for d in manifest["discrepancies"]}
    assert {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11"} <= ids
    for d in manifest["discrepancies"]:
        assert d["resolution"], f"{d['id']} has no recorded resolution"
        assert d["severity"] in {"high", "medium", "low", "informational"}


def test_manifest_records_the_exhaustive_repository_search(manifest):
    entry = next(s for s in manifest["sources"] if s["source_id"] == "official_repo_history")
    note = entry["note"]
    assert "NO IMPLEMENTATION SOURCE CODE" in note
    assert "22 commits" in note


# --- global settings agree between the paper and the repository ------------- #
def test_paper_and_repository_global_settings_agree(pytestconfig):
    cfg = load_yaml(pytestconfig.rootpath / "configs" / "paper_published_numerics.yaml")
    assert cfg["weights"]["omega"]["D1"] == [0.204, 0.796]
    assert cfg["weights"]["omega"]["D2"] == [0.483, 0.517]
    assert cfg["weights"]["W"] == [0.149, 0.851]
    assert cfg["effective_sample_size"]["N_per_subdomain"] == 80
    assert cfg["hyperpriors"] == {
        "a_range": [1, 12], "b_range": [1, 12], "c_range": [1, 25], "d_range": [1, 25]
    }
    assert cfg["grids"]["mu_points"] * cfg["grids"]["nu_points"] == cfg["grids"]["total_G"] == 2000
    assert cfg["grids"]["cdf_points_T"] == 201
    assert cfg["sampling"]["S"] == 3000
    assert cfg["sampling"]["K_per_domain"] == 160
    assert cfg["sampling"]["max_llm_configuration_pairs"] == 512
    assert cfg["seeds"] == {"global": 7, "configurations": 123, "llm_pairs": 999}


def test_table1_literal_none_entries_survive_loading(reference_dir):
    """Miller's 'None' cells must not be swallowed as pandas NA.

    Paper Table 1 records Miller [32] as having no hierarchical structure and no
    prior specification, written literally as "None". A default ``read_csv``
    turns those into NaN and then writes empty cells, silently dropping content
    from a verbatim reference artifact.
    """
    path = reference_dir / "paper_table1_framework_comparison.csv"
    default = pd.read_csv(path)
    literal = pd.read_csv(path, keep_default_na=False)

    assert default["Miller [32]"].isna().sum() == 2, "expected two literal 'None' cells"
    assert (literal != "").all().all()
    assert (literal["Miller [32]"] == "None").sum() == 2
    assert literal.loc[literal["aspect"] == "Hierarchical structure", "Miller [32]"].iloc[0] == "None"
    assert literal.loc[literal["aspect"] == "Prior specification", "Miller [32]"].iloc[0] == "None"


def test_effective_counts_are_recorded_alongside_the_original_accuracies(repo):
    """C_ij = round(theta_hat * 80) must be reported next to the source accuracy."""
    for _, row in repo.iterrows():
        C = accuracy_to_counts(float(row["theta_hat"]), 80)
        assert 0 <= C <= 80
        assert abs(C / 80 - float(row["theta_hat"])) <= 0.5 / 80 + 1e-12
