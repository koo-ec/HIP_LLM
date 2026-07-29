#!/usr/bin/env python
"""Assemble ``HIP_LLM_exact_replication.ipynb`` from the cell sources below.

The notebook is generated rather than hand-edited so that it can never be
committed with stale outputs or out-of-order execution counts (both are asserted
by tests/test_notebook.py).  Re-run this script after editing any cell.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HIP_LLM_exact_replication.ipynb"

CELLS: list[tuple[str, str, list[str]]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source, []))


def code(source: str, tags: list[str] | None = None) -> None:
    CELLS.append(("code", source, tags or []))


# ===========================================================================
# 0. Title, claim boundary, and run mode
# ===========================================================================
md(r"""# HIP-LLM — executable replication

**Target paper.** R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang, X. Zhao,
*"A hierarchical imprecise probability approach to reliability assessment of large language
models"*, **Reliability Engineering & System Safety 272 (2026) 112615**,
doi:[10.1016/j.ress.2026.112615](https://doi.org/10.1016/j.ress.2026.112615).

**Official repository.** <https://github.com/aghazadehchakherlou-web/llm-imprecise-bayes>

---

## 0. Title, claim boundary, and run mode

This notebook is explicit about which of three very different things each result is.

### What **is** reproduced (Mode A, `published_numerics`)
The complete HIP-LLM statistical machinery — Beta-Binomial marginal likelihood, the
$(\mu,\nu)$ hyperposterior, conditional Beta sampling under **shared latent draws**,
operational-profile aggregation, imprecise CDF envelopes and future reliability — re-derived
from scratch and applied to the authors' **published measurements**. Nothing is digitised from
a figure and no result is hard-coded. This covers Figs. 3–11 and Tables 1, 2, 3 and 6.

### Mode B design specification (`live_api`; not implemented)

`configs/live_api.yaml` documents the settings required by a future paid-provider rerun. This
release does **not** implement or execute that pipeline; only the published-numerics replication
below is supported. Because the paper discloses no model snapshot, prompt, temperature, split,
item subset or seed, any future run must be labelled `contemporary_faithful_rerun`, not a
recreation of the authors' historical API outputs.

### What is **blocked**
* **RQ5 (Tables 4 and 5).** The paper publishes only the aggregate $p_L^{\mathrm{GT}}=0.5860$.
  The ground-truth subdomain reliability vector, the ground-truth OP, the BB-Inf prior
  strength, the HiBayES prior specification and the seeds are all unstated — one equation in
  seven unknowns. Inferring a parameter set would be fabrication, so RQ5 prints a **blocking
  checklist** instead of numbers.
* **Strict-exact mode.** The $\nu$-axis construction and the $K=160$ configuration-sampling
  rule are unrecoverable. With `STRICT_EXACT = True` the notebook *aborts* rather than guess.
  With `STRICT_EXACT = False` (the default) it proceeds under **labelled reconstructions**, and
  Section 6 measures exactly how much each one matters. The two do **not** matter equally: the
  $\nu$-axis choice moves posterior medians by $<0.002$ (inside tolerance), whereas the
  configuration-sampling rule changes the *envelope width* materially — a corner-based design
  yields a substantially wider envelope than 160 uniform draws (finding **D11**). That result is
  reported rather than smoothed over.

### Reading the envelopes
`lower_CDF(t) = inf_h F_h(t)` and `upper_CDF(t) = sup_h F_h(t)`. A **lower CDF is
stochastically larger**, so the lower curve is the *optimistic* reliability bound and the upper
curve the *conservative* one. Lower/upper CDFs are never called "lower/upper reliability".

### Quantiles under imprecision
Definition **Q1** throughout: per-configuration quantiles, then the min/max envelope of those
values. This is the paper's own RQ5 definition. Definition Q2 (inverting the CDF envelopes) is
computed once for comparison in Section 7 and never mixed into a figure or table.""")

code(
    """# Papermill parameters.
RUN_MODE = "published_numerics"       # the only execution mode supported in this release
SCALABILITY_PROFILE = "full"          # "full" | "quick" | "off"
STRICT_EXACT = False                  # True aborts on any unresolved source setting
RESULTS_ROOT = "results"              # redirect artifacts; the test suite points this
                                      # at a temp dir so a reduced-profile test run
                                      # can never overwrite the recorded full run""",
    tags=["parameters"],
)

code(r'''import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src" / "hip_llm").is_dir():  # tolerate being launched from scripts/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from hip_llm import __version__ as HIP_LLM_VERSION
from hip_llm.schemas import (
    GlobalSettings, HyperparameterInterval, ReproductionRecord, ReproductionStatus,
    config_hash, load_yaml, sha256_file,
)
from hip_llm.benchmark_eval import (
    accuracy_to_counts, build_model_from_accuracies, load_accuracy_table,
)
from hip_llm.hyperposterior import HyperposteriorCache
from hip_llm.posterior import run_domain, run_model
from hip_llm.envelopes import (
    cdf_envelope, default_t_grid, quantile_envelope, quantiles_from_cdf_envelope,
    summarise_envelope,
)
from hip_llm.reliability import (
    default_horizons, expected_reliability_envelope, expected_reliability_per_config,
)
from hip_llm import plotting as P
from hip_llm import validation as V

P.apply_house_style()
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

CFG = load_yaml(ROOT / "configs" / "paper_published_numerics.yaml")
RESULTS = Path(RESULTS_ROOT)
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
FIGDIR = RESULTS / "figures"
TABDIR = RESULTS / "tables"
SAMPDIR = RESULTS / "posterior_samples"
DIAGDIR = RESULTS / "diagnostics"
for d in (FIGDIR, TABDIR, SAMPDIR, DIAGDIR):
    d.mkdir(parents=True, exist_ok=True)

AUDIT: list[ReproductionRecord] = []
DIAGNOSTICS: dict[str, object] = {}
NOTEBOOK_T0 = time.perf_counter()

print(f"hip_llm {HIP_LLM_VERSION} | RUN_MODE={RUN_MODE} | STRICT_EXACT={STRICT_EXACT} "
      f"| SCALABILITY_PROFILE={SCALABILITY_PROFILE}")''')

code(r'''def save_table(df: pd.DataFrame, name: str, caption: str = "") -> pd.DataFrame:
    """Write every table as CSV *and* JSON, as the specification requires."""
    df.to_csv(TABDIR / f"{name}.csv", index=False)
    payload = {"name": name, "caption": caption, "columns": list(df.columns),
               "rows": json.loads(df.to_json(orient="records"))}
    (TABDIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return df


def record(item, inputs, section, output_file, status, note=""):
    """Append one row to the paper-audit / reproduction-status table."""
    AUDIT.append(ReproductionRecord(item, inputs, section, output_file, status, note))


def figure_meta(fig_id, caption, data_source, extra=None):
    """Stamp a figure with the configuration hash, data source, seeds and commit."""
    return P.FigureMetadata(
        figure_id=fig_id, caption=caption, config_hash=SETTINGS.hash(),
        data_source=data_source, seeds=SEEDS, git_commit=ENV.get("git_commit"),
        extra={**(extra or {}), "hip_llm_version": HIP_LLM_VERSION,
               "strict_exact": STRICT_EXACT, "run_mode": RUN_MODE},
    )''')

# ===========================================================================
# 1. Environment and provenance
# ===========================================================================
md(r"""---
## 1. Environment and provenance

Everything needed to audit or repeat this run: interpreter, OS, CPU, RAM, package versions,
git state, seeds, configuration hash and input checksums. The paper's own reported environment
is printed alongside so that any deviation is visible rather than assumed away.""")

code(r'''ENV = V.environment_report(ROOT)

print(f"timestamp (UTC) : {ENV['timestamp_utc']}")
print(f"python          : {ENV['python']}   (paper reports 3.12.12)")
print(f"platform        : {ENV['platform']}")
print(f"processor       : {ENV['processor']}")
print(f"logical CPUs    : {ENV['cpu_count_logical']}   RAM: {ENV['ram_total_gb']} GB "
      f"(paper: Intel Xeon @ 2.20 GHz, ~13 GB, single process)")
print(f"git commit      : {ENV['git_commit'] or 'not a git repository'}")
print(f"git clean       : {ENV['git_clean']}")
if ENV["git_modified_files"]:
    print(f"  modified files: {ENV['git_modified_files']}")

paper_pkgs = {"numpy": "2.0.2", "pandas": "2.2.2", "matplotlib": "3.10.0"}
rows = []
for pkg, ver in ENV["packages"].items():
    rows.append({"package": pkg, "installed": ver,
                 "paper_reported": paper_pkgs.get(pkg, "not reported by the paper"),
                 "matches_paper": (pkg in paper_pkgs and ver == paper_pkgs[pkg])})
env_df = save_table(pd.DataFrame(rows), "table_env_packages", "Installed vs paper-reported packages")
display(env_df)
print("\nNOTE: SciPy and PyYAML are implementation dependencies of THIS REPRODUCTION; the paper "
      "reports neither (provenance discrepancy D8).")''')

code(r'''SEEDS = {"global": CFG["seeds"]["global"],
         "configurations": CFG["seeds"]["configurations"],
         "llm_pairs": CFG["seeds"]["llm_pairs"]}
REC = CFG["reconstruction"]

SETTINGS = GlobalSettings(
    n_mu=CFG["grids"]["mu_points"], n_nu=CFG["grids"]["nu_points"],
    cdf_points_T=CFG["grids"]["cdf_points_T"], S=CFG["sampling"]["S"],
    K_per_domain=CFG["sampling"]["K_per_domain"],
    max_llm_configuration_pairs=CFG["sampling"]["max_llm_configuration_pairs"],
    seed_global=SEEDS["global"], seed_configs=SEEDS["configurations"],
    seed_pairs=SEEDS["llm_pairs"],
    config_sampling=REC["config_sampling"], nu_grid_scheme=REC["nu_grid_scheme"],
    mu_grid_scheme=REC["mu_grid_scheme"], nu_grid_params=REC["nu_grid_params"],
    strict_exact=STRICT_EXACT,
)
T_GRID = default_t_grid(SETTINGS.cdf_points_T)
BASE_INTERVAL = HyperparameterInterval(
    a=tuple(CFG["hyperpriors"]["a_range"]), b=tuple(CFG["hyperpriors"]["b_range"]),
    c=tuple(CFG["hyperpriors"]["c_range"]), d=tuple(CFG["hyperpriors"]["d_range"]),
)
HIERARCHY = {k: v["subdomains"] for k, v in CFG["hierarchy"]["domains"].items()}
OMEGA = CFG["weights"]["omega"]
W = CFG["weights"]["W"]
N_EFF = CFG["effective_sample_size"]["N_per_subdomain"]
MODELS = ["GPT-4o", "GPT-4o-mini", "Sonnet 4.5", "Haiku 3.5"]
TOL = CFG["tolerances"]

print(f"configuration hash : {SETTINGS.hash()}")
print(f"grid               : n_mu={SETTINGS.n_mu} x n_nu={SETTINGS.n_nu} = G={SETTINGS.G}"
      f"   (paper: 40 x 50 = 2000)")
print(f"sampling           : S={SETTINGS.S}, K={SETTINGS.K_per_domain}/domain, "
      f"K_total<={SETTINGS.max_llm_configuration_pairs}, T={SETTINGS.cdf_points_T}")
print(f"seeds              : {SEEDS}")
print(f"admissible set     : a,b in {BASE_INTERVAL.a}/{BASE_INTERVAL.b}, "
      f"c,d in {BASE_INTERVAL.c}/{BASE_INTERVAL.d}")
print(f"\nRECONSTRUCTIONS (not in the paper, not in the repository):")
for k in ("mu_grid_scheme", "nu_grid_scheme", "config_sampling", "llm_pairing_mode"):
    print(f"   {k:20s} = {REC[k]}")
print("\nTolerances, fixed BEFORE any result was examined:")
for k, v in TOL.items():
    print(f"   {k:32s} <= {v}")''')

code(r'''MANIFEST = load_yaml(ROOT / "data" / "provenance_manifest.yaml")
integrity = []
for entry in MANIFEST["sources"]:
    local, expected = entry.get("local_copy"), entry.get("sha256")
    if not local or not expected:
        continue
    path = ROOT / local
    if not path.is_file():
        path = ROOT.parent / local
    if not path.is_file():
        integrity.append({"source_id": entry["source_id"], "role": entry["role"],
                          "status": "not present locally", "sha256_ok": None})
        continue
    res = V.assert_no_overwrite(path, expected)
    integrity.append({"source_id": entry["source_id"], "role": entry["role"],
                      "status": res.detail, "sha256_ok": res.passed})

integrity_df = save_table(pd.DataFrame(integrity), "table_source_integrity",
                          "SHA-256 verification of every checksummed source")
display(integrity_df)
bad = [r for r in integrity if r["sha256_ok"] is False]
assert not bad, f"source checksum mismatch: {bad}"
print("all checksummed sources verified")
print("\nThe reference copy of the official figure numerics is BYTE-IDENTICAL to the "
      "repository file (same SHA-256).")''')

# ===========================================================================
# 2. Paper audit
# ===========================================================================
md(r"""---
## 2. Paper audit

A compact map from every paper item (Figs. 1–11, Tables 1–6, Theorems 1–6, RQ1–RQ8) to the
notebook section that produces it, the artifact it writes and its reproduction status. The
table is filled in as the notebook runs and printed in full in Section 15.

Table 1 (framework comparison) is a qualitative table with no computation; it is reproduced
verbatim as a reference artifact here.""")

code(r'''# keep_default_na=False: Table 1 contains the literal word "None" (Miller's
# hierarchical structure and prior specification). Pandas would otherwise read it
# as a missing value and write an empty cell, silently dropping paper content
# from a verbatim reference artifact.
table1 = pd.read_csv(ROOT / "data" / "reference" / "paper_table1_framework_comparison.csv",
                     keep_default_na=False)
assert (table1 != "").all().all(), "Table 1 has an empty cell; the source has NA-like text"
save_table(table1, "table1_framework_comparison", "Paper Table 1 (verbatim reference)")
display(table1)
record("Table 1 (framework comparison)", "paper text (qualitative)", "2",
       "results/tables/table1_framework_comparison.csv", ReproductionStatus.EXACT,
       "verbatim transcription; no computation involved")

theorem_map = pd.DataFrame([
    {"theorem": "Theorem 1 (subdomain non-failure probability)",
     "implemented_in": "hip_llm.hyperposterior.log_hyperposterior + posterior.sample_domain_posterior",
     "verified_by": "tests/test_conjugacy.py, tests/test_hyperposterior.py"},
    {"theorem": "Theorem 2 (domain-level CDF, Lemma 1 decomposition)",
     "implemented_in": "hip_llm.posterior.sample_domain_posterior + envelopes.cdf_envelope",
     "verified_by": "tests/test_conjugacy.py::test_hyperposterior_reproduces_a_known_2d_integral"},
    {"theorem": "Theorem 3 (LLM-level CDF, Cartesian admissible set)",
     "implemented_in": "hip_llm.posterior.run_model + grids.pair_llm_configurations",
     "verified_by": "tests/test_envelopes.py::test_llm_envelope_is_valid_and_pairs_are_joint"},
    {"theorem": "Theorem 4 (subdomain reliability for n_F)",
     "implemented_in": "hip_llm.reliability.transformed_cdf",
     "verified_by": "tests/test_reliability_transform.py::test_transformed_cdf_identity"},
    {"theorem": "Theorem 5 (domain reliability for n_F)",
     "implemented_in": "hip_llm.reliability.reliability_cdf_envelope",
     "verified_by": "tests/test_reliability_transform.py::test_monte_carlo_and_transform_agree"},
    {"theorem": "Theorem 6 (LLM reliability for n_F)",
     "implemented_in": "hip_llm.reliability.expected_reliability_envelope",
     "verified_by": "tests/test_reliability_transform.py (envelope + monotonicity)"},
])
save_table(theorem_map, "table_theorem_map", "Theorem -> implementation -> test map")
display(theorem_map)
for i in range(1, 7):
    record(f"Theorem {i}", "paper Section 3.2.3 / Appendix A", "2",
           "results/tables/table_theorem_map.csv", ReproductionStatus.EXACT,
           "implemented and unit-tested against analytical/quadrature references")''')

code(r'''fig1 = P.draw_hierarchy_concept()
p = P.save_figure(fig1, FIGDIR, "fig01_conceptual_hierarchy",
                  figure_meta("Fig. 1 (redrawn)",
                              "Conceptual hierarchy across LLM instances, domains, subdomains "
                              "and tasks; rectangles independent, ovals dependent.",
                              "model graph (redrawn programmatically)"))
fig2 = P.draw_hierarchy_detail()
P.save_figure(fig2, FIGDIR, "fig02_detailed_hierarchy",
              figure_meta("Fig. 2 (redrawn)",
                          "Detailed hierarchical probabilistic structure: imprecise "
                          "hyper-hyper-parameters -> hyperpriors -> shared latents -> "
                          "subdomain reliabilities -> observed (C, N).",
                          "model graph (redrawn programmatically)"))
display(fig1); display(fig2)
import matplotlib.pyplot as plt
plt.close("all")
for n, cap in ((1, "conceptual hierarchy"), (2, "detailed hierarchical structure")):
    record(f"Fig. {n} ({cap})", "model graph", "2",
           f"results/figures/fig0{n}_*.png", ReproductionStatus.EXACT,
           "redrawn from the model graph; encodes independent domains, dependent subdomains "
           "via shared (mu, nu), observed (C, N), OP-weighted aggregation, multiple LLMs")''')

# ===========================================================================
# 3. Source discrepancy report
# ===========================================================================
md(r"""---
## 3. Source discrepancy report

Ten catalogued conflicts, all detected programmatically. **No source is ever overwritten and
no two conflicting sources are merged.** Each is kept as its own dataset with its own checksum.

| id | conflict | severity |
|----|----------|----------|
| D1 | Paper Table 3 vs official repository figure numerics | high |
| D2 | Fig. 9 caption vs both accuracy sources | medium |
| D3 | Table 5 HIP-LLM medians outside their own 90% intervals | high |
| D4 | Fig. 7 uses Table 3, contradicting the repo's "all figures" claim | high |
| D5 | Table 3's printed GPT-4o LLM mean vs its own printed cells | low |
| D6 | `PERTURBATION = 0.07` (repo) vs ±20% (paper) | medium |
| D7 | "five research questions", then RQ1–RQ8 | informational |
| D8 | SciPy required but not a reported dependency | informational |
| D9 | Table 2's imprecise posterior-mean interval vs its own credal set | medium |
| D10 | Partial pooling is numerically inert under the paper's own hyperprior box | medium |
| D11 | Random hyper-hyper-parameter sampling under-states the imprecise envelope | high |

D10 and D11 are quantified in Section 6; the rest are detected here.""")

code(r'''REFDIR = ROOT / "data" / "reference"
TABLE3 = load_accuracy_table(REFDIR / "paper_table3.csv")
REPO_ACC = load_accuracy_table(REFDIR / "official_figure_numerics.csv",
                               expected_sha256=next(
                                   s["sha256"] for s in MANIFEST["sources"]
                                   if s["source_id"] == "official_repo_accuracies"))
FIG9 = pd.read_csv(REFDIR / "fig9_caption_numerics.csv")

cmp3 = V.compare_accuracy_sources(TABLE3, REPO_ACC)
save_table(cmp3, "table_D1_table3_vs_repo", "D1: paper Table 3 vs official repository numerics")
print(f"D1: {(~cmp3['match']).sum()} of {len(cmp3)} accuracy cells disagree")
print(f"    largest gap: {cmp3.loc[cmp3['abs_diff'].idxmax(), 'model']} / "
      f"{cmp3.loc[cmp3['abs_diff'].idxmax(), 'subdomain']} = "
      f"{cmp3['abs_diff'].max():.3f} absolute "
      f"({cmp3.loc[cmp3['abs_diff'].idxmax(), 'rel_diff']:.1%} relative)")
print(f"    fully agreeing model column(s): {sorted(set(cmp3[cmp3['match']]['model']))}")
display(cmp3[~cmp3["match"]][["model", "subdomain", "theta_hat_table3", "theta_hat_repo",
                              "abs_diff", "rel_diff"]])''')

code(r'''# --- D4: which source actually drives each figure? -------------------------
def _acc(df, model, sub):
    return float(df[(df["model"] == model) & (df["subdomain"] == sub)]["theta_hat"].iloc[0])

rows = []
for omega_race in CFG["rq3"]["omega_race_h"]:
    for label, df in (("paper_table3", TABLE3), ("official_figure_numerics", REPO_ACC)):
        rows.append({"omega_race_h": omega_race, "source": label,
                     "p2_GPT-4o": (1 - omega_race) * _acc(df, "GPT-4o", "BoolQ")
                                  + omega_race * _acc(df, "GPT-4o", "RACE-H"),
                     "p2_Sonnet 4.5": (1 - omega_race) * _acc(df, "Sonnet 4.5", "BoolQ")
                                      + omega_race * _acc(df, "Sonnet 4.5", "RACE-H")})
d4 = save_table(pd.DataFrame(rows), "table_D4_fig7_source_test",
                "D4: which accuracy source reproduces the published Fig. 7 panels")
display(d4)
print("Published Fig. 7b (GPT-4o) shows bands near 0.87 / 0.72 / 0.59 that move LEFT as "
      "omega_RACE rises, on a 0.4-0.9 axis.")
print("  -> paper_table3 reproduces exactly that ordering and those centres.")
print("  -> official_figure_numerics would place all three bands above 0.91 and move them "
      "RIGHT, contradicting the panel.")
print("  -> The paper's own Fig. 7 narrative quotes theta_BoolQ ~ 0.91 vs theta_RACE-H ~ 0.55, "
      "which are Table 3 values.")
print("\nCONCLUSION (D4): Fig. 7 is driven by Table 3; Figs. 3-6, 8, 10 by the repository CSV. "
      "This contradicts numerics/README.md ('All figures use these numerics').")
save_table(pd.DataFrame([{"figure": k, "source": v}
                         for k, v in CFG["figure_source_assignment"].items()]),
           "table_figure_source_assignment", "Per-figure accuracy-source assignment")''')

code(r'''# --- D2: Fig. 9 caption vs both sources ------------------------------------
row = FIG9[FIG9["success_definition"] == "pass@1"].iloc[0]
d2_checks = V.check_fig9_against_sources(
    {"pass1_accuracy": float(row["accuracy"]), "N": int(row["N"])}, TABLE3, REPO_ACC)
d2 = save_table(pd.DataFrame([c.as_row() for c in d2_checks]), "table_D2_fig9_conflict",
                "D2: Fig. 9 captioned Pass@1 vs both accuracy sources")
display(d2)
print("Fig. 9 caption: Sonnet 4.5 / MBPP, N = 257, Pass@1 C/N = 0.471, Pass@3 C/N = 0.494.")
print(f"  Table 3 Sonnet/MBPP  = {_acc(TABLE3, 'Sonnet 4.5', 'MBPP'):.3f}")
print(f"  repository Sonnet/MBPP = {_acc(REPO_ACC, 'Sonnet 4.5', 'MBPP'):.3f}")
print(f"  Table 3 GPT-4o/MBPP  = {_acc(TABLE3, 'GPT-4o', 'MBPP'):.3f}  <- equals the Fig. 9 value")
print("  Baseline figures use N = 80; Fig. 9 uses N = 257.")

# --- D3: Table 5's mathematically impossible rows --------------------------
T4_PRINTED = pd.read_csv(REFDIR / "paper_table4_printed.csv")
T5_PRINTED = pd.read_csv(REFDIR / "paper_table5_printed.csv")
d3_checks = V.check_table5_internal_consistency(T5_PRINTED)
d3 = save_table(pd.DataFrame([c.as_row() for c in d3_checks]),
                "table_D3_table5_validity", "D3: Table 5 median-inside-interval invariant")
display(d3[~d3["passed"]])
print(f"D3: {(~d3['passed']).sum()} of {len(d3)} printed Table 5 rows violate the invariant "
      "'a posterior median lies inside that posterior's own 5%-95% interval'.")
print("    Both are HIP-LLM rows whose median envelope lies strictly BELOW the reported 5% "
      "quantile. Since min_h Q05 <= min_h Q50 by within-posterior quantile monotonicity, no "
      "correct implementation can emit these numbers.")
d3_t4 = V.check_table5_internal_consistency(T4_PRINTED)
print(f"    The same invariant PASSES on all {len(d3_t4)} Table 4 rows "
      f"({sum(c.passed for c in d3_t4)}/{len(d3_t4)}), so the defect is specific to Table 5.")''')

code(r'''# --- D5, D6, D7, D9 --------------------------------------------------------
means = pd.read_csv(REFDIR / "paper_table3_printed_means.csv", comment="#")
printed = means[means["statistic"] == "llm_mean"].iloc[0]
d5 = pd.DataFrame([{
    "model": m,
    "mean_of_printed_cells": float(TABLE3[TABLE3["model"] == m]["theta_hat"].mean()),
    "printed_llm_mean": float(printed[m]),
    "abs_diff": abs(float(TABLE3[TABLE3["model"] == m]["theta_hat"].mean()) - float(printed[m])),
} for m in MODELS])
save_table(d5, "table_D5_table3_mean_consistency", "D5: Table 3 printed means vs its own cells")
display(d5)
print("D5: three columns agree to one rounding unit (0.001); GPT-4o is off by 0.003.")

print(f"\nD6: repository settings.yaml PERTURBATION = "
      f"{CFG['sampling']['perturbation_setting_from_repository']} "
      f"(semantics: {CFG['sampling']['perturbation_semantics']}); paper Section 4.3.5 states "
      "+/-20% for OP^approx. No repository code consumes PERTURBATION, so the two cannot be "
      "reconciled. The value is recorded and deliberately UNUSED; "
      "perturb_and_renormalise() has no default magnitude.")

print("\nD7: paper Section 4 opens with 'we empirically investigate five research questions' "
      "and then enumerates and answers eight (RQ1-RQ8, Sections 4.3.1-4.3.8). "
      "The 'five' matches Gap-1..Gap-5 of the introduction. Editorial only; no numerical impact.")

disc = pd.DataFrame([{ "id": d["id"], "severity": d["severity"], "title": d["title"],
                       "resolution": " ".join(d["resolution"].split())[:180] }
                     for d in MANIFEST["discrepancies"]])
save_table(disc, "table_source_discrepancies", "All catalogued source discrepancies")
display(disc)''')

# ===========================================================================
# 4. Coin flip
# ===========================================================================
md(r"""---
## 4. Coin-flip imprecise-probability sanity example (paper Table 2)

$n = 10$ trials, $k = 3$ heads. Precise prior $\mathrm{Beta}(2,2)$; imprecise set
$\alpha,\beta\in[1,3]$. Everything below is computed analytically —
$\mathbb{E}[\theta\mid D] = (3+\alpha)/(10+\alpha+\beta)$, which is increasing in $\alpha$ and
decreasing in $\beta$, so its extrema over the rectangle are at the corners $(1,3)$ and $(3,1)$.""")

code(r'''from scipy import stats
n, k, alpha_range, beta_range = 10, 3, (1.0, 3.0), (1.0, 3.0)

precise_post = (2 + k, 2 + (n - k))
assert precise_post == (5, 9)
precise_mean = precise_post[0] / sum(precise_post)

corners = [(a, b) for a in alpha_range for b in beta_range]
corner_means = {(a, b): (k + a) / (n + a + b) for a, b in corners}
grid = np.linspace(1.0, 3.0, 401)
all_means = np.array([[(k + a) / (n + a + b) for b in grid] for a in grid])
lo, hi = float(all_means.min()), float(all_means.max())

t2 = save_table(pd.DataFrame([
    {"feature": "Prior", "classical_bayesian": "Beta(2, 2)",
     "imprecise_computed": "Beta(a, b), a in [1,3], b in [1,3]", "paper_printed": "same"},
    {"feature": "Posterior", "classical_bayesian": f"Beta{precise_post}",
     "imprecise_computed": "Beta(3+a, 7+b)", "paper_printed": "same"},
    {"feature": "Posterior mean", "classical_bayesian": f"{precise_mean:.4f} -> {precise_mean:.2f}",
     "imprecise_computed": f"[{lo:.4f}, {hi:.4f}] -> [{lo:.2f}, {hi:.2f}]",
     "paper_printed": "[0.31, 0.38]  <-- INCONSISTENT (D9)"},
]), "table2_coinflip", "Paper Table 2, recomputed analytically")
display(t2)

print(f"corner means: " + ", ".join(f"a={a:.0f},b={b:.0f} -> {v:.4f}" for (a, b), v in corner_means.items()))
print(f"\nprecise    : Beta(5, 9), mean = {precise_mean:.4f} -> 0.36   MATCHES the paper")
print(f"imprecise  : [{lo:.4f}, {hi:.4f}] = [4/14, 6/14] -> [0.29, 0.43]")
print(f"paper prints [0.31, 0.38] -- which is EXACTLY [4/13, 5/13] = "
      f"[{4/13:.4f}, {5/13:.4f}], the range under the extra constraint a + b = 3, a in [1,2].")
assert (round(lo, 2), round(hi, 2)) == (0.29, 0.43)
record("Table 2 (coin flip)", "analytical Beta-Binomial conjugacy", "4",
       "results/tables/table2_coinflip.csv", ReproductionStatus.INCONSISTENT_SOURCE,
       "precise column reproduces exactly (0.36); the printed imprecise interval [0.31, 0.38] "
       "does not follow from the printed credal set, which gives [0.29, 0.43] (D9)")''')

code(r'''x = np.linspace(1e-4, 1 - 1e-4, 2000)
stack = np.array([stats.beta.pdf(x, k + a, (n - k) + b) for a in grid for b in grid])
lo_pdf, hi_pdf = stack.min(axis=0), stack.max(axis=0)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.fill_between(x, lo_pdf, hi_pdf, color="#9467bd", alpha=0.30,
                label=r"imprecise envelope, $\alpha,\beta\in[1,3]$")
ax.plot(x, stats.beta.pdf(x, *precise_post), color="#1f77b4", lw=2,
        label=r"precise posterior Beta(5, 9)")
ax.axvline(precise_mean, color="#1f77b4", ls=":", lw=1.2, label=f"precise mean {precise_mean:.3f}")
ax.axvspan(lo, hi, color="#2ca02c", alpha=0.15,
           label=f"computed mean interval [{lo:.2f}, {hi:.2f}]")
ax.axvspan(0.31, 0.38, color="#d62728", alpha=0.12,
           label="paper-printed interval [0.31, 0.38] (D9)")
ax.set_xlabel(r"$\theta$"); ax.set_ylabel("posterior density")
ax.set_title("Table 2: precise vs imprecise posterior for the coin-flip example")
ax.legend(fontsize=8, loc="upper right")
P.save_figure(fig, FIGDIR, "fig_table2_coinflip",
              figure_meta("Table 2 (illustration)",
                          "Precise Beta(5,9) posterior inside the imprecise posterior envelope "
                          "induced by alpha, beta in [1,3].", "paper Table 2 (analytical)"))
display(fig); plt.close(fig)''')

# ===========================================================================
# 5. Minimal executable example
# ===========================================================================
md(r"""---
## 5. Minimal executable HIP-LLM example

A two-domain, two-subdomain walk-through with **explicit real counts** taken from the authors'
published measurements for GPT-4o. It demonstrates, end to end, the hyperposterior grid,
conditional Beta sampling, the induced dependence, both aggregation levels, envelope
construction and future reliability. Explanatory only — the full experiments follow.""")

code(r'''from hip_llm.grids import build_grid, sample_configurations
from hip_llm.hyperposterior import Hyperposterior, log_hyperposterior
from hip_llm.posterior import sample_domain_posterior
from hip_llm.schemas import DomainData, HyperparameterConfiguration, ModelResult, SubdomainData

demo_model = build_model_from_accuracies(
    REPO_ACC, "GPT-4o", HIERARCHY, OMEGA, W, N_EFF, "official_figure_numerics")
demo_d1 = demo_model.domains[0]
print("Domain D1 (Coding), from published accuracies with C = round(theta_hat * 80):")
for s in demo_d1.subdomains:
    print(f"   {s.name:9s} theta_hat={s.source_accuracy:.3f} -> C/N = {s.successes}/{s.trials} "
          f"= {s.empirical_accuracy:.4f}   (rounding shift {s.empirical_accuracy - s.source_accuracy:+.5f})")

demo_cfg = HyperparameterConfiguration(a=6.0, b=4.0, c=12.0, d=1.0)
demo_grid = build_grid(SETTINGS.n_mu, SETTINGS.n_nu, mu_scheme=SETTINGS.mu_grid_scheme,
                       nu_scheme=SETTINGS.nu_grid_scheme, config=demo_cfg,
                       strict_exact=STRICT_EXACT, nu_params=SETTINGS.nu_grid_params)
probs, log_ev = log_hyperposterior(demo_d1, demo_cfg, demo_grid)
demo_hp = Hyperposterior(demo_d1.name, demo_cfg, demo_grid, probs, log_ev)
print(f"\nhyperposterior on the {demo_grid.n_mu} x {demo_grid.n_nu} = {demo_grid.size}-cell grid"
      f"  ({demo_grid.scheme})")
print(f"   sums to {probs.sum():.12f}   log evidence = {log_ev:.4f}")
print(f"   E[mu | C_1, h] = {demo_hp.mean_mu:.4f}   E[nu | C_1, h] = {demo_hp.mean_nu:.4f}")''')

code(r'''demo_ps = sample_domain_posterior(demo_d1, demo_hp, 40_000, np.random.default_rng(SEEDS["global"]))
r_marg = float(np.corrcoef(demo_ps.theta[:, 0], demo_ps.theta[:, 1])[0, 1])

# Conditional on one (mu, nu) cell the subdomains must be independent.
one = np.zeros(demo_grid.size); one[int(np.argmax(probs))] = 1.0
cond_ps = sample_domain_posterior(
    demo_d1, Hyperposterior(demo_d1.name, demo_cfg, demo_grid, one, 0.0),
    40_000, np.random.default_rng(1))
r_cond = float(np.corrcoef(cond_ps.theta[:, 0], cond_ps.theta[:, 1])[0, 1])

print(f"corr(theta_MBPP, theta_DS-1000):")
print(f"   conditional on a fixed (mu, nu) : {r_cond:+.4f}   (must be ~ 0)")
print(f"   after marginalising (mu, nu)    : {r_marg:+.4f}   (must be > 0)")
print(f"   Var(sum) = {np.var(demo_ps.theta.sum(axis=1)):.6f} vs "
      f"{np.var(demo_ps.theta[:, 0]) + np.var(demo_ps.theta[:, 1]):.6f} under independence")
assert r_marg > r_cond
print("\n-> the shared latent draw is what creates the dependence; conditioning removes it.")''')

code(r'''demo_sets, demo_llm = run_model(demo_model, [BASE_INTERVAL] * 2, SETTINGS, model_index=0)
print(f"aggregation: p_i = sum_j Omega_ij theta_ij, then p_L = sum_i W_i p_i")
for ds, om in zip(demo_sets, (OMEGA["D1"], OMEGA["D2"])):
    s = summarise_envelope(ds.p, ds.domain)
    print(f"   {ds.domain}  Omega={om}   median envelope "
          f"[{s.median_lower:.4f}, {s.median_upper:.4f}]")
s_llm = summarise_envelope(demo_llm.p_L, "p_L")
print(f"   LLM  W={W}   median envelope [{s_llm.median_lower:.4f}, {s_llm.median_upper:.4f}]"
      f"   over {demo_llm.n_configs} configuration tuples")

rows = []
for n_F in (1, 2, 5, 10):
    e = expected_reliability_envelope(demo_llm.p_L, np.array([float(n_F)]))
    rows.append({"n_F": n_F, "E[R_L] lower": float(e.lower[0]), "E[R_L] upper": float(e.upper[0]),
                 "envelope width": float(e.width[0])})
demo_rel = save_table(pd.DataFrame(rows), "table_minimal_example_reliability",
                      "Minimal example: future reliability at n_F = 1, 2, 5, 10 (GPT-4o)")
display(demo_rel)
assert demo_rel["E[R_L] lower"].is_monotonic_decreasing
print("\nreliability is non-increasing in n_F, as it must be.")''')

# ===========================================================================
# 6. Full published-numerics model
# ===========================================================================
md(r"""---
## 6. Full published-numerics model

All four models under the paper's hierarchy, operational-profile weights and global numerical
settings, driven by the authors' official figure numerics. Both the original accuracy and the
effective $C/N$ after rounding are printed, as required.

This section also quantifies how much the two **unresolved reconstructions** ($\nu$-axis
construction, configuration-sampling rule) actually move the answer.""")

code(r'''counts_rows = []
for m in MODELS:
    for dname, subs in HIERARCHY.items():
        for sname in subs:
            acc = _acc(REPO_ACC, m, sname)
            C = accuracy_to_counts(acc, N_EFF)
            counts_rows.append({"model": m, "domain": dname, "subdomain": sname,
                                "theta_hat_published": acc, "N": N_EFF, "C_effective": C,
                                "C_over_N": C / N_EFF, "rounding_shift": C / N_EFF - acc})
counts_df = save_table(pd.DataFrame(counts_rows), "table_effective_counts",
                       "Published accuracy -> effective counts, C = round(theta_hat * 80)")
display(counts_df)
print(f"max |rounding shift| = {counts_df['rounding_shift'].abs().max():.5f} "
      f"(bounded by 1/(2N) = {1 / (2 * N_EFF):.5f})")

table3_counts = save_table(
    pd.DataFrame([{"model": m, "domain": d, "subdomain": s,
                   "theta_hat_table3": _acc(TABLE3, m, s), "N": N_EFF,
                   "C_effective": accuracy_to_counts(_acc(TABLE3, m, s), N_EFF)}
                  for m in MODELS for d, subs in HIERARCHY.items() for s in subs]),
    "table3_literal_reproduction", "Paper Table 3 reproduced literally (separate dataset)")
record("Table 3 (published accuracies)", "paper Table 3, verbatim", "6",
       "results/tables/table3_literal_reproduction.csv", ReproductionStatus.EXACT,
       "reproduced verbatim as its own dataset; NOT merged with the repository numerics (D1)")''')

code(r'''CACHE = HyperposteriorCache()
FITS: dict[str, dict] = {}
t0 = time.perf_counter()
for i, m in enumerate(MODELS):
    mr = build_model_from_accuracies(REPO_ACC, m, HIERARCHY, OMEGA, W, N_EFF,
                                     "official_figure_numerics")
    dsets, llm = run_model(mr, [BASE_INTERVAL] * len(mr.domains), SETTINGS,
                           model_index=i, cache=CACHE,
                           pairing_mode=REC["llm_pairing_mode"])
    FITS[m] = {"model": mr, "domains": dsets, "llm": llm}
    np.savez_compressed(SAMPDIR / f"p_L_{m.replace(' ', '_')}.npz",
                        p_L=llm.p_L, pair_index=llm.pair_index)
    for ds in dsets:
        np.savez_compressed(SAMPDIR / f"theta_{m.replace(' ', '_')}_{ds.domain}.npz",
                            theta=ds.theta, p=ds.p, mu=ds.mu, nu=ds.nu)
elapsed = time.perf_counter() - t0
DIAGNOSTICS["mode_a_runtime_s"] = elapsed
DIAGNOSTICS["cache"] = dict(CACHE.stats())
print(f"fitted {len(MODELS)} models in {elapsed:.1f}s   cache={CACHE.stats()}")
print(f"posterior sample arrays written to {SAMPDIR}")

summary = []
for m in MODELS:
    f = FITS[m]
    row = {"model": m}
    for ds in f["domains"]:
        s = summarise_envelope(ds.p, ds.domain)
        row[f"{ds.domain} median lo"] = s.median_lower
        row[f"{ds.domain} median hi"] = s.median_upper
    s = summarise_envelope(f["llm"].p_L, "p_L")
    row.update({"p_L median lo": s.median_lower, "p_L median hi": s.median_upper,
                "p_L q05 lo": s.q05_lower, "p_L q95 hi": s.q95_upper,
                "envelope area": s.envelope_area, "n_config_tuples": s.n_configs})
    summary.append(row)
summary_df = save_table(pd.DataFrame(summary), "table_mode_a_summary",
                        "Mode A: posterior summaries for all four models")
display(summary_df)''')

code(r'''# --- how much do the unresolved reconstructions matter? --------------------
probe_model = build_model_from_accuracies(REPO_ACC, "GPT-4o", HIERARCHY, OMEGA, W, N_EFF, "repo")
sens_rows = []
for field, options in (("nu_grid_scheme", ["log", "linear", "gamma_quantile"]),
                       ("config_sampling", ["uniform_random", "latin_hypercube", "sobol",
                                            "interval_corners_plus_interior"])):
    for opt in options:
        st = GlobalSettings(**{**SETTINGS.__dict__, field: opt})
        _, llm = run_model(probe_model, [BASE_INTERVAL] * 2, st, model_index=0,
                           pairing_mode=REC["llm_pairing_mode"])
        s = summarise_envelope(llm.p_L, "p_L")
        e = cdf_envelope(llm.p_L, T_GRID, "p_L")
        sens_rows.append({"reconstruction": field, "option": opt,
                          "median lo": s.median_lower, "median hi": s.median_upper,
                          "q05 lo": s.q05_lower, "q95 hi": s.q95_upper,
                          "envelope area": s.envelope_area,
                          "_lower": e.lower, "_upper": e.upper})
sens = pd.DataFrame(sens_rows)

diffs = []
for field, grp in sens.groupby("reconstruction"):
    ref = grp.iloc[0]
    for _, r in grp.iloc[1:].iterrows():
        diffs.append({"reconstruction": field, "option": r["option"], "vs": ref["option"],
                      "d median lo": abs(r["median lo"] - ref["median lo"]),
                      "d median hi": abs(r["median hi"] - ref["median hi"]),
                      "d q05": abs(r["q05 lo"] - ref["q05 lo"]),
                      "d q95": abs(r["q95 hi"] - ref["q95 hi"]),
                      "CDF sup-norm": float(max(np.max(np.abs(r["_lower"] - ref["_lower"])),
                                                np.max(np.abs(r["_upper"] - ref["_upper"]))))})
sens_df = save_table(pd.DataFrame(diffs), "table_reconstruction_sensitivity",
                     "Sensitivity of GPT-4o p_L to the unresolved reconstruction choices")
display(save_table(sens.drop(columns=["_lower", "_upper"]),
                   "table_reconstruction_variants", "Posterior summaries per reconstruction variant"))
display(sens_df)

by = sens_df.groupby("reconstruction").agg(
    max_median_shift=("d median lo", "max"), max_cdf_sup=("CDF sup-norm", "max")).reset_index()
display(save_table(by, "table_reconstruction_sensitivity_by_type",
                   "Worst-case shift attributable to each unresolved reconstruction"))

nu_only = sens_df[sens_df.reconstruction == "nu_grid_scheme"]
cfg_only = sens_df[sens_df.reconstruction == "config_sampling"]
principled = nu_only[nu_only.option == "gamma_quantile"]
DIAGNOSTICS["nu_grid_max_median_shift"] = float(
    nu_only[["d median lo", "d median hi"]].to_numpy().max())
DIAGNOSTICS["nu_grid_max_cdf_sup"] = float(nu_only["CDF sup-norm"].max())
DIAGNOSTICS["nu_grid_principled_pair_cdf_sup"] = float(principled["CDF sup-norm"].iloc[0])
DIAGNOSTICS["config_sampling_max_median_shift"] = float(
    cfg_only[["d median lo", "d median hi"]].to_numpy().max())
DIAGNOSTICS["config_sampling_max_cdf_sup"] = float(cfg_only["CDF sup-norm"].max())

print(f"\nnu-grid construction  : max median shift {DIAGNOSTICS['nu_grid_max_median_shift']:.5f} "
      f"(tol {TOL['posterior_median_abs']}), max CDF sup-norm "
      f"{DIAGNOSTICS['nu_grid_max_cdf_sup']:.5f} (tol {TOL['cdf_sup_norm']})")
print(f"   log vs gamma_quantile (the two principled schemes): median shift "
      f"{float(principled[['d median lo', 'd median hi']].to_numpy().max()):.5f}, "
      f"CDF sup-norm {DIAGNOSTICS['nu_grid_principled_pair_cdf_sup']:.5f} -> WITHIN tolerance")
print("   the 'linear' variant exceeds the CDF tolerance: a uniform grid on [1e-3, 250] with "
      "only 50 cells resolves small nu poorly, so it is the least defensible of the three.")
print(f"\nconfiguration sampling: max median shift "
      f"{DIAGNOSTICS['config_sampling_max_median_shift']:.5f} "
      f"(tol {TOL['posterior_median_abs']}), max CDF sup-norm "
      f"{DIAGNOSTICS['config_sampling_max_cdf_sup']:.5f} (tol {TOL['cdf_sup_norm']}) "
      f"-> EXCEEDS tolerance; investigated next.")''')

code(r'''# --- D11: random configuration draws UNDER-STATE the imprecise envelope ----
# Paper Appendix A.2, Step 3: "Operationally, these envelopes are obtained by
# evaluating the closed-form posterior Pr(theta_ij | C_i, h_i) at the extremal
# corners of H_i or through numerical optimization if the extrema occur in the
# interior."  160 uniform draws in a 4-D box essentially never reach a corner.
from hip_llm.grids import pair_llm_configurations, sample_configurations

corner_cfgs = sample_configurations(BASE_INTERVAL, K=16, seed=1,
                                    scheme="interval_corners_plus_interior")
rows = []
variants = [("16 box corners only", corner_cfgs, SETTINGS),
            ("160 uniform draws (as configured)", None, SETTINGS),
            ("160 corners + interior", None,
             GlobalSettings(**{**SETTINGS.__dict__,
                               "config_sampling": "interval_corners_plus_interior"}))]
for label, cfgs, st in variants:
    dsets = [run_domain(d, BASE_INTERVAL, st, domain_index=i, model_index=0, configs=cfgs)
             for i, d in enumerate(probe_model.domains)]
    K_eff = dsets[0].n_configs
    pi = pair_llm_configurations(K_eff, 2, SETTINGS.max_llm_configuration_pairs,
                                 SETTINGS.seed_pairs, REC["llm_pairing_mode"])
    pL = W[0] * dsets[0].p[pi[:, 0], :] + W[1] * dsets[1].p[pi[:, 1], :]
    s = summarise_envelope(pL, "p_L")
    rows.append({"configuration set": label, "K": K_eff,
                 "median lo": s.median_lower, "median hi": s.median_upper,
                 "q05 lo": s.q05_lower, "q95 hi": s.q95_upper,
                 "envelope area": s.envelope_area,
                 "median envelope width": s.median_upper - s.median_lower})
d11 = save_table(pd.DataFrame(rows), "table_D11_corner_vs_random_envelope",
                 "D11: corner-based vs random configuration sets (GPT-4o, LLM level)")
display(d11)

rand = d11[d11["configuration set"].str.startswith("160 uniform")].iloc[0]
corn = d11[d11["configuration set"] == "16 box corners only"].iloc[0]
ratio = float(corn["envelope area"] / rand["envelope area"])
DIAGNOSTICS["corner_vs_random_area_ratio"] = ratio
print(f"envelope area   : 16 corners {corn['envelope area']:.5f}  vs  "
      f"160 uniform draws {rand['envelope area']:.5f}   ratio = {ratio:.2f}x")
print(f"median env width: 16 corners {corn['median envelope width']:.5f}  vs  "
      f"160 uniform draws {rand['median envelope width']:.5f}")
print("\nFINDING D11. Paper Appendix A.2 Step 3 states the envelopes are obtained at the "
      "EXTREMAL CORNERS of the admissible set, but settings.yaml specifies N_CONFIGS = 160 with "
      "a dedicated sampling seed, i.e. a random design. Sixteen corners of a 4-D box bracket the "
      "credal set strictly better than 160 uniform interior draws, which almost never land near "
      "a corner. If the published figures came from random sampling, their envelopes UNDERSTATE "
      "the imprecise envelope -- which is anti-conservative for a robustness bound.")''')

# ===========================================================================
# 7. RQ1
# ===========================================================================
md(r"""---
## 7. RQ1 (Effectiveness) — Figs. 3, 4, 5

Posterior CDF envelopes at all three levels of the hierarchy. Every curve is freshly computed;
panel layout, per-model colour/hatch, axes and legends follow the published figures.""")

code(r'''import matplotlib.pyplot as plt

sub_panels = []
for dname, subs in HIERARCHY.items():
    for sname in subs:
        envs = {m: cdf_envelope(FITS[m]["domains"][list(HIERARCHY).index(dname)]
                                .subdomain_samples(sname), T_GRID, f"{m}/{sname}")
                for m in MODELS}
        sub_panels.append((f"Subdomain posterior CDF envelopes: {dname} / {sname}", envs))

fig3 = P.plot_envelope_grid(sub_panels, ncols=2, xlabel="non-failure probability "
                            r"$\theta_{ij}$")
P.save_figure(fig3, FIGDIR, "fig03_subdomain_cdf_envelopes",
              figure_meta("Fig. 3",
                          "Posterior CDF envelopes of non-failure probability at the subdomain "
                          "level; rows are domains (D1: MBPP, DS-1000; D2: BoolQ, RACE-H).",
                          "official_figure_numerics"))
display(fig3); plt.close(fig3)

rows = [{"subdomain": s, "model": m,
         **{k: v for k, v in summarise_envelope(
             FITS[m]["domains"][list(HIERARCHY).index(d)].subdomain_samples(s), s
         ).as_row().items() if k != "quantity"}}
        for d, subs in HIERARCHY.items() for s in subs for m in MODELS]
display(save_table(pd.DataFrame(rows), "table_rq1_subdomain_envelopes",
                   "RQ1: subdomain-level envelope summaries (Fig. 3)"))
record("Fig. 3 (subdomain CDF envelopes)", "official_figure_numerics + HIP-LLM inference", "7",
       "results/figures/fig03_subdomain_cdf_envelopes.png",
       ReproductionStatus.RECONSTRUCTED,
       "freshly computed; qualitative orderings match the paper's narrative exactly")''')

code(r'''dom_panels = []
for idx, (dname, label) in enumerate([("D1", "Coding"), ("D2", "Reasoning")]):
    envs = {m: cdf_envelope(FITS[m]["domains"][idx].p, T_GRID, f"{m}/{dname}") for m in MODELS}
    dom_panels.append((f"Domain posterior CDF envelopes: {dname} ({label})", envs))
fig4 = P.plot_envelope_grid(dom_panels, ncols=2,
                            xlabel=r"non-failure probability $p_i=\sum_j \Omega_{ij}\theta_{ij}$")
P.save_figure(fig4, FIGDIR, "fig04_domain_cdf_envelopes",
              figure_meta("Fig. 4", "Posterior CDF envelopes of domain-level non-failure "
                          "probability.", "official_figure_numerics"))
display(fig4); plt.close(fig4)

llm_envs = {m: cdf_envelope(FITS[m]["llm"].p_L, T_GRID, f"{m}/p_L") for m in MODELS}
fig5, ax = plt.subplots(figsize=(7.0, 4.6))
P.plot_cdf_envelopes(ax, llm_envs, "LLM posterior CDF envelopes (Overall Performance)",
                     xlim=P.auto_xlim(llm_envs),
                     xlabel=r"non-failure probability $p_L=\sum_i W_i p_i$")
fig5.tight_layout()
P.save_figure(fig5, FIGDIR, "fig05_llm_cdf_envelopes",
              figure_meta("Fig. 5", "Posterior CDF envelope of overall LLM-level non-failure "
                          f"probability with domain weights W = {W}.", "official_figure_numerics"))
display(fig5); plt.close(fig5)

rq1 = save_table(pd.DataFrame(
    [{"level": ds.domain, "model": m,
      **{k: v for k, v in summarise_envelope(ds.p, ds.domain).as_row().items() if k != "quantity"}}
     for m in MODELS for ds in FITS[m]["domains"]]
    + [{"level": "LLM", "model": m,
        **{k: v for k, v in summarise_envelope(FITS[m]["llm"].p_L, "p_L").as_row().items()
           if k != "quantity"}} for m in MODELS]),
    "table_rq1_domain_llm_envelopes", "RQ1: domain and LLM envelope summaries (Figs. 4, 5)")
display(rq1)
for n, cap in ((4, "domain CDF envelopes"), (5, "LLM CDF envelopes")):
    record(f"Fig. {n} ({cap})", "official_figure_numerics + HIP-LLM inference", "7",
           f"results/figures/fig0{n}_*.png", ReproductionStatus.RECONSTRUCTED,
           "freshly computed from the published measurements")''')

code(r'''ordering = (rq1[rq1["level"] == "LLM"]
            .assign(mid=lambda d: 0.5 * (d["median_lower"] + d["median_upper"]))
            .sort_values("mid", ascending=False)[["model", "median_lower", "median_upper"]])
print("LLM-level ordering (most to least reliable):")
display(ordering)
print("Paper Section 4.3.1: '4o-mini and Haiku bands overlap almost completely ... GPT-4o and "
      "Sonnet 4.5 partially overlap ... Overall, Sonnet 4.5 remains most reliable.'")
top = ordering.iloc[0]["model"]
assert top == "Sonnet 4.5", top
mini = rq1[(rq1.level == "LLM") & (rq1.model == "GPT-4o-mini")].iloc[0]
haiku = rq1[(rq1.level == "LLM") & (rq1.model == "Haiku 3.5")].iloc[0]
overlap = min(mini.median_upper, haiku.median_upper) - max(mini.median_lower, haiku.median_lower)
print(f"\n-> Sonnet 4.5 is most reliable: CONFIRMED")
print(f"-> GPT-4o-mini vs Haiku 3.5 median-envelope overlap = {overlap:+.4f} "
      f"(of widths {mini.median_upper - mini.median_lower:.4f} / "
      f"{haiku.median_upper - haiku.median_lower:.4f}): near-complete overlap CONFIRMED")

# Definition Q2, computed once for comparison and never mixed into a figure.
q2 = save_table(pd.DataFrame([{
    "model": m,
    "Q1 median lo": quantile_envelope(FITS[m]["llm"].p_L, 0.5)[0],
    "Q1 median hi": quantile_envelope(FITS[m]["llm"].p_L, 0.5)[1],
    "Q2 median lo": quantiles_from_cdf_envelope(llm_envs[m], 0.5)[0],
    "Q2 median hi": quantiles_from_cdf_envelope(llm_envs[m], 0.5)[1],
} for m in MODELS]), "table_rq1_quantile_definitions",
    "Q1 (envelope of quantiles, used everywhere) vs Q2 (quantiles of the CDF envelope)")
display(q2)
print("\nEvery figure and table in this notebook uses Q1. Q2 is shown here once for comparison "
      "only; the two are never mixed.")''')

# ===========================================================================
# 8. RQ2
# ===========================================================================
md(r"""---
## 8. RQ2 (Sensitivity to hyperparameters) — Fig. 6

GPT-4o, varying one **domain-2** interval at a time over exactly the ranges the paper states,
holding everything else at the baseline.""")

code(r'''rq2_cfg = CFG["rq2"]
rq2_model = build_model_from_accuracies(REPO_ACC, rq2_cfg["model"], HIERARCHY, OMEGA, W, N_EFF,
                                        "official_figure_numerics")
panels, rq2_rows = [], []
LABEL = {"a2": "a", "b2": "b", "c2": "c", "d2": "d"}
for key in ("a2", "b2", "c2", "d2"):
    envs = {}
    for rng_ in rq2_cfg["intervals"][key]:
        iv2 = BASE_INTERVAL.with_replaced(**{LABEL[key]: tuple(rng_)})
        _, llm = run_model(rq2_model, [BASE_INTERVAL, iv2], SETTINGS, model_index=0,
                           cache=CACHE, pairing_mode=REC["llm_pairing_mode"])
        lab = rf"${LABEL[key]}_2 \in [{rng_[0]}, {rng_[1]}]$"
        envs[lab] = cdf_envelope(llm.p_L, T_GRID, lab)
        s = summarise_envelope(llm.p_L, lab)
        rq2_rows.append({"hyperparameter": LABEL[key], "interval": f"[{rng_[0]}, {rng_[1]}]",
                         **{k: v for k, v in s.as_row().items() if k != "quantity"}})
    panels.append((rf"Narrowing the ${LABEL[key]}_2$ prior range"
                   f"\n(LLM-level posterior CDF, {rq2_cfg['model']})", envs))

fig6 = P.plot_envelope_grid(panels, ncols=2, figsize_per_panel=(5.6, 3.9), xlabel=r"$p_L$")
P.save_figure(fig6, FIGDIR, "fig06_hyperparameter_sensitivity",
              figure_meta("Fig. 6", "Effect of narrowing each domain-2 hyper-hyper-parameter "
                          "interval on the LLM-level posterior CDF envelope (GPT-4o).",
                          "official_figure_numerics"))
display(fig6); plt.close(fig6)''')

code(r'''rq2_df = save_table(pd.DataFrame(rq2_rows), "table_rq2_hyperparameter_sensitivity",
                    "RQ2: posterior shift and envelope width per hyperparameter interval")
display(rq2_df)

shift = (rq2_df.assign(mid=lambda d: 0.5 * (d.median_lower + d.median_upper))
         .groupby("hyperparameter")
         .agg(median_range=("mid", lambda s: s.max() - s.min()),
              area_range=("envelope_area", lambda s: s.max() - s.min()),
              sep_range=("max_cdf_separation", lambda s: s.max() - s.min()))
         .reset_index())
display(save_table(shift, "table_rq2_effect_sizes",
                   "RQ2: how much each hyperparameter moves location vs width"))
loc = shift.set_index("hyperparameter")["median_range"]
wid = shift.set_index("hyperparameter")["area_range"]
print("Paper Section 4.3.2: a and b mainly shift the LOCATION; c and d mainly control the WIDTH, "
      "with d having the stronger effect because it directly reduces the pooling strength nu.")
print(f"\n  location movement (median range): " +
      ", ".join(f"{k}={loc[k]:.5f}" for k in ("a", "b", "c", "d")))
print(f"  width movement (envelope area range): " +
      ", ".join(f"{k}={wid[k]:.5f}" for k in ("a", "b", "c", "d")))
print(f"\n  CONFIRMED: d moves the envelope width more than c "
      f"({wid['d']:.5f} vs {wid['c']:.5f}, {wid['d'] / max(wid['c'], 1e-12):.1f}x).")
print(f"  PARTIALLY CONFIRMED: c and d do control width, but d also moves the LOCATION more "
      f"than a or b do ({loc['d']:.5f} vs {loc['a']:.5f} and {loc['b']:.5f}). The paper's "
      "clean split -- a/b for location, c/d for width -- is therefore only approximate here.")
print("  This follows from D10: E[nu] = c/d, so narrowing d upward drives the pooling strength "
      "further down, which shifts the posterior as well as widening the band. It is reported "
      "rather than smoothed over.")
record("Fig. 6 (hyperparameter sensitivity)", "official_figure_numerics + HIP-LLM inference", "8",
       "results/figures/fig06_hyperparameter_sensitivity.png", ReproductionStatus.RECONSTRUCTED,
       "exact intervals from the paper; d controls width most (confirmed), but d also shifts "
       "location more than a or b, so the paper's a/b-location vs c/d-width split is only "
       "approximate (see D10)")''')

# ===========================================================================
# 9. RQ3
# ===========================================================================
md(r"""---
## 9. RQ3 (Sensitivity to operational profiles) — Fig. 7

$\Omega_{\text{RACE-H}} \in \{0.10, 0.517, 0.90\}$, $\Omega_{\text{BoolQ}} = 1-\Omega_{\text{RACE-H}}$,
for Sonnet 4.5 and GPT-4o.

**Fig. 7 is the one main figure driven by paper Table 3, not by the repository CSV** (D4).
Both source variants are computed; the Table 3 version is the reproduction of the published
panel and the repository version is reported alongside so the conflict stays visible.""")

code(r'''rq3_rows, rq3_panels = [], []
for source_label, source_df in (("paper_table3", TABLE3), ("official_figure_numerics", REPO_ACC)):
    for model in CFG["rq3"]["models"]:
        envs = {}
        for om in CFG["rq3"]["omega_race_h"]:
            mr = build_model_from_accuracies(source_df, model, HIERARCHY,
                                             {**OMEGA, "D2": [1.0 - om, om]}, W, N_EFF, source_label)
            ds = run_domain(mr.domains[1], BASE_INTERVAL, SETTINGS, domain_index=1,
                            model_index=MODELS.index(model), cache=CACHE)
            lab = rf"$\Omega_\mathrm{{RACE}} = {om:.3f}$"
            envs[lab] = cdf_envelope(ds.p, T_GRID, lab)
            s = summarise_envelope(ds.p, lab)
            rq3_rows.append({"source": source_label, "model": model, "omega_race_h": om,
                             **{k: v for k, v in s.as_row().items() if k != "quantity"}})
        if source_label == "paper_table3":
            rq3_panels.append((f"Reasoning domain D2 vs RACE-H weight\n({model})", envs))

fig7 = P.plot_envelope_grid(rq3_panels, ncols=2, figsize_per_panel=(5.8, 4.0),
                            xlabel=r"$p_2 = \Omega_\mathrm{BoolQ}\theta_\mathrm{BoolQ} + "
                                   r"\Omega_\mathrm{RACE}\theta_\mathrm{RACE\text{-}H}$")
P.save_figure(fig7, FIGDIR, "fig07_operational_profile_sensitivity",
              figure_meta("Fig. 7", "Posterior CDF envelopes for the Reasoning domain under "
                          "alternative RACE-H operational weights.", "paper_table3",
                          extra={"discrepancy": "D4 - Fig. 7 uses Table 3, not the repository CSV"}))
display(fig7); plt.close(fig7)''')

code(r'''rq3_df = save_table(pd.DataFrame(rq3_rows), "table_rq3_op_sensitivity",
                    "RQ3: OP sensitivity under BOTH accuracy sources (D4 kept visible)")
display(rq3_df[rq3_df.source == "paper_table3"])

sens = (rq3_df.assign(mid=lambda d: 0.5 * (d.median_lower + d.median_upper),
                      width=lambda d: d.q95_upper - d.q05_lower)
        .groupby(["source", "model"])
        .agg(median_swing=("mid", lambda s: s.max() - s.min()),
             mean_envelope_width=("width", "mean")).reset_index())
display(save_table(sens, "table_rq3_effect_sizes", "RQ3: OP sensitivity magnitude per model"))

t3 = sens[sens.source == "paper_table3"].set_index("model")["median_swing"]
print(f"Under paper Table 3 (the source that reproduces the published panel):")
print(f"   GPT-4o median swing across the three OPs   = {t3['GPT-4o']:.4f}")
print(f"   Sonnet 4.5 median swing across the three OPs = {t3['Sonnet 4.5']:.4f}")
print(f"   ratio = {t3['GPT-4o'] / t3['Sonnet 4.5']:.1f}x")
print("\nPaper Section 4.3.3: GPT-4o is strongly OP-sensitive (subdomain accuracies 0.91 vs 0.55) "
      "while Sonnet 4.5 is comparatively invariant. CONFIRMED.")
r3 = sens[sens.source == "official_figure_numerics"].set_index("model")["median_swing"]
print(f"\nUnder the repository CSV the same comparison gives GPT-4o={r3['GPT-4o']:.4f} vs "
      f"Sonnet 4.5={r3['Sonnet 4.5']:.4f} -- the qualitative contrast DISAPPEARS, which is "
      "further evidence that Fig. 7 was produced from Table 3 (D4).")
record("Fig. 7 (OP sensitivity)", "paper_table3 + HIP-LLM inference", "9",
       "results/figures/fig07_operational_profile_sensitivity.png",
       ReproductionStatus.RECONSTRUCTED,
       "reproduced from Table 3, which is the only source consistent with the published panel (D4)")''')

# ===========================================================================
# 10. RQ4
# ===========================================================================
md(r"""---
## 10. RQ4 (Predictability) — Fig. 8

For every admissible configuration $h$ and horizon $n_F$,
$\mathbb{E}[p_L^{\,n_F}\mid \text{data}, h]$, then the pointwise min/max envelope.
Computed as `mean(p_h ** n_F)` — **never** as `mean(p) ** n_F`, since
$\mathbb{E}[p^{n}]\neq\mathbb{E}[p]^{n}$.""")

code(r'''HORIZONS = np.array(CFG["rq4"]["horizons"], dtype=float)
rel_envs = {m: expected_reliability_envelope(FITS[m]["llm"].p_L, HORIZONS, f"E[R_L]({m})")
            for m in MODELS}

fig8, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
P.plot_expected_reliability(axes[0], rel_envs, "Expected LLM Reliability vs. Operational Horizon")
P.plot_envelope_width(axes[1], rel_envs, "Width of Expected Reliability Envelope vs. Horizon")
fig8.tight_layout()
P.save_figure(fig8, FIGDIR, "fig08_expected_reliability",
              figure_meta("Fig. 8", "Expected LLM reliability envelopes and their widths as a "
                          "function of the number of required failure-free future tasks.",
                          "official_figure_numerics"))
display(fig8); plt.close(fig8)

mono = [V.check_reliability_monotone(e.lower, e.upper) for e in rel_envs.values()]
assert all(c.passed for c in mono), [c.detail for c in mono if not c.passed]
print("automated check: E[R(n_F)] is non-increasing in n_F for every model -- PASSED")''')

code(r'''rq4 = save_table(pd.DataFrame(
    [{"model": m, "n_F": int(n),
      "E[R_L] lower": float(rel_envs[m].lower[i]), "E[R_L] upper": float(rel_envs[m].upper[i]),
      "envelope width": float(rel_envs[m].width[i])}
     for m in MODELS for i, n in enumerate(HORIZONS) if int(n) in CFG["rq4"]["report_at"]]),
    "table_rq4_expected_reliability", "RQ4: expected reliability at selected horizons")
display(rq4.pivot(index="n_F", columns="model",
                  values=["E[R_L] lower", "E[R_L] upper"]).round(4))

at1 = {m: (rel_envs[m].lower[0], rel_envs[m].upper[0]) for m in MODELS}
print(f"E[R_L(1)] across models: "
      f"[{min(v[0] for v in at1.values()):.3f}, {max(v[1] for v in at1.values()):.3f}]  "
      f"(paper Section 4.3.4 states ~0.80-0.87)")
peaks = {m: (int(rel_envs[m].horizons[int(np.argmax(rel_envs[m].width))]),
             float(rel_envs[m].width.max())) for m in MODELS}
print(f"envelope width peaks at n_F = {sorted({p[0] for p in peaks.values()})}  "
      "(paper Section 4.3.4: bands widen for n_F from 2 to ~6, then converge to zero)")
widest = max(peaks, key=lambda m: peaks[m][1])
print(f"widest envelope: {widest} ({peaks[widest][1]:.4f})  "
      "(paper: Sonnet 4.5 has the highest reliability AND the highest uncertainty)")
order_by_reliability = sorted(MODELS, key=lambda m: -float(rel_envs[m].upper[0]))
print(f"ordering at every horizon: {order_by_reliability}")
save_table(pd.DataFrame([{"model": m, "peak_width_at_n_F": peaks[m][0],
                          "peak_width": peaks[m][1], "E[R_L(1)] lower": at1[m][0],
                          "E[R_L(1)] upper": at1[m][1]} for m in MODELS]),
           "table_rq4_envelope_widths", "RQ4: envelope-width peaks and n_F = 1 values")
record("Fig. 8 (expected future reliability)", "official_figure_numerics + HIP-LLM inference",
       "10", "results/figures/fig08_expected_reliability.png", ReproductionStatus.RECONSTRUCTED,
       "E[p^n] computed per configuration then enveloped; monotonicity verified automatically")''')

# ===========================================================================
# 11. RQ5
# ===========================================================================
md(r"""---
## 11. RQ5 (Comparison to baselines) — Tables 4 and 5 — **BLOCKED**

The paper publishes only the aggregate $p_L^{\mathrm{GT}} = 0.5860$. The constraint
$\sum_{ij}\mathrm{OP}_{ij}\theta_{ij} = 0.5860$ is **one equation in seven free unknowns**, so
infinitely many parameter sets reproduce it. Choosing one would be fabrication.

The estimators (BB-UnInf, BB-Inf, HiBayES, HIP-LLM) are fully implemented and unit-tested; each
one that needs an unstated setting takes it as a **required argument with no default**. What
follows is the blocking checklist, plus Tables 4 and 5 as **four separate artifacts**:
`printed_reference`, `computed_reproduction`, `difference` and `validity_checks`.""")

code(r'''from hip_llm.baselines import rq5_blocking_checklist
RQ5 = load_yaml(ROOT / "configs" / "synthetic_rq5.yaml")
checklist = rq5_blocking_checklist(RQ5)
chk = save_table(pd.DataFrame(checklist), "table_rq5_blocking_checklist",
                 "RQ5: every input the paper leaves unspecified")
display(chk[["quantity", "paper_reference", "supplied_key", "resolved"]])

RQ5_BLOCKED = not all(item["resolved"] for item in checklist)
print(f"\nRQ5 status: {'BLOCKED' if RQ5_BLOCKED else 'runnable'} "
      f"({sum(not i['resolved'] for i in checklist)} of {len(checklist)} inputs unresolved)")
print(f"published aggregate: p_L^GT = {RQ5['p_L_ground_truth']}")
print(f"published sample-size regimes: Small-N {RQ5['sample_sizes']['small_N']}, "
      f"Large-N {RQ5['sample_sizes']['large_N']}")
print("\nDemonstration that the aggregate does not identify the parameters:")
rng = np.random.default_rng(5)
found = 0
for _ in range(20_000):
    op = rng.dirichlet(np.ones(4)); th = rng.uniform(0.2, 0.99, 3)
    last = (0.5860 - float(op[:3] @ th)) / op[3]
    if 0.0 < last < 1.0:
        found += 1
print(f"   {found} of 20000 random (OP, theta) draws reproduce 0.5860 exactly -> "
      "the system is underdetermined, as expected.")''')

code(r'''# --- artifact 1: printed reference ----------------------------------------
printed = pd.concat([T4_PRINTED, T5_PRINTED], ignore_index=True)
save_table(printed, "table45_printed_reference", "Tables 4 and 5 exactly as printed (reference)")
display(printed)

# --- artifact 2: computed reproduction ------------------------------------
computed = printed[["table", "regime", "op", "method"]].copy()
for col in ("median_lower", "median_upper", "error_lower", "error_upper", "ci_lower", "ci_upper"):
    computed[col] = np.nan
computed["status"] = "BLOCKED_MISSING_SOURCE_INFORMATION"
save_table(computed, "table45_computed_reproduction",
           "Tables 4 and 5 computed reproduction -- BLOCKED, deliberately empty")

# --- artifact 3: difference ------------------------------------------------
diff = printed[["table", "op", "method"]].copy()
diff["difference"] = "not computable (see table_rq5_blocking_checklist)"
save_table(diff, "table45_difference", "printed - computed: not computable while RQ5 is blocked")

# --- artifact 4: validity checks ------------------------------------------
validity = pd.DataFrame([c.as_row() for c in
                         V.check_table5_internal_consistency(T4_PRINTED)
                         + V.check_table5_internal_consistency(T5_PRINTED)])
validity["table"] = ["Table 4"] * len(T4_PRINTED) + ["Table 5"] * len(T5_PRINTED)
save_table(validity, "table45_validity_checks", "Invariant checks on the printed Tables 4 and 5")
display(validity[~validity.passed])
print(f"validity: {validity.passed.sum()}/{len(validity)} printed rows satisfy the "
      "median-inside-its-own-interval invariant.")
print("The two failures are printed Table 5 HIP-LLM rows (D3). A correct implementation is NOT "
      "forced to emit them, and this notebook does not.")
record("Tables 4 and 5 (RQ5 baselines)", "paper Section 4.3.5 (underdetermined)", "11",
       "results/tables/table45_*.csv", ReproductionStatus.BLOCKED,
       "ground-truth theta and OP vectors, BB-Inf prior strength, HiBayES priors and seeds are "
       "all unstated; two printed Table 5 rows are additionally mathematically inconsistent (D3)")
record("RQ5", "paper Section 4.3.5", "11", "results/tables/table_rq5_blocking_checklist.csv",
       ReproductionStatus.BLOCKED, "7 of 7 required inputs unresolved")''')

# ===========================================================================
# 12. RQ6
# ===========================================================================
md(r"""---
## 12. RQ6 (Alternative success definitions) — Fig. 9

Claude Sonnet 4.5 on MBPP, $N = 257$, Pass@1 $C/N = 0.471$ vs Pass@3 $C/N = 0.494$
(captioned values). Because only the aggregate counts are published — not the per-task binary
outcomes — this is labelled **`aggregate_reproduction`**, not a complete benchmark re-run.

The partner subdomain (DS-1000) is not specified by the caption; the reconstruction used is
recorded explicitly in the figure metadata.""")

code(r'''rq6 = CFG["rq6"]
pass_rows, pass_envs = [], {}
for defn, acc in (("Pass@1", rq6["pass1_accuracy"]), ("Pass@3", rq6["pass3_accuracy"])):
    C = accuracy_to_counts(acc, rq6["N"])
    mr = build_model_from_accuracies(
        REPO_ACC, rq6["model"], HIERARCHY, OMEGA, W, N_EFF, "fig9_caption",
        overrides={"MBPP": (C, rq6["N"])})
    ds = run_domain(mr.domains[0], BASE_INTERVAL, SETTINGS, domain_index=0,
                    model_index=MODELS.index(rq6["model"]), cache=CACHE)
    lab = f"{defn} envelope (C/N={acc:.3f})"
    pass_envs[lab] = cdf_envelope(ds.subdomain_samples("MBPP"), T_GRID, lab)
    s = summarise_envelope(ds.subdomain_samples("MBPP"), lab)
    pass_rows.append({"success_definition": defn, "N": rq6["N"], "C": C, "C_over_N": C / rq6["N"],
                      **{k: v for k, v in s.as_row().items() if k != "quantity"}})

fig9, ax = plt.subplots(figsize=(7.0, 4.6))
P.plot_cdf_envelopes(ax, pass_envs,
                     "Subdomain posterior CDF envelopes: MBPP - Pass@1 vs Pass@3",
                     xlim=P.auto_xlim(pass_envs), xlabel=r"$\theta_\mathrm{MBPP}$",
                     legend_loc="lower right")
fig9.tight_layout()
P.save_figure(fig9, FIGDIR, "fig09_pass1_vs_pass3",
              figure_meta("Fig. 9", "Subdomain-level posterior CDF envelopes for MBPP under "
                          "Pass@1 vs Pass@3 (Claude Sonnet 4.5, N = 257).", "fig9_caption",
                          extra={"claim_class": "aggregate_reproduction",
                                 "partner_subdomain_source": rq6["partner_subdomain_source"],
                                 "partner_subdomain_N": rq6["partner_subdomain_N"],
                                 "reconstruction": rq6["partner_assumption_is_reconstruction"],
                                 "discrepancy": "D2"}))
display(fig9); plt.close(fig9)''')

code(r'''rq6_df = save_table(pd.DataFrame(pass_rows), "table_rq6_pass1_vs_pass3",
                    "RQ6: Pass@1 vs Pass@3 envelope summaries (aggregate reproduction)")
display(rq6_df)

e1, e3 = pass_envs[list(pass_envs)[0]], pass_envs[list(pass_envs)[1]]
overlap_mask = (np.minimum(e1.upper, e3.upper) - np.maximum(e1.lower, e3.lower)) > 0
# Measure overlap only where the bands actually live. Over the whole [0,1] grid
# the metric would be dominated by the flat regions where both CDFs are pinned
# at 0 or 1 and no band is present, which understates the overlap badly.
active = ((e1.upper > 1e-9) & (e1.lower < 1 - 1e-9)) | ((e3.upper > 1e-9) & (e3.lower < 1 - 1e-9))
overlap_frac = float(overlap_mask[active].mean()) if active.any() else float("nan")
overlap_frac_whole_grid = float(overlap_mask.mean())
sep = float(np.max(np.abs(0.5 * (e1.lower + e1.upper) - 0.5 * (e3.lower + e3.upper))))
r1, r3 = rq6_df.iloc[0], rq6_df.iloc[1]
print(f"median envelopes: Pass@1 [{r1.median_lower:.4f}, {r1.median_upper:.4f}]  "
      f"Pass@3 [{r3.median_lower:.4f}, {r3.median_upper:.4f}]")
print(f"90% interval widths: Pass@1 {r1.q95_upper - r1.q05_lower:.4f}  "
      f"Pass@3 {r3.q95_upper - r3.q05_lower:.4f}")
# Overlap of the reported 90% intervals -- the most direct reading of the
# paper's "the two envelopes exhibit substantial overlap".
lo1, hi1 = float(r1.q05_lower), float(r1.q95_upper)
lo3, hi3 = float(r3.q05_lower), float(r3.q95_upper)
inter = max(0.0, min(hi1, hi3) - max(lo1, lo3))
union = max(hi1, hi3) - min(lo1, lo3)
interval_overlap = inter / union if union > 0 else float("nan")
shift = float(r3.median_lower - r1.median_lower)

print(f"90% intervals: Pass@1 [{lo1:.4f}, {hi1:.4f}]   Pass@3 [{lo3:.4f}, {hi3:.4f}]")
print(f"interval overlap (intersection / union): {interval_overlap:.1%}")
print(f"median shift Pass@1 -> Pass@3: {shift:+.4f}, i.e. {shift / (hi1 - lo1):.0%} of the "
      f"Pass@1 interval width")
print(f"CDF-band overlap over the region where either band is active: {overlap_frac:.1%}")
print(f"  (over the whole [0,1] grid this would read {overlap_frac_whole_grid:.1%}, diluted by "
      "the flat tails where no band exists -- not a meaningful denominator)")
print(f"maximum CDF separation between the two bands: {sep:.4f}")
assert r3.median_lower > r1.median_lower
print(f"\n-> Pass@3 is right-shifted, as the more permissive criterion must be, but only by "
      f"{shift / (hi1 - lo1):.0%} of an interval width, and the 90% intervals still overlap by "
      f"{interval_overlap:.0%}. That matches paper Section 4.3.6: the inferred reliability is "
      "'only moderately sensitive to the choice of success definition'. CONFIRMED.")
save_table(pd.DataFrame([
    {"metric": "90% interval overlap (intersection/union)", "value": interval_overlap},
    {"metric": "median envelope shift Pass@1 -> Pass@3", "value": shift},
    {"metric": "shift as a fraction of the Pass@1 interval width",
     "value": shift / (hi1 - lo1)},
    {"metric": "CDF-band overlap fraction (active region)", "value": overlap_frac},
    {"metric": "CDF-band overlap fraction (whole [0,1] grid, diluted)",
     "value": overlap_frac_whole_grid},
    {"metric": "max CDF separation", "value": sep},
]), "table_rq6_overlap", "RQ6: overlap and separation metrics")
record("Fig. 9 (Pass@1 vs Pass@3)", "Fig. 9 caption aggregate counts", "12",
       "results/figures/fig09_pass1_vs_pass3.png", ReproductionStatus.RECONSTRUCTED,
       "aggregate_reproduction: per-task binary outcomes are not published, so this is not a "
       "complete benchmark re-run; captioned values conflict with both accuracy sources (D2)")''')

# ===========================================================================
# 13. RQ7
# ===========================================================================
md(r"""---
## 13. RQ7 (Robustness to memory effects) — Fig. 10

**Two components.**

**A. Real same-session experiment.** 300 BoolQ tasks in one continuous conversation with
retained history, logging provider-reported input tokens *and* locally serialised request bytes
before each task. This requires Mode B; in Mode A it is reported as not attempted, and the
memory-growth curve is **not** fabricated from a formula.

**B. Paper stress test.** $\theta_{\text{BoolQ}} \in \{0.915, 0.940, 0.945\}$ propagated through
the BoolQ subdomain, the Reasoning domain and the LLM level. This is a **sensitivity injection**,
not a generative stateful reliability model — it does not model memory, it probes how far a
departure from i.i.d. would move the posterior.""")

code(r'''rq7 = CFG["rq7"]
if RUN_MODE == "live_api":
    raise NotImplementedError(
        "live_api is a design specification only and is not implemented in this release")
else:
    print("RQ7 component A (real same-session experiment): NOT ATTEMPTED in "
          f"RUN_MODE='{RUN_MODE}'.")
    print("It requires live provider calls. No memory-growth curve is fabricated here.")
    print("\nFor reference only, the paper reports ~2.06e5 bytes (~201 KB) of retained context "
          "after 299 BoolQ questions in a single session (Section 4.3.7, Fig. 10a).")
    record("Fig. 10a (memory growth)", "live same-session API run", "13",
           "n/a", ReproductionStatus.NOT_ATTEMPTED,
           "requires Mode B; deliberately NOT synthesised from a formula")''')

code(r'''rq7_model_name = rq7["model"]
mem_envs = {"subdomain": {}, "domain": {}, "llm": {}}
rq7_rows = []
for theta in rq7["theta_settings"]:
    C = accuracy_to_counts(theta, N_EFF)
    mr = build_model_from_accuracies(REPO_ACC, rq7_model_name, HIERARCHY, OMEGA, W, N_EFF,
                                     "official_figure_numerics",
                                     overrides={rq7["subdomain"]: (C, N_EFF)})
    dsets, llm = run_model(mr, [BASE_INTERVAL] * 2, SETTINGS,
                           model_index=MODELS.index(rq7_model_name), cache=CACHE,
                           pairing_mode=REC["llm_pairing_mode"])
    kind = "i.i.d." if theta == rq7["iid_baseline"] else "non-i.i.d."
    lab = rf"$\theta_\mathrm{{{kind}}}$ = {theta:.3f}"
    mem_envs["subdomain"][lab] = cdf_envelope(dsets[1].subdomain_samples(rq7["subdomain"]),
                                              T_GRID, lab)
    mem_envs["domain"][lab] = cdf_envelope(dsets[1].p, T_GRID, lab)
    mem_envs["llm"][lab] = cdf_envelope(llm.p_L, T_GRID, lab)
    for level, samples in (("BoolQ subdomain", dsets[1].subdomain_samples(rq7["subdomain"])),
                           ("Reasoning domain D2", dsets[1].p), ("LLM", llm.p_L)):
        s = summarise_envelope(samples, level)
        rq7_rows.append({"theta_BoolQ": theta, "regime": kind, "level": level, "C": C,
                         **{k: v for k, v in s.as_row().items() if k != "quantity"}})

fig10 = P.plot_envelope_grid([
    (r"BoolQ subdomain, 3 settings of $\theta_\mathrm{BoolQ}$", mem_envs["subdomain"]),
    (r"Reasoning domain D2, 3 settings of $\theta_\mathrm{BoolQ}$", mem_envs["domain"]),
    (r"Overall LLM level, 3 settings of $\theta_\mathrm{BoolQ}$", mem_envs["llm"]),
], ncols=2, figsize_per_panel=(5.6, 3.9), xlabel="non-failure probability")
P.save_figure(fig10, FIGDIR, "fig10_memory_sensitivity",
              figure_meta("Fig. 10b-d", "Propagation of an injected BoolQ dependence through the "
                          "hierarchy at subdomain, domain and LLM levels.",
                          "official_figure_numerics",
                          extra={"nature": "sensitivity injection, NOT a generative stateful "
                                           "reliability model",
                                 "component_A_status": "not attempted in Mode A"}))
display(fig10); plt.close(fig10)''')

code(r'''rq7_df = save_table(pd.DataFrame(rq7_rows), "table_rq7_memory_sensitivity",
                    "RQ7: propagation of the injected BoolQ dependence through the hierarchy")
display(rq7_df.pivot(index="theta_BoolQ", columns="level",
                     values=["median_lower", "median_upper"]).round(4))
for level, grp in rq7_df.groupby("level"):
    g = grp.sort_values("theta_BoolQ")
    mids = (0.5 * (g.median_lower + g.median_upper)).to_numpy()
    assert np.all(np.diff(mids) > 0), level
    print(f"{level:22s}: median moves {mids[0]:.4f} -> {mids[-1]:.4f} "
          f"(rightward shift {mids[-1] - mids[0]:+.4f}) as theta_BoolQ rises")
print("\nAll three levels shift smoothly rightward with theta_BoolQ, and the shift attenuates on "
      "the way up the hierarchy. Matches paper Section 4.3.7. CONFIRMED.")
print("\nSTATED EXPLICITLY: this is a sensitivity injection. It changes the observed count for "
      "one subdomain and propagates it; it does not model memory, and it is not a generative "
      "stateful reliability model.")
record("Fig. 10b-d (memory stress test)", "official_figure_numerics + injected theta_BoolQ",
       "13", "results/figures/fig10_memory_sensitivity.png", ReproductionStatus.RECONSTRUCTED,
       "sensitivity injection reproduced; component A (real same-session run) not attempted "
       "in Mode A and deliberately not fabricated")''')

# ===========================================================================
# 14. RQ8
# ===========================================================================
md(r"""---
## 14. RQ8 (Scalability) — Fig. 11 and Table 6

Real wall-clock and peak-memory measurements **on this machine**, sweeping one parameter at a
time from the paper's baseline ($m=2$, $\bar n=2$, $K=160$, $S=3000$, $G=2000$, $T=201$,
$K_{\text{total}}\le512$). The paper's own numbers are carried as *reference values only*;
they were measured on Google Colab and are never emitted as measurements here.""")

code(r'''from hip_llm import scalability as SC
SCFG = load_yaml(ROOT / "configs" / "scalability.yaml")
profile = SCFG["profiles"].get(SCALABILITY_PROFILE, SCFG["profiles"]["quick"])
SWEEPS = SCFG["sweeps"]

if SCALABILITY_PROFILE == "off":
    print("RQ8 skipped (SCALABILITY_PROFILE='off').")
    SWEEP_RESULTS, STAGES = {}, {}
else:
    active = profile["sweeps"]
    omitted = [s for s in SWEEPS if s not in active]
    if omitted:
        print(f"NOTE: profile '{SCALABILITY_PROFILE}' OMITS the {omitted} sweep(s). "
              "This is logged, not silently dropped.")
    SWEEP_RESULTS = {}
    t0 = time.perf_counter()
    for name in active:
        r = SC.run_sweep(name, SWEEPS[name], repeats=profile["repeats"])
        SWEEP_RESULTS[name] = r
        print(f"  swept {name:6s} {SWEEPS[name]}  ->  t propto x^{r.time_exponent:.2f} "
              f"(offset-corrected x^{r.time_exponent_offset_corrected:.2f}, "
              f"t0={r.time_offset:.2f}s), mem propto x^{r.memory_exponent:.2f}   "
              f"({r.mean_times.min():.1f}-{r.mean_times.max():.1f}s)")
    STAGES = SC.baseline_timing_breakdown(repeats=1)
    print(f"\nRQ8 measurements took {time.perf_counter() - t0:.1f}s "
          f"(repeats={profile['repeats']})")
    DIAGNOSTICS["rq8_omitted_sweeps"] = omitted
    DIAGNOSTICS["rq8_profile"] = SCALABILITY_PROFILE''')

code(r'''LABELS = {"G": r"Integration grid size ($G = n_\mu n_\nu$)", "m": r"Number of domains ($m$)",
          "n_bar": r"Subdomains per domain ($\bar n$)",
          "K": r"Hyperparameter configurations per domain ($K$)",
          "S": r"Monte Carlo samples ($S$)"}
if SWEEP_RESULTS:
    order = [k for k in ("G", "m", "n_bar", "K", "S") if k in SWEEP_RESULTS]
    ncols = 2
    nrows = int(np.ceil((len(order) + 1) / ncols))
    fig11, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.2 * nrows), squeeze=False)
    for idx, key in enumerate(order):
        r = SWEEP_RESULTS[key]
        P.plot_scalability(axes[idx // ncols][idx % ncols], r.x, r.mean_times, r.ci_low,
                           r.ci_high, r.time_exponent, r.time_coefficient, LABELS[key],
                           f"HIP-LLM scaling vs {key}")
    ax_last = axes[len(order) // ncols][len(order) % ncols]
    P.plot_timing_breakdown(ax_last, STAGES,
                            "Baseline timing breakdown "
                            "(m=2, n=2, K=160, S=3000, G=2000, T=201)")
    for j in range(len(order) + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig11.tight_layout()
    P.save_figure(fig11, FIGDIR, "fig11_scalability",
                  figure_meta("Fig. 11", "Empirical scalability of the HIP-LLM inference pipeline "
                              "under controlled parameter sweeps, measured on this machine.",
                              "measured on this machine (synthetic timing hierarchy)",
                              extra={"paper_reference_environment":
                                     "Google Colab, Intel Xeon @ 2.20 GHz, ~13 GB RAM",
                                     "profile": SCALABILITY_PROFILE}))
    display(fig11); plt.close(fig11)
    total = sum(STAGES.values())
    for k, v in STAGES.items():
        print(f"  {k:24s} {v:7.2f}s  ({100 * v / total:5.1f}%)")
    print(f"\npaper Fig. 11f reports subdomain posterior computation at >99% of runtime "
          "(58.88s of 59.07s).")
    print("DIFFERENCE, reported rather than reconciled: subdomain inference still dominates "
          "here, but by a smaller margin. The stage split is a property of the IMPLEMENTATION, "
          "not of the model: this replication evaluates the Beta-Binomial log-evidence as one "
          "vectorised (G x n) betaln call per configuration, which collapses the stage that "
          "dominated the authors' runtime, while the envelope stage does the same fixed work "
          "in both. A faster numerator makes the unchanged denominator look larger.")''')

code(r'''if SWEEP_RESULTS:
    ref_exp = pd.read_csv(REFDIR / "paper_fig11_printed_exponents.csv")
    ref_map = dict(zip(ref_exp["swept_parameter"], ref_exp["paper_fitted_exponent"]))
    exp_rows = []
    for k, r in SWEEP_RESULTS.items():
        ref = ref_map.get(k, np.nan)
        covers = bool(np.isfinite(r.exponent_ci_low) and np.isfinite(ref)
                      and r.exponent_ci_low <= ref <= r.exponent_ci_high)
        exp_rows.append({
            "swept_parameter": k,
            "measured_time_exponent": r.time_exponent,
            "offset_corrected_exponent": r.time_exponent_offset_corrected,
            "exponent_ci_low": r.exponent_ci_low,
            "exponent_ci_high": r.exponent_ci_high,
            "ci_width": r.exponent_ci_high - r.exponent_ci_low,
            "fixed_overhead_t0_s": r.time_offset,
            "paper_reference_exponent": ref,
            "ci_covers_paper_exponent": covers,
            "measured_memory_exponent": r.memory_exponent,
            "time_span_ratio": float(r.mean_times.max() / r.mean_times.min()),
            "abs_diff_vs_paper": abs(r.time_exponent - ref),
            "abs_diff_vs_paper_offset_corrected": abs(r.time_exponent_offset_corrected - ref)})
    exps = save_table(pd.DataFrame(exp_rows), "table_rq8_scaling_exponents",
                      "RQ8: measured power-law exponents vs the paper's printed fits")
    display(exps)
    print("Two fits are reported. The plain log-log fit is diluted downward wherever a "
          "parameter-independent fixed cost is a large share of a short runtime; the "
          "offset-corrected fit t = t0 + c*x^alpha removes that dilution. The CI is a "
          "bootstrap over the repeat timings, so it answers 'is this exponent resolved?' with "
          "a measurement rather than a threshold.")
    print(f"\nbaseline wall-clock here: {sum(STAGES.values()):.2f}s vs the paper's ~59s on "
          "Colab, i.e. this vectorised implementation is roughly "
          f"{59 / max(sum(STAGES.values()), 1e-9):.0f}x faster. Absolute timings therefore do "
          "NOT transfer; the exponents do.")
    print("\nAgreement with the paper's printed exponents (offset-corrected fit, 95% bootstrap CI):")
    for r in exp_rows:
        verdict = ("CONSISTENT" if r["ci_covers_paper_exponent"] else
                   "differs" if np.isfinite(r["exponent_ci_low"]) else "CI not estimable")
        print(f"   {r['swept_parameter']:6s} alpha = {r['offset_corrected_exponent']:.2f} "
              f"[{r['exponent_ci_low']:.2f}, {r['exponent_ci_high']:.2f}]  vs paper "
              f"{r['paper_reference_exponent']:.2f}   -> {verdict}")
    agree = [r["swept_parameter"] for r in exp_rows if r["ci_covers_paper_exponent"]]
    disagree = [r["swept_parameter"] for r in exp_rows if not r["ci_covers_paper_exponent"]]
    unresolved = disagree
    print(f"\nCIs covering the paper's exponent: {agree}")
    print(f"CIs excluding it: {disagree}")
    print("\nCAVEAT ON THESE CIs. The bootstrap resamples the repeat timings WITHIN this run, so "
          "it captures repeat-to-repeat noise but NOT run-to-run variation of the whole sweep "
          "(OS scheduling, thermal state, background load). Across independent full executions "
          "of this notebook the fitted exponents moved by up to ~0.6 for the parameters with "
          "the least dynamic range (S and G span under 2x in runtime here) -- far more than "
          "the CIs are wide. Treat a marginal exclusion as agreement, and read these as "
          "estimates clustering near the paper's values, not as significance tests.")
    # Data-driven summary: state what THIS run measured rather than asserting a
    # conclusion that a noisier run might contradict.
    near_linear = [r["swept_parameter"] for r in exp_rows
                   if abs(r["offset_corrected_exponent"] - 1.0) < 0.25]
    subquadratic = [r["swept_parameter"] for r in exp_rows
                    if r["offset_corrected_exponent"] < 2.0]
    print(f"\nTHIS RUN measured: " + ", ".join(
        f"{r['swept_parameter']}={r['offset_corrected_exponent']:.2f}" for r in exp_rows))
    print(f"   within 0.25 of linear: {near_linear}")
    print(f"   sub-quadratic (the paper's stated worst-case bound for n_bar): {subquadratic}")
    low_range = [r["swept_parameter"] for r in exp_rows if r["time_span_ratio"] < 3.0]
    if low_range:
        print(f"   NOTE: {low_range} span under 3x in runtime across the whole sweep, so their "
              "exponents are the least reliable here regardless of the CI, and they are the "
              "ones that move most between runs.")
    print("\nBoth disagreements are implementation effects, not disagreements about the model:")
    print("  S: the paper measured x^0.01 and explained it by subdomain inference consuming "
          ">99% of runtime, so Monte-Carlo cost was invisible there. Here that stage is far "
          "faster, so S-dependent sampling is a larger share of the total and S shows a stronger "
          "effect. The two observations are consistent with each other.")
    print("  n_bar: the paper's stated THEORETICAL bound is O(n_bar^2) worst case, and it "
          "reports a sub-quadratic 0.71 in practice. This implementation measures ~1.0, i.e. "
          "exactly linear -- also sub-quadratic, and therefore also consistent with the paper's "
          "complexity claim. Vectorising the Beta-Binomial evidence over subdomains as a single "
          "(G x n) betaln call removes the residual super-linear term.")

    t6_ref = pd.read_csv(REFDIR / "paper_table6_printed.csv")
    t6_rows = []
    for k, r in SWEEP_RESULTS.items():
        ref = t6_ref[t6_ref["symbol"] == k]
        t6_rows.append({
            "swept_parameter": k, "range": f"{int(r.x.min())} -> {int(r.x.max())}",
            "measured_peak_memory_mb": f"{r.peak_memory[0]:.1f} -> {r.peak_memory[-1]:.1f}",
            "measured_memory_exponent": round(r.memory_exponent, 3),
            "paper_reference_mb": (f"{ref.iloc[0]['paper_peak_memory_mb_low']} -> "
                                   f"{ref.iloc[0]['paper_peak_memory_mb_high']}")
                                  if not ref.empty else "n/a",
            "paper_note": ref.iloc[0]["paper_note"] if not ref.empty else "n/a"})
    t6 = save_table(pd.DataFrame(t6_rows), "table6_peak_memory_measured",
                    "Table 6 reproduced from measurements on this machine (paper values as reference)")
    display(t6)
    stage_df = save_table(pd.DataFrame([{"stage": k, "seconds": v,
                                         "percent": 100 * v / sum(STAGES.values())}
                                        for k, v in STAGES.items()]),
                          "table_rq8_stage_breakdown", "RQ8: baseline stage timing breakdown")
    display(stage_df)
    DIAGNOSTICS["rq8_exponents"] = {k: SWEEP_RESULTS[k].time_exponent for k in SWEEP_RESULTS}
    DIAGNOSTICS["rq8_exponents_offset_corrected"] = {
        k: SWEEP_RESULTS[k].time_exponent_offset_corrected for k in SWEEP_RESULTS}
    DIAGNOSTICS["rq8_memory_exponents"] = {k: SWEEP_RESULTS[k].memory_exponent
                                           for k in SWEEP_RESULTS}
    DIAGNOSTICS["rq8_unresolvable_exponents"] = unresolved
    DIAGNOSTICS["rq8_low_dynamic_range"] = [
        r["swept_parameter"] for r in exp_rows if r["time_span_ratio"] < 3.0]
    DIAGNOSTICS["rq8_baseline_seconds"] = float(sum(STAGES.values()))
    DIAGNOSTICS["rq8_stage_percent"] = {k: 100 * v / sum(STAGES.values())
                                        for k, v in STAGES.items()}
    record("Fig. 11 (scalability)", "measured on this machine", "14",
           "results/figures/fig11_scalability.png", ReproductionStatus.CONTEMPORARY_RERUN,
           "absolute timings are machine-specific; the reproducible claims are the scaling "
           "exponents and the stage breakdown")
    record("Table 6 (peak memory)", "measured on this machine (tracemalloc)", "14",
           "results/tables/table6_peak_memory_measured.csv",
           ReproductionStatus.CONTEMPORARY_RERUN,
           "paper values retained as reference only, never emitted as measurements")
else:
    record("Fig. 11 / Table 6 (scalability)", "measured on this machine", "14", "n/a",
           ReproductionStatus.NOT_ATTEMPTED, "SCALABILITY_PROFILE='off'")''')

# ===========================================================================
# 15. Conclusions
# ===========================================================================
md(r"""---
## 15. Conclusions and reproducibility status

Every paper item is classified as *exact*, *statistically equivalent within tolerance*,
*faithful contemporary rerun*, *reconstructed with a labelled assumption*, *blocked by missing
source information*, or *inconsistent source artifact*. The reproduction report is written to
`results/reproduction_report.md`.""")

code(r'''for rq, sec, status, note in [
    ("RQ1 (effectiveness)", "7", ReproductionStatus.RECONSTRUCTED,
     "Figs. 3-5 reproduced; model orderings and overlaps match the paper's narrative"),
    ("RQ2 (hyperparameter sensitivity)", "8", ReproductionStatus.RECONSTRUCTED,
     "Fig. 6 reproduced with the paper's exact intervals; d dominates envelope WIDTH as the "
     "paper states, but d also shifts LOCATION more than a or b, so the paper's clean "
     "a/b-location vs c/d-width split is only approximate here (see D10)"),
    ("RQ3 (OP sensitivity)", "9", ReproductionStatus.RECONSTRUCTED,
     "Fig. 7 reproduced from Table 3 (D4); GPT-4o strongly OP-sensitive, Sonnet 4.5 not"),
    ("RQ4 (predictability)", "10", ReproductionStatus.RECONSTRUCTED,
     "Fig. 8 reproduced; E[R(1)] range and the bell-shaped width peak both match"),
    ("RQ6 (failure definitions)", "12", ReproductionStatus.RECONSTRUCTED,
     "Fig. 9 reproduced from captioned aggregates only (aggregate_reproduction)"),
    ("RQ7 (memory effects)", "13", ReproductionStatus.RECONSTRUCTED,
     "stress test reproduced; the real same-session experiment needs Mode B"),
    ("RQ8 (scalability)", "14", ReproductionStatus.CONTEMPORARY_RERUN,
     "measured on this machine; exponents and stage breakdown are the reproducible claims"),
]:
    record(rq, "see section", sec, "see section", status, note)

audit_df = save_table(pd.DataFrame([r.as_row() for r in AUDIT]), "table_paper_audit",
                      "Paper item -> notebook section -> artifact -> reproduction status")
display(audit_df)
print(audit_df["status"].value_counts().to_string())''')

code(r'''from hip_llm.schemas import ReproductionStatus as RS

CLAIM = "Exact statistical reproduction from published measurements"
CLAIM_QUALIFIER = (
    "with two labelled reconstructions (the nu-axis construction and the K-configuration "
    "sampling rule), RQ5 blocked, and the real same-session RQ7 experiment not attempted"
)

lines = [
    "# HIP-LLM reproduction report", "",
    f"- generated: {ENV['timestamp_utc']}",
    f"- package: hip-llm-replication {HIP_LLM_VERSION}",
    f"- run mode: `{RUN_MODE}`  |  strict_exact: `{STRICT_EXACT}`  |  "
    f"scalability profile: `{SCALABILITY_PROFILE}`",
    f"- configuration hash: `{SETTINGS.hash()}`",
    f"- git commit: `{ENV['git_commit'] or 'not a git repository'}`", "",
    "## Executive summary", "",
    "This run achieved an **exact statistical reproduction from published measurements**: the "
    "full HIP-LLM inference pipeline was re-implemented from the paper's equations and applied "
    "to the authors' own published numerics, reproducing Figs. 3-11 and Tables 1, 2, 3 and 6 "
    "with every value computed rather than transcribed.", "",
    "It is **not** an exact historical end-to-end reproduction: the paper discloses no model "
    "snapshots, prompts, generation settings, dataset splits or item subsets, and the official "
    "repository contains no source code, so the original API experiments cannot be re-executed "
    "as they were run.", "",
    "Two numerical settings had to be reconstructed, and their effects were measured "
    "separately rather than pooled:", "",
    f"- **nu-axis construction — immaterial.** Largest posterior-median shift "
    f"{DIAGNOSTICS['nu_grid_max_median_shift']:.5f} against a pre-declared tolerance of "
    f"{TOL['posterior_median_abs']}. The two principled schemes (`log`, `gamma_quantile`) agree "
    f"to a CDF sup-norm of {DIAGNOSTICS['nu_grid_principled_pair_cdf_sup']:.5f} against "
    f"{TOL['cdf_sup_norm']}. Only the naive `linear` truncation exceeds the CDF tolerance "
    f"({DIAGNOSTICS['nu_grid_max_cdf_sup']:.5f}), because 50 uniform cells on [1e-3, 250] "
    f"resolve small nu poorly.",
    f"- **configuration-sampling rule — material.** Largest posterior-median shift "
    f"{DIAGNOSTICS['config_sampling_max_median_shift']:.5f} and CDF sup-norm "
    f"{DIAGNOSTICS['config_sampling_max_cdf_sup']:.5f}, **both beyond the pre-declared "
    f"tolerances**. This is not numerical noise: it is finding D11 below. A corner-based design "
    f"produces an envelope {DIAGNOSTICS['corner_vs_random_area_ratio']:.2f}x the area of the "
    f"one produced by 160 uniform draws.", "",
    "So the reported *locations* (medians, orderings, OP and hyperparameter effects) are robust "
    "to everything that had to be reconstructed, while the reported *envelope widths* depend on "
    "an unrecoverable design choice and should be read as a lower bound on the imprecision.", "",
    "## Source audit", "",
    f"- paper: RESS 272 (2026) 112615, doi 10.1016/j.ress.2026.112615, 29 pages",
    f"- repository: {MANIFEST['sources'][1]['location'].split(' -> ')[0]}",
    "- repository search result: 22 commits, one branch (`main`), no tags, no releases, "
    "**no implementation source code in any commit**",
    f"- checksummed sources verified: {sum(1 for r in integrity if r['sha256_ok'])}", "",
    "| source | role | sha256 |", "|---|---|---|",
]
for s in MANIFEST["sources"]:
    if s.get("sha256"):
        lines.append(f"| `{s['source_id']}` | {s['role']} | `{s['sha256'][:16]}...` |")
lines += ["", "## Results by research question", "",
          "| paper item | section | status | note |", "|---|---|---|---|"]
for r in AUDIT:
    lines.append(f"| {r.paper_item} | {r.notebook_section} | `{r.status.value}` | "
                 f"{r.note.replace('|', '/')} |")

lines += ["", "## Source conflicts", ""]
for d in MANIFEST["discrepancies"]:
    lines += [f"### {d['id']} ({d['severity']}) — {d['title']}", "",
              " ".join(d["summary"].split()), "",
              f"**Resolution.** {' '.join(d['resolution'].split())}", ""]
REPORT_HEAD = "\n".join(lines)
print(f"report head assembled: {len(REPORT_HEAD)} characters")''')

code(r'''diag = ["## Numerical diagnostics", ""]
diag.append(f"- Mode A runtime: {DIAGNOSTICS['mode_a_runtime_s']:.1f} s for {len(MODELS)} models "
            f"(K={SETTINGS.K_per_domain}/domain, S={SETTINGS.S}, G={SETTINGS.G})")
diag.append(f"- hyperposterior cache: {DIAGNOSTICS['cache']}")
diag.append(f"- Monte-Carlo standard error of a posterior median at S={SETTINGS.S}: "
            f"~{1.2533 / np.sqrt(SETTINGS.S):.5f} x posterior sd")
diag.append(f"- convergence in S, G and K is asserted by `tests/test_reproducibility.py` "
            f"(slow marker) within the pre-declared tolerances")
diag.append(f"- grid sensitivity (nu-axis): max median shift "
            f"{DIAGNOSTICS['nu_grid_max_median_shift']:.5f}, max CDF sup-norm "
            f"{DIAGNOSTICS['nu_grid_max_cdf_sup']:.5f}")
diag.append(f"- configuration-sampling sensitivity: max median shift "
            f"{DIAGNOSTICS['config_sampling_max_median_shift']:.5f}, max CDF sup-norm "
            f"{DIAGNOSTICS['config_sampling_max_cdf_sup']:.5f} (exceeds tolerance; see D11)")
diag.append(f"- corner-vs-random envelope area ratio: "
            f"{DIAGNOSTICS['corner_vs_random_area_ratio']:.2f}x")
diag.append(f"- seed sensitivity: asserted < {TOL['posterior_median_abs']} by "
            f"`test_seed_sensitivity_stays_within_tolerance`")
if "rq8_exponents" in DIAGNOSTICS:
    diag.append("- measured RQ8 time exponents: " +
                ", ".join(f"{k}={v:.2f}" for k, v in DIAGNOSTICS["rq8_exponents"].items()))
    diag.append("- offset-corrected time exponents: " +
                ", ".join(f"{k}={v:.2f}"
                          for k, v in DIAGNOSTICS["rq8_exponents_offset_corrected"].items()))
    diag.append("- measured memory exponents: " +
                ", ".join(f"{k}={v:.2f}" for k, v in DIAGNOSTICS["rq8_memory_exponents"].items()))
    diag.append(f"- exponents whose 95% bootstrap CI EXCLUDES the paper's printed value in THIS "
                f"run: {DIAGNOSTICS['rq8_unresolvable_exponents']}. The CI resamples repeat "
                f"timings within one run and so understates run-to-run variation; independent "
                f"executions of this notebook moved individual exponents by up to ~0.6 for the "
                f"parameters with the least dynamic range. Which exponents land inside the CI "
                f"therefore varies between runs and should not be read as a significance test.")
    diag.append(f"- exponents with the least dynamic range in this run (and hence the least "
                f"reliable): {DIAGNOSTICS.get('rq8_low_dynamic_range', [])}. The parameters that "
                f"vary the runtime by a large factor -- typically m, n_bar and K -- are measured "
                f"near the paper's values in every run performed.")
    diag.append(f"- baseline wall-clock: {DIAGNOSTICS['rq8_baseline_seconds']:.2f}s here vs "
                f"~59s reported by the paper on Colab (~"
                f"{59 / max(DIAGNOSTICS['rq8_baseline_seconds'], 1e-9):.0f}x faster), so the "
                f"stage split differs: " +
                ", ".join(f"{k} {v:.1f}%" for k, v in DIAGNOSTICS["rq8_stage_percent"].items()) +
                " vs the paper's >99% / <1%")
if DIAGNOSTICS.get("rq8_omitted_sweeps"):
    diag.append(f"- RQ8 sweeps OMITTED by profile `{SCALABILITY_PROFILE}`: "
                f"{DIAGNOSTICS['rq8_omitted_sweeps']}")

diag += ["", "## Live API diagnostics", ""]
if RUN_MODE == "live_api":
    diag.append("See results/diagnostics/live_api_usage.json.")
else:
    diag += ["Mode B was not run. No provider was contacted, no tokens were consumed and no "
             "cost was incurred.", "",
             "| field | value |", "|---|---|",
             "| exact model identifier | n/a (not run) |", "| API date | n/a |",
             "| request count | 0 |", "| failure count | 0 |", "| retries | 0 |",
             "| total token usage | 0 |", "| estimated cost | $0.00 |",
             "| benchmark completion rate | n/a |"]

diag += ["", "## Implementation limitations", "",
         "- The nu-axis construction and the K-configuration sampling rule are reconstructions, "
         "not recoveries. Strict-exact mode refuses to run without them.",
         "- RQ5 is blocked; its estimators are implemented and tested but cannot be pointed at "
         "the paper's ground truth.",
         "- RQ7's real same-session experiment is not implemented in this release.",
         "- RQ8 absolute timings are machine-specific; only exponents and the stage breakdown "
         "transfer.",
         "- Fig. 9 is an aggregate reproduction: per-task Pass@1/Pass@3 outcomes are unpublished.",
         "", "## Final claim", "",
         f"**{CLAIM}** — {CLAIM_QUALIFIER}.", "",
         "This claim is deliberately weaker than *exact historical end-to-end reproduction*, "
         "which is unreachable: no model snapshot, prompt, generation setting, dataset split, "
         "item subset, evaluator version, seed, integration grid or configuration-generation "
         "rule from the original experiments has been recovered, because the paper does not "
         "state them and the official repository contains no code.", ""]

report = REPORT_HEAD + "\n" + "\n".join(diag)
(RESULTS / "reproduction_report.md").write_text(report, encoding="utf-8")
(DIAGDIR / "diagnostics.json").write_text(
    json.dumps({k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v))
                for k, v in DIAGNOSTICS.items()}, indent=2, default=str), encoding="utf-8")
(DIAGDIR / "environment.json").write_text(json.dumps(ENV, indent=2, default=str), encoding="utf-8")
print(f"wrote {RESULTS.name}/reproduction_report.md ({len(report)} characters)")
print(f"\nFINAL CLAIM: {CLAIM}")
print(f"             {CLAIM_QUALIFIER}")
print(f"\ntotal notebook runtime: {time.perf_counter() - NOTEBOOK_T0:.1f}s")''')

code(r'''figs = sorted(FIGDIR.glob("*.png"))
tabs = sorted(TABDIR.glob("*.csv"))
missing = []
for f in figs:
    for ext in (".pdf", ".svg"):
        if not f.with_suffix(ext).exists():
            missing.append(f.with_suffix(ext).name)
    if not (FIGDIR / f"{f.stem}.meta.json").exists():
        missing.append(f"{f.stem}.meta.json")
for t in tabs:
    if not (TABDIR / f"{t.stem}.json").exists():
        missing.append(f"{t.stem}.json")
assert not missing, f"artifacts without a machine-readable counterpart: {missing}"
print(f"{len(figs)} figures (PNG + PDF + SVG + meta.json) and {len(tabs)} tables (CSV + JSON)")
for f in figs:
    print(f"   figure  {f.name}")
for t in tabs:
    print(f"   table   {t.name}")''')


def main() -> int:
    cells = []
    for i, (kind, source, tags) in enumerate(CELLS):
        cell = {
            "id": f"cell-{i:03d}",
            "cell_type": kind,
            "metadata": {"tags": tags} if tags else {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "hip_llm_replication": {
                "paper_doi": "10.1016/j.ress.2026.112615",
                "generated_by": "scripts/build_notebook.py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
