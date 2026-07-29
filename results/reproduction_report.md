# HIP-LLM reproduction report

- generated: 2026-07-29T09:14:25+00:00
- package: hip-llm-replication 1.0.0
- run mode: `published_numerics`  |  strict_exact: `False`  |  scalability profile: `full`
- configuration hash: `b16d8266b8eda3752442ea4416434b8cfabf09b0b9c3f149edd2f579da03dd1a`
- git commit: `e9a1b36cbfd3c38bd7bb050467ea0a866abcf803`

## Executive summary

This run achieved an **exact statistical reproduction from published measurements**: the full HIP-LLM inference pipeline was re-implemented from the paper's equations and applied to the authors' own published numerics, reproducing Figs. 3-11 and Tables 1, 2, 3 and 6 with every value computed rather than transcribed.

It is **not** an exact historical end-to-end reproduction: the paper discloses no model snapshots, prompts, generation settings, dataset splits or item subsets, and the official repository contains no source code, so the original API experiments cannot be re-executed as they were run.

Two numerical settings had to be reconstructed, and their effects were measured separately rather than pooled:

- **nu-axis construction — immaterial.** Largest posterior-median shift 0.00129 against a pre-declared tolerance of 0.005. The two principled schemes (`log`, `gamma_quantile`) agree to a CDF sup-norm of 0.00767 against 0.02. Only the naive `linear` truncation exceeds the CDF tolerance (0.03333), because 50 uniform cells on [1e-3, 250] resolve small nu poorly.
- **configuration-sampling rule — material.** Largest posterior-median shift 0.01022 and CDF sup-norm 0.17667, **both beyond the pre-declared tolerances**. This is not numerical noise: it is finding D11 below. A corner-based design produces an envelope 1.68x the area of the one produced by 160 uniform draws.

So the reported *locations* (medians, orderings, OP and hyperparameter effects) are robust to everything that had to be reconstructed, while the reported *envelope widths* depend on an unrecoverable design choice and should be read as a lower bound on the imprecision.

## Source audit

- paper: RESS 272 (2026) 112615, doi 10.1016/j.ress.2026.112615, 29 pages
- repository: https://github.com/aghazadehchakherlou-web/llm-imprecise-bayes
- repository search result: 22 commits, one branch (`main`), no tags, no releases, **no implementation source code in any commit**
- checksummed sources verified: 6

| source | role | sha256 |
|---|---|---|
| `paper_pdf` | paper_mathematics | `28a6ac520eade33e...` |
| `official_repo_accuracies` | official_repo_numerics | `ccbabdf8fe4d0a83...` |
| `official_repo_settings` | official_repo_numerics | `8494f87534ea0000...` |
| `official_repo_readme` | official_repo_numerics | `2cc3e42ef5c33826...` |
| `paper_table3` | printed_table | `81ef350a29615b36...` |
| `fig9_caption` | figure_caption | `052fae55c45809f6...` |
| `paper_table4` | printed_table | `719981377e32771d...` |
| `paper_table5` | printed_table | `706693f2fc806c8a...` |
| `paper_table6` | printed_table | `fff97db819df635a...` |

## Results by research question

| paper item | section | status | note |
|---|---|---|---|
| Table 1 (framework comparison) | 2 | `exact` | verbatim transcription; no computation involved |
| Theorem 1 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Theorem 2 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Theorem 3 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Theorem 4 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Theorem 5 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Theorem 6 | 2 | `exact` | implemented and unit-tested against analytical/quadrature references |
| Fig. 1 (conceptual hierarchy) | 2 | `exact` | redrawn from the model graph; encodes independent domains, dependent subdomains via shared (mu, nu), observed (C, N), OP-weighted aggregation, multiple LLMs |
| Fig. 2 (detailed hierarchical structure) | 2 | `exact` | redrawn from the model graph; encodes independent domains, dependent subdomains via shared (mu, nu), observed (C, N), OP-weighted aggregation, multiple LLMs |
| Table 2 (coin flip) | 4 | `inconsistent_source_artifact` | precise column reproduces exactly (0.36); the printed imprecise interval [0.31, 0.38] does not follow from the printed credal set, which gives [0.29, 0.43] (D9) |
| Table 3 (published accuracies) | 6 | `exact` | reproduced verbatim as its own dataset; NOT merged with the repository numerics (D1) |
| Fig. 3 (subdomain CDF envelopes) | 7 | `reconstructed_with_labelled_assumption` | freshly computed; qualitative orderings match the paper's narrative exactly |
| Fig. 4 (domain CDF envelopes) | 7 | `reconstructed_with_labelled_assumption` | freshly computed from the published measurements |
| Fig. 5 (LLM CDF envelopes) | 7 | `reconstructed_with_labelled_assumption` | freshly computed from the published measurements |
| Fig. 6 (hyperparameter sensitivity) | 8 | `reconstructed_with_labelled_assumption` | exact intervals from the paper; d controls width most (confirmed), but d also shifts location more than a or b, so the paper's a/b-location vs c/d-width split is only approximate (see D10) |
| Fig. 7 (OP sensitivity) | 9 | `reconstructed_with_labelled_assumption` | reproduced from Table 3, which is the only source consistent with the published panel (D4) |
| Fig. 8 (expected future reliability) | 10 | `reconstructed_with_labelled_assumption` | E[p^n] computed per configuration then enveloped; monotonicity verified automatically |
| Tables 4 and 5 (RQ5 baselines) | 11 | `blocked_by_missing_source_information` | ground-truth theta and OP vectors, BB-Inf prior strength, HiBayES priors and seeds are all unstated; two printed Table 5 rows are additionally mathematically inconsistent (D3) |
| RQ5 | 11 | `blocked_by_missing_source_information` | 7 of 7 required inputs unresolved |
| Fig. 9 (Pass@1 vs Pass@3) | 12 | `reconstructed_with_labelled_assumption` | aggregate_reproduction: per-task binary outcomes are not published, so this is not a complete benchmark re-run; captioned values conflict with both accuracy sources (D2) |
| Fig. 10a (memory growth) | 13 | `not_attempted` | requires Mode B; deliberately NOT synthesised from a formula |
| Fig. 10b-d (memory stress test) | 13 | `reconstructed_with_labelled_assumption` | sensitivity injection reproduced; component A (real same-session run) not attempted in Mode A and deliberately not fabricated |
| Fig. 11 (scalability) | 14 | `faithful_contemporary_rerun` | absolute timings are machine-specific; the reproducible claims are the scaling exponents and the stage breakdown |
| Table 6 (peak memory) | 14 | `faithful_contemporary_rerun` | paper values retained as reference only, never emitted as measurements |
| RQ1 (effectiveness) | 7 | `reconstructed_with_labelled_assumption` | Figs. 3-5 reproduced; model orderings and overlaps match the paper's narrative |
| RQ2 (hyperparameter sensitivity) | 8 | `reconstructed_with_labelled_assumption` | Fig. 6 reproduced with the paper's exact intervals; d dominates envelope WIDTH as the paper states, but d also shifts LOCATION more than a or b, so the paper's clean a/b-location vs c/d-width split is only approximate here (see D10) |
| RQ3 (OP sensitivity) | 9 | `reconstructed_with_labelled_assumption` | Fig. 7 reproduced from Table 3 (D4); GPT-4o strongly OP-sensitive, Sonnet 4.5 not |
| RQ4 (predictability) | 10 | `reconstructed_with_labelled_assumption` | Fig. 8 reproduced; E[R(1)] range and the bell-shaped width peak both match |
| RQ6 (failure definitions) | 12 | `reconstructed_with_labelled_assumption` | Fig. 9 reproduced from captioned aggregates only (aggregate_reproduction) |
| RQ7 (memory effects) | 13 | `reconstructed_with_labelled_assumption` | stress test reproduced; the real same-session experiment needs Mode B |
| RQ8 (scalability) | 14 | `faithful_contemporary_rerun` | measured on this machine; exponents and stage breakdown are the reproducible claims |

## Source conflicts

### D1 (high) — Paper Table 3 vs official repository figure numerics

12 of 16 accuracy cells differ. Only the Haiku 3.5 column agrees exactly on all four subdomains. The largest gap is GPT-4o on RACE-H (0.552 printed vs 0.920 in the repository, |diff| = 0.368).

**Resolution.** Keep both datasets. Figures use official_repo_accuracies; the literal Table 3 reproduction uses paper_table3. Never merged, never auto-swapped.

### D2 (medium) — Fig. 9 caption values conflict with both accuracy sources

Fig. 9 reports Claude Sonnet 4.5 on MBPP with N = 257 and Pass@1 C/N = 0.471. The repository gives Sonnet/MBPP = 0.486 and Table 3 gives 0.450. Curiously 0.471 is exactly Table 3's GPT-4o MBPP entry. The baseline figures use N = 80, not 257.

**Resolution.** Treat Fig. 9 as a figure-specific experiment and use its captioned values for Fig. 9 only. Its partner subdomain (DS-1000) data is not stated by the caption; the reconstruction used is recorded in the config and labelled.

### D3 (high) — Table 5 HIP-LLM rows are mathematically inconsistent

Under OP^approx the printed median envelope [0.5752, 0.5764] lies entirely BELOW the printed 90% interval [0.5794, 0.5978]; under OP^GT the median envelope [0.5784, 0.5797] lies entirely below [0.5832, 0.6024]. A posterior median cannot fall outside that posterior's own 5%-95% interval, and a min/max envelope over a family of posteriors cannot either, because min_h Q05 <= min_h Q50 by monotonicity of quantiles within each posterior.

**Resolution.** Reproduce the printed table verbatim as a reference artifact, flag the two rows with an automated invariant test, and never treat those numbers as a numerical target for a correct implementation.

### D4 (high) — Fig. 7 uses Table 3 values, contradicting the repository's 'all figures' claim

numerics/README.md states "All figures use these numerics unless otherwise noted", yet Fig. 7 is driven by Table 3, not by the repository CSV.

**Resolution.** Fig. 7 is reproduced from paper_table3 and Figs. 3-6, 8, 10 from official_repo_accuracies. The per-figure assignment is explicit in configs/paper_published_numerics.yaml :: figure_source_assignment.

### D5 (low) — Table 3 printed LLM mean for GPT-4o disagrees with its own printed cells

The printed GPT-4o "LLM Mean" is 0.585, but the unweighted mean of that column's four printed subdomain accuracies is (0.471 + 0.420 + 0.909 + 0.552)/4 = 0.588. The other three columns are self-consistent to three decimals.

**Resolution.** Reported as an internal inconsistency of the printed table; not used as an input.

### D6 (medium) — PERTURBATION 0.07 in settings.yaml vs +/-20% in the paper

settings.yaml declares sampling.PERTURBATION = 0.07 among settings said to apply to all figures; the paper's only perturbation is RQ5's OP^approx at +/-20%. No repository code consumes PERTURBATION, so its semantics cannot be verified.

**Resolution.** Recorded, and deliberately unused. hip_llm.operational_profile. perturb_and_renormalise takes the magnitude as an explicit argument and has no default.

### D7 (informational) — Editorial: 'five research questions' followed by RQ1-RQ8

Section 4 opens with "we empirically investigate five research questions (RQs)" (p. 10) and then enumerates and answers eight (RQ1-RQ8, Sections 4.3.1-4.3.8). Also on p. 20: "the 5 gaps identified in the introduction", consistent with Gap-1..Gap-5.

**Resolution.** Recorded; no numerical impact.

### D8 (informational) — SciPy is required by this reproduction but is not a reported dependency

The paper reports Python 3.12.12, NumPy 2.0.2, pandas 2.2.2 and matplotlib 3.10.0 only. Stable evaluation of betaln/gammaln/logsumexp and the Gamma / Beta quantile functions needs SciPy.

**Resolution.** SciPy is pinned in requirements-lock.txt and declared an implementation dependency of the reproduction, not of the original experiment.

### D9 (medium) — Table 2's imprecise posterior-mean interval does not follow from its own credal set

Table 2 states prior Beta(alpha, beta) with alpha, beta in [1, 3], data n = 10 / k = 3, posterior Beta(3+alpha, 7+beta), and reports E[theta | D] in [0.31, 0.38]. The posterior mean is (3+alpha)/(10+alpha+beta), whose extrema over the printed rectangle are attained at the corners (alpha, beta) = (1, 3) and (3, 1), giving [4/14, 6/14] = [0.2857, 0.4286], i.e. [0.29, 0.43] to two decimals -- not [0.31, 0.38].

**Resolution.** Section 4 of the notebook computes the interval analytically from the printed credal set, reports [0.29, 0.43], and flags the printed [0.31, 0.38] as an inconsistent source artifact. The printed values are never used as a numerical target.

### D10 (medium) — Partial pooling is numerically inert under the paper's own baseline hyperprior box

With N = 80 per subdomain and hyper-hyper-parameters drawn uniformly from a, b in [1, 12] and c, d in [1, 25], the hyperposterior for the pooling strength nu concentrates near 1 (E[nu | data] ~= 1.0-1.7 for typical configurations), because E[nu] = c/d and the ratio of two independent Uniform[1, 25] draws has median ~ 1. A prior strength of nu ~ 1 against N = 80 observations produces essentially no shrinkage.

**Resolution.** Reported as a finding about the paper's chosen configuration, not as an implementation defect. The dependence structure is preserved exactly (shared latent draws) and is verified by tests in both regimes.

### D11 (high) — Random hyper-hyper-parameter sampling under-states the imprecise envelope

Paper Appendix A.2, Step 3, states that the posterior envelopes "are obtained by evaluating the closed-form posterior at the extremal corners of H_i or through numerical optimization if the extrema occur in the interior". The repository's settings.yaml instead specifies N_CONFIGS = 160 together with a dedicated sampling seed (seeds.configs = 123), which describes a random design. The two prescriptions are not equivalent, and the difference is not a rounding effect.

**Resolution.** All four candidate designs are implemented and compared in Section 6 of the notebook; none is presented as the authors' rule. The consequence is stated explicitly: reported envelope WIDTHS should be read as a lower bound on the imprecision, whereas reported LOCATIONS (medians, orderings, OP and hyperparameter effects) are robust across every design tested.

## Numerical diagnostics

- Mode A runtime: 20.6 s for 4 models (K=160/domain, S=3000, G=2000)
- hyperposterior cache: {'hits': 0, 'misses': 1280, 'hit_rate': 0.0, 'entries': 1280}
- Monte-Carlo standard error of a posterior median at S=3000: ~0.02288 x posterior sd
- convergence in S, G and K is asserted by `tests/test_reproducibility.py` (slow marker) within the pre-declared tolerances
- grid sensitivity (nu-axis): max median shift 0.00129, max CDF sup-norm 0.03333
- configuration-sampling sensitivity: max median shift 0.01022, max CDF sup-norm 0.17667 (exceeds tolerance; see D11)
- corner-vs-random envelope area ratio: 1.68x
- seed sensitivity: asserted < 0.005 by `test_seed_sensitivity_stays_within_tolerance`
- measured RQ8 time exponents: m=1.09, n_bar=0.62, S=0.09, K=0.90, G=0.33
- offset-corrected time exponents: m=1.28, n_bar=1.09, S=0.09, K=0.90, G=0.33
- measured memory exponents: m=0.83, n_bar=0.53, S=1.00, K=0.43, G=-0.00
- exponents whose 95% bootstrap CI EXCLUDES the paper's printed value in THIS run: ['n_bar', 'G']. The CI resamples repeat timings within one run and so understates run-to-run variation; independent executions of this notebook moved individual exponents by up to ~0.6 for the parameters with the least dynamic range. Which exponents land inside the CI therefore varies between runs and should not be read as a significance test.
- exponents with the least dynamic range in this run (and hence the least reliable): ['S', 'G']. The parameters that vary the runtime by a large factor -- typically m, n_bar and K -- are measured near the paper's values in every run performed.
- baseline wall-clock: 2.38s here vs ~59s reported by the paper on Colab (~25x faster), so the stage split differs: Subdomain posteriors 59.7%, Domain/LLM envelopes 40.3% vs the paper's >99% / <1%

## Live API diagnostics

Mode B was not run. No provider was contacted, no tokens were consumed and no cost was incurred.

| field | value |
|---|---|
| exact model identifier | n/a (not run) |
| API date | n/a |
| request count | 0 |
| failure count | 0 |
| retries | 0 |
| total token usage | 0 |
| estimated cost | $0.00 |
| benchmark completion rate | n/a |

## Implementation limitations

- The nu-axis construction and the K-configuration sampling rule are reconstructions, not recoveries. Strict-exact mode refuses to run without them.
- RQ5 is blocked; its estimators are implemented and tested but cannot be pointed at the paper's ground truth.
- RQ7's real same-session experiment requires live API access.
- RQ8 absolute timings are machine-specific; only exponents and the stage breakdown transfer.
- Fig. 9 is an aggregate reproduction: per-task Pass@1/Pass@3 outcomes are unpublished.

## Final claim

**Exact statistical reproduction from published measurements** — with two labelled reconstructions (the nu-axis construction and the K-configuration sampling rule), RQ5 blocked, and the real same-session RQ7 experiment not attempted.

This claim is deliberately weaker than *exact historical end-to-end reproduction*, which is unreachable: no model snapshot, prompt, generation setting, dataset split, item subset, evaluator version, seed, integration grid or configuration-generation rule from the original experiments has been recovered, because the paper does not state them and the official repository contains no code.
