#!/usr/bin/env python
"""Assemble ``HIP_LLM_pfd_quickstart.ipynb``.

A user-facing quickstart: run a small Q&A pipeline against a real OpenAI
snapshot, then wrap the outcomes in HIP-LLM to get a probability-of-failure
envelope.  Generated rather than hand-edited so it can never carry stale output.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HIP_LLM_pfd_quickstart.ipynb"

CELLS: list[tuple[str, str, list[str]]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source, []))


def code(source: str, tags: list[str] | None = None) -> None:
    CELLS.append(("code", source, tags or []))


# --------------------------------------------------------------------------- #
md(r"""# HIP-LLM quickstart — probability of failure for your own Q&A pipeline

Run a small Q&A evaluation against a real OpenAI model, then wrap the outcomes in
HIP-LLM to get a **probability of failure** with an imprecise-probability envelope.

### What you get

| Quantity | Meaning |
|---|---|
| `pfd` median envelope | the range of plausible median failure probabilities across admissible priors |
| `pfd` 90% interval | the widest credible interval compatible with that prior uncertainty |
| `P(≥1 failure in n)` | probability of at least one failure over the next `n` tasks |

### The one conversion to remember

HIP-LLM models **non-failure** probability $p_L$. So

$$\mathrm{pfd} = 1 - p_L, \qquad R(n_F) = p_L^{\,n_F}, \qquad
P(\ge 1 \text{ failure in } n_F) = 1 - \mathbb{E}[R(n_F)].$$

Because subtracting flips order, quantile envelopes swap ends:
`pfd_median_low = 1 - p_L_median_high`. The cells below always build the `pfd`
samples first and derive everything from those, so the flip happens once.

### What you must decide (these are modelling inputs, not defaults)

1. **Subdomains** — the task types you want reliability broken down by.
2. **`OMEGA`** — your *operational profile*: how often each task type occurs in
   real usage. Not your dataset sizes. This is the whole point of the framework.
3. **`W`** — weights across domains, if you have more than one.

Set `LIVE = False` to check the plumbing with stub outcomes and no API cost.""")

code(
    '''# --- parameters ------------------------------------------------------------
LIVE = False                       # True = real OpenAI calls (costs money)
MODEL_SNAPSHOT = "gpt-4o-2024-08-06"   # pin a DATED snapshot, not "gpt-4o"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 16
SEED = 7                           # OpenAI honours `seed` on a best-effort basis

# HIP-LLM numerical settings (the paper's baseline)
K_CONFIGS = 160                    # admissible prior configurations
S_SAMPLES = 3000                   # Monte-Carlo draws per configuration
PACKAGE_DIR = None                 # None = this notebook sits inside the package''',
    tags=["parameters"],
)

code(r'''import json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(PACKAGE_DIR) if PACKAGE_DIR else Path.cwd()
if not (ROOT / "src" / "hip_llm").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "src" / "hip_llm").is_dir(), f"could not find src/hip_llm/ from {Path.cwd()}"
sys.path.insert(0, str(ROOT / "src"))

from hip_llm.schemas import (
    DomainData, GlobalSettings, HyperparameterInterval, ModelResult, SubdomainData,
)
from hip_llm.posterior import run_model
from hip_llm.envelopes import cdf_envelope, default_t_grid, quantile_envelope
from hip_llm.reliability import expected_reliability_envelope
from hip_llm.benchmark_eval import parse_boolq_answer, parse_race_answer
from hip_llm import plotting as P

P.apply_house_style()
print(f"package: {ROOT}")
print(f"mode   : {'LIVE (real API calls)' if LIVE else 'OFFLINE (stub outcomes, no cost)'}")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 1. Define your tasks

Two task types, so the hierarchy has something to pool over. Replace these with
your own items — the only requirement is that each task has a **deterministic**
grader. Never use an LLM as the judge: the framework assumes a perfect test
oracle, and a noisy judge makes that assumption unverifiable.

The two parsers below (`parse_boolq_answer`, `parse_race_answer`) come from the
package and are unit-tested.""")

code(r'''TASKS = {
    "factual_yesno": [
        {"q": "Is the Pacific Ocean the largest ocean on Earth?", "gold": True},
        {"q": "Is Mount Everest located in the Andes?", "gold": False},
        {"q": "Does water boil at 100 degrees Celsius at sea level?", "gold": True},
        {"q": "Is Portuguese the official language of Brazil?", "gold": True},
        {"q": "Is the Sahara located in South America?", "gold": False},
        {"q": "Was the Magna Carta signed in 1215?", "gold": True},
        {"q": "Is Venus the closest planet to the Sun?", "gold": False},
        {"q": "Do adult humans normally have 206 bones?", "gold": True},
    ],
    "reasoning_mcq": [
        {"q": "A train leaves at 09:00 and arrives at 11:30. How long is the trip?\n"
              "A) 1h30  B) 2h00  C) 2h30  D) 3h00", "gold": 2},
        {"q": "If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are:\n"
              "A) Lazzies  B) not Lazzies  C) sometimes Lazzies  D) undetermined", "gold": 0},
        {"q": "What is 15% of 240?\nA) 24  B) 36  C) 42  D) 48", "gold": 1},
        {"q": "Sequence: 2, 6, 12, 20, 30, ?\nA) 36  B) 40  C) 42  D) 44", "gold": 2},
        {"q": "A shirt costs 40 after a 20% discount. Original price?\n"
              "A) 44  B) 48  C) 50  D) 52", "gold": 2},
        {"q": "Which is heaviest?\nA) 1 kg feathers  B) 1000 g lead  C) equal  D) unknown",
         "gold": 2},
    ],
}

SYSTEM_PROMPT = ("Answer with only the final answer. For yes/no questions reply "
                 "exactly 'Yes' or 'No'. For multiple choice reply with the single "
                 "option letter.")

for name, items in TASKS.items():
    print(f"{name:16s} {len(items)} tasks")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 2. Run the pipeline

Each task is an **independent call with cleared context** — that is what makes the
i.i.d. Bernoulli assumption defensible. Never reuse a conversation here.

API errors are recorded separately and never silently counted as task failures,
so a flaky network cannot masquerade as an unreliable model.""")

code(r'''def grade(name: str, response: str, gold) -> int | None:
    """Deterministic grader. Returns 1, 0, or None if the reply is unparseable."""
    if name == "factual_yesno":
        parsed = parse_boolq_answer(response)
    else:
        parsed = parse_race_answer(response, n_options=4)
    if parsed is None:
        return None
    return int(parsed == gold)


def ask_openai(question: str) -> str:
    """One generation with cleared context against the pinned snapshot."""
    from openai import OpenAI

    kwargs = dict(model=MODEL_SNAPSHOT, temperature=TEMPERATURE,
                  max_tokens=MAX_OUTPUT_TOKENS,
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question}])
    if SEED is not None:
        kwargs["seed"] = SEED
    resp = OpenAI().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""''')

code(r'''def run_evaluation(tasks: dict, live: bool) -> tuple[dict, dict, list]:
    """Return (outcomes, unparseable counts, error records)."""
    outcomes, unparseable, errors = {}, {}, []
    rng = np.random.default_rng(20260729)
    stub_rate = {"factual_yesno": 0.90, "reasoning_mcq": 0.72}

    for name, items in tasks.items():
        results, bad = [], 0
        for i, item in enumerate(items):
            if not live:
                results.append(int(rng.random() < stub_rate[name]))
                continue
            try:
                text = ask_openai(item["q"])
            except Exception as exc:                 # API errors stay errors
                errors.append({"task": name, "index": i, "error": repr(exc)[:200]})
                continue
            g = grade(name, text, item["gold"])
            if g is None:
                bad += 1
            else:
                results.append(g)
        outcomes[name] = results
        unparseable[name] = bad
    return outcomes, unparseable, errors


t0 = time.perf_counter()
if LIVE:
    assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY before running LIVE"
OUTCOMES, UNPARSEABLE, ERRORS = run_evaluation(TASKS, LIVE)
print(f"evaluation finished in {time.perf_counter() - t0:.1f}s\n")

for name, res in OUTCOMES.items():
    C, N = int(sum(res)), len(res)
    print(f"  {name:16s} C/N = {C:3d}/{N:3d} = {C/N:.4f}"
          f"   unparseable: {UNPARSEABLE[name]}   API errors: "
          f"{sum(1 for e in ERRORS if e['task'] == name)}")
if ERRORS:
    print(f"\n{len(ERRORS)} API error(s) recorded separately -- NOT counted as failures.")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 3. Declare the hierarchy and your operational profile

`OMEGA` is the modelling decision that matters most. Set it to how often each task
type occurs in **your** deployment, not to how many test items you happen to have.
If you use dataset proportions instead, you are reproducing the paper's own
`OP^data` mismatch scenario, which it shows increases error.

Weights must sum to 1; the schema rejects anything else.""")

code(r'''OMEGA = {"QA": [0.65, 0.35]}     # factual_yesno, reasoning_mcq -- YOUR usage mix
W = [1.0]                        # a single domain

subdomains = tuple(
    SubdomainData(name=name, successes=int(sum(res)), trials=len(res))
    for name, res in OUTCOMES.items()
)
domain = DomainData("QA", subdomains, np.asarray(OMEGA["QA"], dtype=float))
MODEL = ModelResult("my-qa-pipeline", (domain,), np.asarray(W, dtype=float),
                    source_label=f"live:{MODEL_SNAPSHOT}" if LIVE else "offline stub")

SETTINGS = GlobalSettings(
    n_mu=40, n_nu=50, cdf_points_T=201, S=S_SAMPLES, K_per_domain=K_CONFIGS,
    max_llm_configuration_pairs=512, seed_global=7, seed_configs=123, seed_pairs=999,
    config_sampling="uniform_random", nu_grid_scheme="log", mu_grid_scheme="midpoint",
)
INTERVAL = HyperparameterInterval(a=(1, 12), b=(1, 12), c=(1, 25), d=(1, 25))

print(f"config hash: {SETTINGS.hash()[:16]}...   K={K_CONFIGS}, S={S_SAMPLES}")
for s, w in zip(subdomains, OMEGA["QA"]):
    print(f"  {s.name:16s} C/N={s.successes}/{s.trials}  Omega={w}")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 4. Fit and read off the probability of failure

`pfd` samples are built once, up front, so the order flip happens in exactly one
place and every number below inherits it correctly.""")

code(r'''domain_sets, llm = run_model(MODEL, [INTERVAL], SETTINGS)

p_L = llm.p_L                 # (n_configs, S) non-failure probability
PFD = 1.0 - p_L               # probability of failure on the next task

med_lo, med_hi = quantile_envelope(PFD, 0.50)
q05_lo, _ = quantile_envelope(PFD, 0.05)
_, q95_hi = quantile_envelope(PFD, 0.95)

print(f"OVERALL PROBABILITY OF FAILURE  (model: {MODEL.source_label})\n")
print(f"  median envelope : [{med_lo:.4f}, {med_hi:.4f}]")
print(f"  90% interval    : [{q05_lo:.4f}, {q95_hi:.4f}]")
print(f"  worst case      :  {q95_hi:.4f}   <- use this for conservative claims")
print(f"\n  over {llm.n_configs} admissible prior configurations x {llm.n_samples} draws")''')

code(r'''rows = []
for name in OUTCOMES:
    theta = domain_sets[0].subdomain_samples(name)
    lo, hi = quantile_envelope(1.0 - theta, 0.50)
    c05, _ = quantile_envelope(1.0 - theta, 0.05)
    _, c95 = quantile_envelope(1.0 - theta, 0.95)
    rows.append({"subdomain": name, "pfd_median_lo": lo, "pfd_median_hi": hi,
                 "pfd_90ci_lo": c05, "pfd_90ci_hi": c95})
rows.append({"subdomain": "OVERALL (OP-weighted)", "pfd_median_lo": med_lo,
             "pfd_median_hi": med_hi, "pfd_90ci_lo": q05_lo, "pfd_90ci_hi": q95_hi})

import pandas as pd
PFD_TABLE = pd.DataFrame(rows)
display(PFD_TABLE.round(4))''')

code(r'''horizons = np.array([1, 2, 5, 10, 20, 50], dtype=float)
rel = expected_reliability_envelope(p_L, horizons)

print("probability of at least one failure over the next n tasks\n")
print(f"  {'n':>4}   {'E[R(n)] envelope':^24}   {'P(>=1 failure)':^24}")
for n, lo, hi in zip(rel.horizons, rel.lower, rel.upper):
    print(f"  {int(n):>4}   [{lo:.4f}, {hi:.4f}]        [{1-hi:.4f}, {1-lo:.4f}]")
print("\nComputed as E[p^n] per configuration then enveloped -- never as E[p]^n, "
      "since those differ.")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 5. Plot the failure-probability envelope

The band is the set of posterior CDFs admitted by the imprecise prior. Read it as:
*for a threshold `t` on the x-axis, the probability that the true failure rate is
below `t` lies somewhere in the shaded band.*""")

code(r'''import matplotlib.pyplot as plt

t_grid = default_t_grid(201)
env = cdf_envelope(PFD, t_grid, quantity="pfd")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.fill_between(env.t_grid, env.lower, env.upper, color="#d62728", alpha=0.30,
                edgecolor="#d62728", linewidth=1.0, label="posterior CDF envelope")
ax.axvline(med_lo, color="#1f77b4", ls="--", lw=1.1, label=f"median envelope")
ax.axvline(med_hi, color="#1f77b4", ls="--", lw=1.1)
ax.axvline(q95_hi, color="#333333", ls=":", lw=1.2,
           label=f"conservative 95% bound = {q95_hi:.3f}")
ax.set_xlim(*P.auto_xlim({"pfd": env}))
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("probability of failure on the next task")
ax.set_ylabel("CDF")
ax.set_title(f"Failure-probability envelope — {MODEL.source_label}")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()''')

# --------------------------------------------------------------------------- #
md(r"""---
## 6. Two settings that change your answer

Both are findings from replicating the paper, and both apply to *your* numbers.

**(a) Pooling is inert at the default hyperprior box.** With `c, d ∈ [1, 25]` the
posterior for the pooling strength $\nu$ concentrates near 1, so against any
reasonable $N$ your subdomains barely inform each other. If you *want* the
subdomains to share strength, narrow the box.

**(b) The default envelope width is a lower bound.** The paper's own Appendix A.2
says envelopes come from the *extremal corners* of the admissible set, but 160
uniform draws in a 4-D box essentially never reach a corner. A corner-inclusive
design gives a wider — and more honest — envelope.""")

code(r'''def refit(interval, config_sampling="uniform_random"):
    st = GlobalSettings(**{**SETTINGS.__dict__, "config_sampling": config_sampling})
    _, out = run_model(MODEL, [interval], st)
    pfd = 1.0 - out.p_L
    lo, hi = quantile_envelope(pfd, 0.50)
    _, c95 = quantile_envelope(pfd, 0.95)
    return {"pfd_median_lo": lo, "pfd_median_hi": hi,
            "median_width": hi - lo, "pfd_95_worst_case": c95}


STRONG_POOLING = HyperparameterInterval(a=(1, 12), b=(1, 12), c=(20, 25), d=(1, 2))
variants = {
    "default (paper baseline)": refit(INTERVAL),
    "strong pooling (c=[20,25], d=[1,2])": refit(STRONG_POOLING),
    "corner-inclusive design": refit(INTERVAL, "interval_corners_plus_interior"),
}
SENSITIVITY = pd.DataFrame(variants).T
display(SENSITIVITY.round(4))

base = SENSITIVITY.loc["default (paper baseline)", "median_width"]
corner = SENSITIVITY.loc["corner-inclusive design", "median_width"]
print(f"\ncorner-inclusive median envelope is {corner / base:.2f}x wider than the default.")
print("If you are making a conservative reliability claim, prefer the wider one.")''')

# --------------------------------------------------------------------------- #
md(r"""---
## 7. Save the result

Everything needed to audit or repeat this run: counts, weights, seeds,
configuration hash, model snapshot and the resulting envelopes.""")

code(r'''OUT_DIR = Path.cwd() / "pfd_results"
OUT_DIR.mkdir(exist_ok=True)

summary = {
    "model_snapshot": MODEL_SNAPSHOT if LIVE else None,
    "mode": "live" if LIVE else "offline_stub",
    "temperature": TEMPERATURE,
    "counts": {n: {"C": int(sum(r)), "N": len(r)} for n, r in OUTCOMES.items()},
    "unparseable": UNPARSEABLE,
    "api_errors": len(ERRORS),
    "operational_profile": OMEGA,
    "domain_weights": W,
    "hyperparameter_box": {"a": INTERVAL.a, "b": INTERVAL.b,
                           "c": INTERVAL.c, "d": INTERVAL.d},
    "settings_hash": SETTINGS.hash(),
    "seeds": {"global": SETTINGS.seed_global, "configs": SETTINGS.seed_configs,
              "pairs": SETTINGS.seed_pairs},
    "pfd": {"median_envelope": [med_lo, med_hi],
            "interval_90": [q05_lo, q95_hi],
            "conservative_95": q95_hi},
}
(OUT_DIR / "pfd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PFD_TABLE.to_csv(OUT_DIR / "pfd_by_subdomain.csv", index=False)
SENSITIVITY.to_csv(OUT_DIR / "pfd_sensitivity.csv")
np.savez_compressed(OUT_DIR / "pfd_samples.npz", pfd=PFD)

print(f"written to {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(f"   {f.name}")
print(f"\nheadline: pfd median envelope [{med_lo:.4f}, {med_hi:.4f}], "
      f"conservative 95% bound {q95_hi:.4f}")''')

md(r"""---
### Caveats worth carrying forward

- **`OMEGA` drives the answer.** Reliability is usage-weighted by construction; a
  different task mix gives a different failure probability for the same model.
- **API errors are not task failures.** They are recorded separately above. If you
  need service-level reliability, count them as failures in a *separate* number
  and say which one you are quoting.
- **Small `N` gives a wide envelope.** That is the framework working correctly, not
  a defect. Add tasks to narrow it.
- **This is offline assessment.** It says nothing about drift after the snapshot
  you pinned changes.""")


def main() -> int:
    cells = []
    for i, (kind, source, tags) in enumerate(CELLS):
        cell = {
            "id": f"qs-{i:03d}",
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
