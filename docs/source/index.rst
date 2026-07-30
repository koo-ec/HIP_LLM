:hide-toc:

HIP-LLM
=======

.. raw:: html

   <section class="hip-hero" aria-label="HIP-LLM overview">
     <p class="hip-hero__kicker">Reliability assessment for large language models</p>
     <p class="hip-hero__title">Evidence-based LLM failure estimation under real workloads</p>
     <p class="hip-hero__summary">
       HIP-LLM combines labelled evaluation outcomes, explicit operational profiles,
       hierarchical Bayesian inference and imprecise priors to quantify large-language-model
       reliability without hiding unresolved assumptions.
     </p>
     <div class="hip-actions">
       <a class="hip-button hip-button--primary" href="quickstart/">Get started</a>
       <a class="hip-button hip-button--secondary" href="https://colab.research.google.com/drive/19H582xYhuThqVFcZQzgwEa-vv_XeuJZQ?usp=sharing">Open in Google Colab</a>
       <a class="hip-button hip-button--secondary" href="https://github.com/koo-ec/HIP_LLM">View on GitHub</a>
       <a class="hip-button hip-button--secondary" href="https://doi.org/10.1016/j.ress.2026.112615">Read the paper</a>
     </div>
   </section>

What HIP-LLM provides
---------------------

.. raw:: html

   <div class="hip-grid">
     <article class="hip-card">
       <span class="hip-card__mark">01</span>
       <h3>Explicit operational profiles</h3>
       <p>Represent how frequently each workload stratum occurs in the intended deployment rather than assuming a uniform test distribution.</p>
     </article>
     <article class="hip-card">
       <span class="hip-card__mark">02</span>
       <h3>Imprecise Bayesian bounds</h3>
       <p>Report lower and upper reliability estimates across admissible prior configurations instead of presenting one unjustifiably precise number.</p>
     </article>
     <article class="hip-card">
       <span class="hip-card__mark">03</span>
       <h3>Reproducible evidence</h3>
       <p>Connect code, labelled outcomes, provenance records, notebooks, generated results and reconstruction assumptions in one tested package.</p>
     </article>
   </div>

Use the correct reliability quantity
------------------------------------

HIP-LLM exposes two related but different interfaces:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Interface
     - What it estimates
     - Appropriate use
   * - ``OperationalFailureProb``
     - Failure under a declared operational profile, with uncertainty bounds across admissible priors.
     - Reliability assessment when labelled outcomes and a target workload distribution are available.
   * - ``FailureProb``
     - ``1 - token confidence`` from provider log probabilities.
     - Diagnostic analysis only, unless calibrated and validated against labelled target-task data.

.. raw:: html

   <div class="hip-callout">
     <p><strong>Important:</strong> token confidence is not automatically a calibrated probability of factual error. Use <code>OperationalFailureProb</code> for the operational-profile calculation described by HIP-LLM.</p>
   </div>

Minimal example
---------------

.. code-block:: python

   from HIPLLM import OperationalFailureProb, quick_inference_settings

   outcomes = [1, 1, 0, 1, 0, 0, 1, 0]
   strata = ["short"] * 4 + ["long"] * 4
   profile = {"short": 0.30, "long": 0.70}

   estimator = OperationalFailureProb(
       profile=profile,
       settings=quick_inference_settings(samples=1500, configurations=48),
   )
   result = estimator.fit(outcomes=outcomes, strata=strata)

   print(result.summary())

The method and replication package
----------------------------------

This implementation accompanies the reliability method described in:

R. Aghazadeh-Chakherlou, Q. Guo, S. Khastgir, P. Popov, X. Zhang and X. Zhao,
*“A hierarchical imprecise probability approach to reliability assessment of large
language models,”* Reliability Engineering & System Safety 272 (2026), 112615.
`DOI: 10.1016/j.ress.2026.112615 <https://doi.org/10.1016/j.ress.2026.112615>`_.

The package reproduces disclosed statistical inference from published measurements. It
does not claim an exact historical replay of undisclosed prompts, model snapshots,
generation settings, item subsets or random seeds. See :doc:`paper` for the documented
scope and provenance resources.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   quickstart
   api
   paper
