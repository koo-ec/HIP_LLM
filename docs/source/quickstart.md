# Quick start

## Install

Install the released core package from PyPI:

```bash
pip install HIPLLM
```

To use functionality that has not yet been released to PyPI, install the current repository revision:

```bash
pip install "git+https://github.com/koo-ec/HIP_LLM.git@main"
```

For the optional Google Vertex AI integration:

```bash
pip install "HIPLLM[vertex]"
```

For a reproducible source checkout with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/koo-ec/HIP_LLM.git
cd HIP_LLM
uv sync --frozen --extra test
```

## Estimate failure under an operational profile

The operational-profile calculation requires labelled binary outcomes and a target workload distribution. The profile is explicit and must sum to one.

```python
from HIPLLM import OperationalFailureProb, quick_inference_settings

outcomes = [1, 1, 0, 1, 0, 0, 1, 0]
strata = ["short"] * 4 + ["long"] * 4
profile = {"short": 0.30, "long": 0.70}

estimator = OperationalFailureProb(
    profile=profile,
    settings=quick_inference_settings(samples=1500, configurations=48),
)
result = estimator.fit(outcomes=outcomes, strata=strata)

result.summary()
result.to_df()
```

`summary()` reports:

- the direct operational-profile-weighted observed failure rate;
- lower and upper posterior expected-failure bounds;
- lower and upper posterior median-failure bounds; and
- an outer credible envelope across all admissible prior configurations.

Use `paper_inference_settings()` instead of `quick_inference_settings()` when the paper-sized numerical settings are required. The unresolved finite-grid and configuration-sampling choices remain labelled reconstruction assumptions.

## Run StrategyQA with OpenAI

The Colab notebook at `notebooks/strategyqa_openai_operational_profile_colab.ipynb` performs the complete workflow:

1. load the official labelled StrategyQA development split;
2. send independent yes/no questions to a pinned OpenAI model snapshot;
3. parse answers deterministically and compare them with ground truth;
4. define an operational profile over decomposition-length strata; and
5. estimate operational failure probability and uncertainty bounds.

The notebook expects `OPENAI_API_KEY` in the environment and writes resumable prediction and summary files.

## Token-confidence scoring is different

The legacy `FailureProb` interface transforms provider token probabilities:

```python
from langchain_google_vertexai import ChatVertexAI
from HIPLLM import FailureProb

prompts = [
    "What is the capital of France?",
    "Explain why the sky appears blue.",
]

llm = ChatVertexAI(model="gemini-2.5-pro")
scorer = FailureProb(llm=llm, scorers=["min_probability"])
scores = await scorer.generate_and_score(prompts=prompts)
scores.to_df()
```

The output column called `failure_probability` is `1 - token confidence`. It does not use an operational profile and should not be interpreted as a calibrated factual-error probability without task-specific labelled calibration.

Supported token-confidence scorers are:

- `min_probability`: minimum generated-token probability;
- `sequence_probability`: geometric mean generated-token probability.

When one scorer is selected, the transformed value is named `failure_probability`. With multiple scorers, each transformed column is named `<scorer>_failure_probability`.
