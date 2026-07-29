# Quick start

## Install

Install the core package from PyPI:

```bash
pip install HIPLLM
```

For the Google Vertex AI integration used below:

```bash
pip install "HIPLLM[vertex]"
```

For a reproducible source checkout with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/koo-ec/HIP_LLM.git
cd HIP_LLM
uv sync --frozen --extra test
```

## Score generated responses

```python
from langchain_google_vertexai import ChatVertexAI
from HIPLLM import FailureProb

prompts = [
    "What is the capital of France?",
    "Explain why the sky appears blue.",
]

llm = ChatVertexAI(model="gemini-2.5-pro")
FP = FailureProb(llm=llm, scorers=["min_probability"])
results = await FP.generate_and_score(prompts=prompts)
results.to_df()
```

The DataFrame contains the prompt, generated response, confidence score and its
`failure_probability = 1 - min_probability` transformation.

## Supported scorers

- `min_probability`: minimum generated-token probability.
- `sequence_probability`: geometric mean of generated-token probabilities.

When one scorer is selected, the transformed value is named
`failure_probability`. With multiple scorers, each transformed column is named
`<scorer>_failure_probability` so that its meaning remains explicit.
