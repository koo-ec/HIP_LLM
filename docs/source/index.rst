HIPLLM documentation
====================

HIPLLM provides a small LangChain-compatible interface for token-confidence
failure scoring alongside the repository's hierarchical imprecise-probability
reliability implementation.

.. warning::

   Prompt-level token-confidence scores are heuristics. They are not calibrated
   probabilities of factual error unless validated and calibrated on labelled
   data from the target task.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   paper
