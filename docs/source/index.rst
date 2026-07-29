HIPLLM documentation
====================

HIPLLM provides explicit operational-profile failure inference from labelled
benchmark outcomes, together with the repository's hierarchical
imprecise-probability replication implementation. A separate LangChain-compatible
interface is available for token-confidence diagnostics.

.. warning::

   ``FailureProb`` token-confidence scores are heuristics. They do not use an
   operational profile and are not calibrated probabilities of factual error
   unless validated and calibrated on labelled target-task data. Use
   ``OperationalFailureProb`` for workload-level operational failure inference.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   paper
