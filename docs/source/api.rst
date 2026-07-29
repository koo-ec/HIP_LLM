API reference
=============

.. currentmodule:: HIPLLM

Operational-profile failure inference
-------------------------------------

.. autoclass:: OperationalFailureProb
   :members:

.. autoclass:: OperationalFailureResult
   :members:

.. autofunction:: quick_inference_settings

.. autofunction:: paper_inference_settings

StrategyQA utilities
--------------------

.. autofunction:: load_strategyqa

.. autofunction:: parse_strategyqa_answer

.. autofunction:: decomposition_stratum

.. autoexception:: StrategyQALoadError

Token-confidence diagnostics
----------------------------

.. autoclass:: FailureProb
   :members:

.. autoclass:: FailureProbResult
   :members:

.. autoexception:: LogprobsUnavailableError
