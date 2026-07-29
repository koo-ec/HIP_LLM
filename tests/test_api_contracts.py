"""Live-API contract tests (specification items 36-40).

Provider fixtures are used **only** to exercise schema parsing.  No fixture ever
becomes an experimental result: the tests that would produce a benchmark outcome
are marked ``live`` and hit the real API, where a missing key or an unavailable
snapshot is a FAILURE, not a skip.
"""

from __future__ import annotations

import os

import pytest

from hip_llm.api_clients import (
    GenerationSettings,
    LLMClient,
    MissingSettingError,
    ModelUnavailableError,
    ProviderError,
    ResponseCache,
    RunLabel,
    build_client,
    validate_snapshot_is_immutable,
)
from hip_llm.benchmark_eval import (
    FailureKind,
    SandboxPolicy,
    SandboxRefusedError,
    evaluate_mbpp_task,
    extract_python_code,
    parse_boolq_answer,
    parse_race_answer,
    pass_at_k,
)
from hip_llm.schemas import BenchmarkResult, load_yaml

SETTINGS = GenerationSettings(
    system_prompt="You are a careful assistant.",
    prompt_template="{question}",
    temperature=0.0,
    top_p=1.0,
    max_output_tokens=256,
    n_generations=1,
    seed=7,
    timeout_seconds=30.0,
    max_retries=0,
)


# --- 36. fixtures parse, but are never results ------------------------------ #
class _FixtureClient(LLMClient):
    """Schema-parsing only.  Returns a canned payload marked as a fixture."""

    provider = "fixture"

    def _list_models(self):
        return ["fixture-model-2026-01-01"]

    def _call(self, messages, index):
        return {
            "text": "Answer: yes",
            "resolved_model": self.snapshot,
            "finish_reason": "stop",
            "input_tokens": 11,
            "output_tokens": 3,
            "raw_request": {"messages": list(messages)},
            "raw_response": {"__fixture__": True, "id": "resp_fixture"},
        }


def test_fixture_response_parses_into_the_expected_schema():
    client = _FixtureClient("fixture-model-2026-01-01", SETTINGS)
    record = client.generate("Is the sky blue?")
    for key in ("text", "resolved_model", "input_tokens", "output_tokens", "raw_response"):
        assert key in record
    assert record["context_cleared"] is True
    assert record["raw_response"]["__fixture__"] is True


def test_a_fixture_response_can_never_be_mistaken_for_an_experimental_result():
    client = _FixtureClient("fixture-model-2026-01-01", SETTINGS)
    record = client.generate("Is the sky blue?")
    assert record["provider"] == "fixture"
    assert record["raw_response"].get("__fixture__") is True
    # Any BenchmarkResult built from fixtures must carry a non-provider snapshot,
    # which the notebook refuses to report.
    br = BenchmarkResult(
        benchmark="BoolQ", model_snapshot=record["resolved_model"],
        task_ids=("t0",), outcomes=(1,),
    )
    assert br.model_snapshot.startswith("fixture-")


# --- 37. costly tests are marked live --------------------------------------- #
@pytest.mark.live
def test_openai_snapshot_is_available():
    cfg = load_yaml(os.path.join("configs", "live_api.yaml"))
    snapshot = cfg["models"]["GPT-4o"]["snapshot"]
    assert snapshot, "configs/live_api.yaml must name an immutable GPT-4o snapshot"
    client = build_client("openai", snapshot, SETTINGS)
    client.validate_availability()


@pytest.mark.live
def test_anthropic_snapshot_is_available():
    cfg = load_yaml(os.path.join("configs", "live_api.yaml"))
    snapshot = cfg["models"]["Sonnet 4.5"]["snapshot"]
    assert snapshot, "configs/live_api.yaml must name an immutable Sonnet 4.5 snapshot"
    client = build_client("anthropic", snapshot, SETTINGS)
    client.validate_availability()


@pytest.mark.live
def test_live_generation_returns_a_real_provider_response():
    cfg = load_yaml(os.path.join("configs", "live_api.yaml"))
    snapshot = cfg["models"]["Sonnet 4.5"]["snapshot"]
    client = build_client("anthropic", snapshot, SETTINGS)
    client.validate_availability()
    record = client.generate("Reply with exactly the word: pong")
    assert record["raw_response"] and "__fixture__" not in record["raw_response"]
    assert record["resolved_model"]
    assert record["output_tokens"] > 0


# --- 38. missing keys / unavailable models FAIL, never skip ----------------- #
def test_missing_api_key_raises_rather_than_skipping(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_client("openai", "gpt-4o-2024-08-06", SETTINGS)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        client._connect()


def test_unavailable_snapshot_raises_and_never_substitutes():
    class _Narrow(_FixtureClient):
        def _list_models(self):
            return ["some-other-model-2026-01-01", "yet-another-2025-12-31"]

    client = _Narrow("required-snapshot-2026-01-01", SETTINGS)
    with pytest.raises(ModelUnavailableError, match="Refusing to substitute"):
        client.validate_availability()


def test_moving_aliases_are_rejected_in_strict_mode():
    for alias in ("gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5", "claude-3-5-haiku-latest"):
        with pytest.raises(MissingSettingError, match="moving alias"):
            validate_snapshot_is_immutable(alias, strict=True)
    for snapshot in ("gpt-4o-2024-08-06", "claude-sonnet-4-5-20250929"):
        assert validate_snapshot_is_immutable(snapshot, strict=True) == snapshot


def test_generation_settings_have_no_defaults():
    for missing in ("system_prompt", "temperature", "top_p", "max_output_tokens",
                    "n_generations", "seed", "timeout_seconds", "max_retries", "prompt_template"):
        cfg = {
            "system_prompt": "s", "prompt_template": "{q}", "temperature": 0.0, "top_p": 1.0,
            "max_output_tokens": 64, "n_generations": 1, "seed": 1,
            "timeout_seconds": 10.0, "max_retries": 0,
        }
        cfg.pop(missing)
        with pytest.raises(MissingSettingError, match="missing required key"):
            GenerationSettings.from_config(cfg)


def test_live_config_ships_with_unset_required_fields(root):
    """Shipping defaults for undisclosed settings would fake an exact reproduction."""
    cfg = load_yaml(root / "configs" / "live_api.yaml")
    assert cfg["run_label"] == RunLabel.CONTEMPORARY
    assert all(m["snapshot"] is None for m in cfg["models"].values())
    for key in ("system_prompt", "temperature", "top_p", "max_output_tokens", "prompt_template"):
        assert cfg["generation"][key] is None, key
    for bench in cfg["benchmarks"].values():
        assert bench["split"] is None and bench["prompt_template"] is None


def test_a_run_cannot_be_labelled_historical_exact(root):
    cfg = load_yaml(root / "configs" / "live_api.yaml")
    assert cfg["run_label"] != RunLabel.HISTORICAL_EXACT


# --- errors stay errors ----------------------------------------------------- #
def test_api_errors_are_never_converted_into_task_outcomes():
    class _AlwaysFails(_FixtureClient):
        def _call(self, messages, index):
            raise RuntimeError("503 upstream unavailable")

    client = _AlwaysFails("fixture-model-2026-01-01", SETTINGS)
    with pytest.raises(ProviderError, match="call failed"):
        client.generate("anything")
    assert client.usage.failures == 1
    assert client.usage.errors[0]["type"] == "RuntimeError"


def test_service_level_and_conditional_reliability_are_reported_separately():
    br = BenchmarkResult(
        benchmark="BoolQ", model_snapshot="fixture-model-2026-01-01",
        task_ids=tuple(f"t{i}" for i in range(10)), outcomes=(1,) * 9 + (0,), n_api_errors=5,
    )
    conditional = br.to_subdomain("BoolQ")
    service = br.service_level_subdomain("BoolQ")
    assert conditional.trials == 10 and conditional.successes == 9
    assert service.trials == 15 and service.successes == 9
    assert service.empirical_accuracy < conditional.empirical_accuracy


# --- caching only stores real responses ------------------------------------- #
def test_cache_refuses_a_record_without_a_real_response(tmp_path):
    cache = ResponseCache(tmp_path)
    with pytest.raises(ValueError, match="without a real provider response"):
        cache.put("k" * 64, {"text": "fabricated", "raw_response": None})


def test_cache_round_trips_a_real_record(tmp_path):
    cache = ResponseCache(tmp_path)
    key = ResponseCache.key("openai", "gpt-4o-2024-08-06", "fp", "prompt", 0)
    record = {"text": "hi", "raw_response": {"id": "resp_1"}, "input_tokens": 3}
    cache.put(key, record)
    assert cache.get(key)["raw_response"]["id"] == "resp_1"
    assert cache.hits == 1


def test_cache_key_is_sensitive_to_every_component():
    base = ("openai", "gpt-4o-2024-08-06", "fp", "prompt", 0)
    k = ResponseCache.key(*base)
    for i in range(5):
        changed = list(base)
        changed[i] = 1 if i == 4 else str(changed[i]) + "x"
        assert ResponseCache.key(*changed) != k


# --- 39, 40. context handling ------------------------------------------------ #
def test_iid_generation_starts_with_empty_conversation_context():
    client = _FixtureClient("fixture-model-2026-01-01", SETTINGS)
    record = client.generate("task 1")
    messages = record["raw_request"]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert record["context_cleared"] is True


def test_every_iid_task_is_independent():
    client = _FixtureClient("fixture-model-2026-01-01", SETTINGS)
    lengths = []
    for i in range(5):
        rec = client.generate(f"task {i}")
        lengths.append(len(rec["raw_request"]["messages"]))
    assert lengths == [1, 1, 1, 1, 1], "context leaked between i.i.d. tasks"


def test_memory_experiment_retains_context_within_one_session():
    client = _FixtureClient("fixture-model-2026-01-01", SETTINGS)
    history: list[dict[str, str]] = []
    turns, sizes = [], []
    for i in range(6):
        rec, history = client.generate_in_session(history, f"question {i}")
        assert rec["context_cleared"] is False
        turns.append(rec["history_turns"])
        sizes.append(rec["serialized_request_bytes"])
    assert turns == [1, 3, 5, 7, 9, 11]                 # strictly growing context
    assert all(b > a for a, b in zip(sizes, sizes[1:]))  # monotone byte growth


def test_session_and_iid_paths_are_distinct_methods():
    assert LLMClient.generate is not LLMClient.generate_in_session


# --- deterministic evaluators (never an LLM judge) -------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: yes", True), ("Answer: NO", False), ("**Answer:** Yes", True),
        ("yes, the passage says so", True), ("No.", False),
        ("True", True), ("false", False), ("", None), ("I am not sure", None),
    ],
)
def test_boolq_parsing_is_deterministic(text, expected):
    assert parse_boolq_answer(text) is expected
    assert parse_boolq_answer(text) is parse_boolq_answer(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Answer: C", 2), ("answer - (b)", 1), ("A) because ...", 0),
        ("The correct choice is D", 3), ("", None), ("no letter here", None),
    ],
)
def test_race_parsing_is_deterministic(text, expected):
    assert parse_race_answer(text) == expected


def test_pass_at_k_semantics():
    assert pass_at_k([True, False, False], 1) is True
    assert pass_at_k([False, False, False], 1) is False
    assert pass_at_k([False, False, True], 3) is True
    assert pass_at_k([False, False, False], 3) is False
    with pytest.raises(ValueError):
        pass_at_k([True], 3)


def test_pass1_uses_only_the_first_generation():
    """Paper footnote 20: Pass@1 produces one solution and scores that one."""
    assert pass_at_k([False, True, True], 1) is False


def test_code_extraction_is_deterministic():
    assert extract_python_code("```python\ndef f():\n    return 1\n```").startswith("def f")
    assert extract_python_code("def g(): return 2") == "def g(): return 2"
    assert extract_python_code("no code here at all") is None


# --- sandbox ---------------------------------------------------------------- #
def test_subprocess_backend_is_refused_by_default():
    with pytest.raises(SandboxRefusedError, match="weaker isolation"):
        SandboxPolicy(backend="subprocess")


def test_sandbox_executes_and_classifies_failures():
    policy = SandboxPolicy(backend="subprocess", allow_unsafe_subprocess=True, wall_time_s=20.0)

    ok = evaluate_mbpp_task("```python\ndef add(a,b):\n    return a+b\n```", ["assert add(1,2)==3"], policy)
    assert ok.passed and ok.failure is FailureKind.NONE

    bad = evaluate_mbpp_task("```python\ndef add(a,b):\n    return a-b\n```", ["assert add(1,2)==3"], policy)
    assert not bad.passed and bad.failure is FailureKind.ASSERTION_FAILED

    syntax = evaluate_mbpp_task("```python\ndef add(a,b)\n    return a+b\n```", ["assert True"], policy)
    assert not syntax.passed and syntax.failure is FailureKind.COMPILE_ERROR

    runtime = evaluate_mbpp_task("```python\ndef f():\n    return 1/0\n```", ["assert f()==1"], policy)
    assert not runtime.passed and runtime.failure is FailureKind.RUNTIME_ERROR

    unparseable = evaluate_mbpp_task("I cannot help with that.", ["assert True"], policy)
    assert not unparseable.passed and unparseable.failure is FailureKind.UNPARSEABLE


def test_sandbox_enforces_the_wall_clock_limit():
    policy = SandboxPolicy(backend="subprocess", allow_unsafe_subprocess=True, wall_time_s=3.0)
    out = evaluate_mbpp_task(
        "```python\nimport time\ndef f():\n    time.sleep(60)\n```", ["f()"], policy
    )
    assert out.failure is FailureKind.TIMEOUT and not out.passed


def test_sandbox_blocks_network_access():
    policy = SandboxPolicy(backend="subprocess", allow_unsafe_subprocess=True, wall_time_s=20.0)
    out = evaluate_mbpp_task(
        "```python\nimport socket\ndef f():\n    return socket.create_connection(('example.com', 80))\n```",
        ["f()"], policy,
    )
    assert not out.passed
    assert "network access is disabled" in out.stderr or out.failure is not FailureKind.NONE


def test_sandbox_scrubs_credentials_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-visible")
    monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
    policy = SandboxPolicy(backend="subprocess", allow_unsafe_subprocess=True, wall_time_s=20.0)
    out = evaluate_mbpp_task(
        "```python\nimport os\ndef f():\n    return [k for k in os.environ if 'KEY' in k or 'SECRET' in k]\n```",
        ["assert f() == [], f()"], policy,
    )
    assert out.passed, f"credentials leaked into the sandbox: {out.stderr}"
