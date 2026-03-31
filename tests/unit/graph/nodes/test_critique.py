from unittest.mock import MagicMock, AsyncMock, patch
from app.graph.nodes.critique import critique_node, CritiqueOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(
    parsed: CritiqueOutput | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MagicMock:
    """Mock get_llm() that supports .with_structured_output(...).ainvoke()."""
    raw = MagicMock()
    raw.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}

    structured_chain = MagicMock()
    structured_chain.ainvoke = AsyncMock(return_value={
        "raw": raw,
        "parsed": parsed,
        "parsing_error": None if parsed else ValueError("parse failed"),
    })

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_chain)
    return llm


def _base_state(**overrides) -> dict:
    state = {
        "code_snippet": "int x = 0;",
        "retrieved_rules": [{"rule_id": "MISRA_1.1", "title": "No dead code"}],
        "validation_result": "Code is compliant with all retrieved rules.",
        "cited_rules": ["MISRA_1.1"],
        "is_compliant": True,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_approved_sets_critique_approved_true():
    parsed = CritiqueOutput(approved=True, feedback="Pass")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state())

    assert result["critique_approved"] is True
    assert result["critique_feedback"] == "Pass"


async def test_rejected_sets_critique_approved_false_with_feedback():
    parsed = CritiqueOutput(approved=False, feedback="Criteria 1 failed: hallucinated rule.")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state())

    assert result["critique_approved"] is False
    assert "Criteria 1 failed" in result["critique_feedback"]


async def test_parse_failure_sets_not_approved():
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed=None)):
        result = await critique_node(_base_state())

    assert result["critique_approved"] is False
    assert isinstance(result["critique_feedback"], str)
    assert len(result["critique_feedback"]) > 0


async def test_returns_only_expected_state_keys():
    parsed = CritiqueOutput(approved=True, feedback="Pass")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state())

    assert {"critique_approved", "critique_feedback"}.issubset(result.keys())


async def test_empty_retrieved_rules_no_crash():
    parsed = CritiqueOutput(approved=True, feedback="Pass")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state(retrieved_rules=[]))

    assert result["critique_approved"] is True


async def test_non_compliant_state_no_crash():
    parsed = CritiqueOutput(approved=False, feedback="Logical inconsistency.")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(
            _base_state(is_compliant=False, validation_result="Recursion found.")
        )

    assert result["critique_approved"] is False


async def test_get_llm_called_with_zero_temperature():
    parsed = CritiqueOutput(approved=True, feedback="Pass")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)) as mock_get_llm:
        await critique_node(_base_state())

    mock_get_llm.assert_called_once_with(temperature=0.0)


async def test_with_structured_output_called_with_correct_schema():
    mock = _mock_llm(CritiqueOutput(approved=True, feedback="Pass"))
    with patch("app.graph.nodes.critique.get_llm", return_value=mock):
        await critique_node(_base_state())

    mock.with_structured_output.assert_called_once_with(CritiqueOutput, include_raw=True)


async def test_critique_history_populated_on_approval():
    parsed = CritiqueOutput(approved=True, feedback="Pass")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state(iteration_count=2))

    assert len(result["critique_history"]) == 1
    entry = result["critique_history"][0]
    assert entry["approved"] is True
    assert entry["issues_found"] == []
    assert entry["iteration"] == 2


async def test_critique_history_populated_on_rejection():
    parsed = CritiqueOutput(approved=False, feedback="Rule hallucination detected.")
    with patch("app.graph.nodes.critique.get_llm", return_value=_mock_llm(parsed)):
        result = await critique_node(_base_state(iteration_count=1))

    entry = result["critique_history"][0]
    assert entry["approved"] is False
    assert "Rule hallucination detected." in entry["issues_found"]
