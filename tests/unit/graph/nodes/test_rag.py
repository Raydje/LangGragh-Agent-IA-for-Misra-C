# tests/unit/graph/nodes/test_rag.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.nodes.rag import rag_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(query: str = "check pointer arithmetic") -> dict:
    return {"query": query}


def _pinecone_result(*rule_ids_and_scores: tuple[str, float]) -> dict:
    """Build a minimal Pinecone response dict."""
    return {
        "matches": [
            {"id": rid, "score": score}
            for rid, score in rule_ids_and_scores
        ]
    }


def _mongo_doc(
    rule_id: str,
    *,
    section: str = "1.1",
    category: str = "N/A",
    title: str = "",
    full_text: str = "some rule text",
) -> dict:
    return {
        "rule_id": rule_id,
        "section": section,
        "category": category,
        "title": title,
        "full_text": full_text,
    }


FAKE_VECTOR = [0.1] * 768

MONGO_MOCK = "app.graph.nodes.rag.get_mongodb_service"
PINECONE_MOCK = "app.graph.nodes.rag.get_pinecone_service"


def _setup_pinecone_mock(mock_svc, result: dict) -> AsyncMock:
    """Attach an AsyncMock .query() to a get_pinecone_service mock and return it."""
    query_mock = AsyncMock(return_value=result)
    mock_svc.return_value.query = query_mock
    return query_mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_happy_path_returns_retrieved_rules(
    mock_embed, mock_pinecone_svc, mock_mongo_svc
):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(("MISRA-R1", 0.9), ("MISRA-R2", 0.7)))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[
        _mongo_doc("MISRA-R1", title="Rule 1"),
        _mongo_doc("MISRA-R2", title="Rule 2"),
    ])

    result = await rag_node(_make_state("pointer arithmetic"))

    assert len(result["retrieved_rules"]) == 2
    assert result["retrieved_rules"][0]["rule_id"] == "MISRA-R1"
    assert result["retrieved_rules"][0]["relevance_score"] == 0.9
    assert result["retrieved_rules"][1]["rule_id"] == "MISRA-R2"
    assert result["retrieved_rules"][1]["relevance_score"] == 0.7


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_returns_correct_state_keys(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(("MISRA-R1", 0.8)))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[_mongo_doc("MISRA-R1")])

    result = await rag_node(_make_state())

    assert set(result.keys()) == {"retrieved_rules", "rag_query_used", "metadata_filters_applied"}


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_metadata_filter_is_always_misra_c(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    result = await rag_node(_make_state())

    assert result["metadata_filters_applied"] == {"scope": "MISRA C:2023"}


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_rag_query_used_equals_state_query(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    result = await rag_node(_make_state("my specific query"))

    assert result["rag_query_used"] == "my specific query"


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_empty_pinecone_matches_skips_mongo(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    result = await rag_node(_make_state())

    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids.assert_not_called()
    assert result["retrieved_rules"] == []


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_missing_matches_key_in_pinecone_response(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {})  # no "matches" key

    result = await rag_node(_make_state())

    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids.assert_not_called()
    assert result["retrieved_rules"] == []


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_retrieved_rules_sorted_by_score_descending(
    mock_embed, mock_pinecone_svc, mock_mongo_svc
):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(
        ("MISRA-R3", 0.5),
        ("MISRA-R1", 0.95),
        ("MISRA-R2", 0.75),
    ))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[
        _mongo_doc("MISRA-R3"),
        _mongo_doc("MISRA-R1"),
        _mongo_doc("MISRA-R2"),
    ])

    result = await rag_node(_make_state())

    scores = [r["relevance_score"] for r in result["retrieved_rules"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_rule_standard_is_hardcoded_to_misra_c(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(("MISRA-R1", 0.8)))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[_mongo_doc("MISRA-R1")])

    result = await rag_node(_make_state())

    assert result["retrieved_rules"][0]["standard"] == "MISRA C:2023"


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_full_text_fallback_to_text_field(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    """If 'full_text' is absent, 'text' key should be used as fallback."""
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(("MISRA-R1", 0.8)))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[
        {"rule_id": "MISRA-R1", "text": "fallback text content"}
    ])

    result = await rag_node(_make_state())

    assert result["retrieved_rules"][0]["full_text"] == "fallback text content"


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_missing_optional_mongo_fields_use_defaults(
    mock_embed, mock_pinecone_svc, mock_mongo_svc
):
    """Mongo doc with only rule_id → all optional fields fall back to defaults."""
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(("MISRA-R1", 0.6)))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[{"rule_id": "MISRA-R1"}])

    result = await rag_node(_make_state())

    rule = result["retrieved_rules"][0]
    assert rule["section"] == ""
    assert rule["category"] == "N/A"
    assert rule["title"] == "Rule MISRA-R1"
    assert rule["full_text"] == ""


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_partial_mongo_results_are_accepted(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    """Pinecone returns 3 IDs but MongoDB only resolves 2 — no crash."""
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, _pinecone_result(
        ("MISRA-R1", 0.9), ("MISRA-R2", 0.8), ("MISRA-R3", 0.7)
    ))
    mock_mongo_svc.return_value.get_misra_rules_by_pinecone_ids = AsyncMock(return_value=[
        _mongo_doc("MISRA-R1"),
        _mongo_doc("MISRA-R2"),
    ])

    result = await rag_node(_make_state())

    assert len(result["retrieved_rules"]) == 2


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_pinecone_called_with_misra_c_filter_and_top_k_5(
    mock_embed, mock_pinecone_svc, mock_mongo_svc
):
    mock_embed.return_value = FAKE_VECTOR
    query_mock = _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    await rag_node(_make_state("null pointer"))

    query_mock.assert_called_once_with(
        vector=FAKE_VECTOR,
        top_k=5,
        filter={"scope": "MISRA C:2023"},
    )


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_embedding_called_with_query_text(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    await rag_node(_make_state("array bounds check"))

    mock_embed.assert_called_once_with("array bounds check")


@pytest.mark.asyncio
@patch(MONGO_MOCK)
@patch(PINECONE_MOCK)
@patch("app.graph.nodes.rag.get_embedding", new_callable=AsyncMock)
async def test_empty_query_string_is_handled(mock_embed, mock_pinecone_svc, mock_mongo_svc):
    """State with no 'query' key should default to empty string without crash."""
    mock_embed.return_value = FAKE_VECTOR
    _setup_pinecone_mock(mock_pinecone_svc, {"matches": []})

    result = await rag_node({})  # empty state

    assert result["rag_query_used"] == ""
    assert result["retrieved_rules"] == []
