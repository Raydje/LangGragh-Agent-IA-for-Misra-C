import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.mongodb_service import MongoDBService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_collection(docs: list[dict]) -> MagicMock:
    """Return a Motor collection mock whose find().to_list() returns docs."""
    coll = MagicMock()
    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    coll.find.return_value = cursor
    return coll


def _make_service(docs: list[dict]) -> MongoDBService:
    """Create a MongoDBService with a mocked collection (bypasses __init__)."""
    svc = object.__new__(MongoDBService)
    svc.collection = _make_mock_collection(docs)
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_ids_return_annotated_docs():
    docs = [
        {"rule_type": "RULE", "section": 15, "rule_number": 1, "title": "No recursion"},
        {"rule_type": "RULE", "section": 1, "rule_number": 3, "title": "Undefined behaviour"},
    ]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.1", "MISRA_RULE_1.3"])

    assert len(result) == 2
    rule_ids = {doc["rule_id"] for doc in result}
    assert "MISRA_RULE_15.1" in rule_ids
    assert "MISRA_RULE_1.3" in rule_ids


@pytest.mark.asyncio
async def test_dir_ids_return_annotated_docs():
    docs = [
        {"rule_type": "DIR", "section": 4, "rule_number": 1, "title": "Run-time failures shall be minimized"},
    ]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_DIR_4.1"])

    assert len(result) == 1
    assert result[0]["rule_id"] == "MISRA_DIR_4.1"


@pytest.mark.asyncio
async def test_empty_ids_returns_empty_without_db_call():
    svc = _make_service([])
    result = await svc.get_misra_rules_by_pinecone_ids([])

    svc.collection.find.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_all_malformed_ids_skips_db():
    svc = _make_service([])
    result = await svc.get_misra_rules_by_pinecone_ids(["badformat", "MISRA_abc.def", "MISRA_"])

    svc.collection.find.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_mixed_valid_and_malformed_ids_only_queries_valid():
    docs = [{"rule_type": "RULE", "section": 1, "rule_number": 1, "title": "T1"}]
    svc = _make_service(docs)
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1", "bad_id", "MISRA_abc.1"])

    assert len(result) == 1
    assert result[0]["rule_id"] == "MISRA_RULE_1.1"


@pytest.mark.asyncio
async def test_rule_id_annotated_on_returned_doc():
    doc = {"rule_type": "RULE", "section": 15, "rule_number": 5, "title": "No goto"}
    svc = _make_service([doc])
    result = await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.5"])

    assert result[0]["rule_id"] == "MISRA_RULE_15.5"


@pytest.mark.asyncio
async def test_find_called_with_correct_or_conditions():
    doc = {"rule_type": "RULE", "section": 15, "rule_number": 1, "title": "T"}
    svc = _make_service([doc])
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_15.1"])

    call_filter = svc.collection.find.call_args[0][0]
    assert "$or" in call_filter
    assert {"rule_type": "RULE", "section": 15, "rule_number": 1} in call_filter["$or"]


@pytest.mark.asyncio
async def test_multiple_ids_build_multiple_or_conditions():
    docs = [
        {"rule_type": "RULE", "section": 1, "rule_number": 1},
        {"rule_type": "RULE", "section": 2, "rule_number": 3},
    ]
    svc = _make_service(docs)
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1", "MISRA_RULE_2.3"])

    call_filter = svc.collection.find.call_args[0][0]
    assert len(call_filter["$or"]) == 2


@pytest.mark.asyncio
async def test_excludes_id_field_from_projection():
    doc = {"rule_type": "RULE", "section": 1, "rule_number": 1}
    svc = _make_service([doc])
    await svc.get_misra_rules_by_pinecone_ids(["MISRA_RULE_1.1"])

    projection = svc.collection.find.call_args[0][1]
    assert projection == {"_id": 0}
