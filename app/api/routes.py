from fastapi import APIRouter, HTTPException
from app.models.requests import ComplianceQueryRequest
from app.models.responses import (
    ComplianceQueryResponse,
    HealthResponse,
    IngestResponse,
)
from app.api.dependencies import get_compiled_graph
from app.services.mongodb_service import get_rules_by_metadata

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from app.services.mongodb_service import _get_db
    from app.services.pinecone_service import _get_index

    mongo_ok = False
    pinecone_ok = False

    try:
        db = _get_db()
        await db.command("ping")
        mongo_ok = True
    except Exception:
        pass

    try:
        index = _get_index()
        index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass

    status = "healthy" if (mongo_ok and pinecone_ok) else "degraded"
    return HealthResponse(
        status=status,
        mongodb_connected=mongo_ok,
        pinecone_connected=pinecone_ok,
    )


@router.post("/query", response_model=ComplianceQueryResponse)
async def query_compliance(request: ComplianceQueryRequest):
    graph = get_compiled_graph()

    initial_state = {
        "query": request.query,
        "code_snippet": request.code_snippet or "",
        "standard": request.standard,
        "iteration_count": 0,
        "max_iterations": 3,
        "critique_history": [],
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ComplianceQueryResponse(
        intent=result.get("intent", "unknown"),
        final_response=result.get("final_response", ""),
        is_compliant=result.get("is_compliant"),
        confidence_score=result.get("confidence_score"),
        cited_rules=result.get("cited_rules", []),
        critique_iterations=result.get("iteration_count", 0),
        critique_passed=result.get("critique_approved", True),
        critique_history=result.get("critique_history", []),
        retrieved_rule_ids=[r["rule_id"] for r in result.get("retrieved_rules", [])],
        error=result.get("error"),
    )


@router.get("/rules")
async def list_rules(standard: str = "DO-178B", dal_level: str | None = None):
    filters = {"standard": standard}
    if dal_level:
        filters["dal_level"] = dal_level
    rules = await get_rules_by_metadata(filters)
    return {"rules": rules, "count": len(rules)}


@router.post("/seed", response_model=IngestResponse)
async def seed_database():
    from app.data.ingest import ingest
    await ingest()
    return IngestResponse(
        message="Seed data ingested successfully",
        rules_ingested=10,
        vectors_upserted=10,
    )
