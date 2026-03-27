import re
from app.models.state import ComplianceState, RetrievedRule
from app.services.embedding_service import get_embedding
from app.services.pinecone_service import query_pinecone
from app.services.mongodb_service import get_rules_by_ids, get_rules_by_metadata


async def rag_node(state: ComplianceState) -> dict:
    query = state["query"]
    standard = state.get("standard", "DO-178B")

    # Step 1: Generate embedding for the query
    query_embedding = await get_embedding(query)

    # Step 2: Semantic search in Pinecone
    pinecone_results = await query_pinecone(
        vector=query_embedding,
        top_k=5,
        filter={"standard": standard},
    )

    # Step 3: Extract rule_ids and scores
    semantic_hits = {
        match["id"]: match["score"]
        for match in pinecone_results.get("matches", [])
    }
    rule_ids = list(semantic_hits.keys())

    # Step 4: Enrich from MongoDB (full text + metadata)
    mongo_rules = await get_rules_by_ids(rule_ids) if rule_ids else []

    # Step 5: Extract metadata filters from query
    dal_keywords = _extract_dal_level(query)
    section_keywords = _extract_section_ref(query)

    metadata_filters = {}
    if dal_keywords:
        metadata_filters["dal_level"] = {"$in": dal_keywords}
    if section_keywords:
        metadata_filters["section"] = {"$regex": section_keywords, "$options": "i"}

    # Step 6: Supplemental MongoDB metadata search
    supplemental_rules = []
    if metadata_filters:
        metadata_filters["standard"] = standard
        supplemental_rules = await get_rules_by_metadata(metadata_filters)

    # Step 7: Merge and deduplicate
    merged = _merge_results(mongo_rules, supplemental_rules, semantic_hits)

    return {
        "retrieved_rules": merged,
        "rag_query_used": query,
        "metadata_filters_applied": metadata_filters,
    }


def _extract_dal_level(query: str) -> list[str]:
    matches = re.findall(
        r"(?:level|dal|assurance level)\s*([A-E])", query, re.IGNORECASE
    )
    return [m.upper() for m in matches]


def _extract_section_ref(query: str) -> str | None:
    match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", query)
    return match.group(1) if match else None


def _merge_results(
    mongo_rules: list[dict],
    supplemental: list[dict],
    semantic_scores: dict[str, float],
) -> list[RetrievedRule]:
    seen = set()
    results: list[RetrievedRule] = []

    for rule in mongo_rules:
        rid = rule["rule_id"]
        if rid not in seen:
            seen.add(rid)
            results.append(RetrievedRule(
                rule_id=rid,
                standard=rule["standard"],
                section=rule["section"],
                dal_level=rule["dal_level"],
                title=rule["title"],
                full_text=rule["full_text"],
                relevance_score=semantic_scores.get(rid, 0.0),
            ))

    for rule in supplemental:
        rid = rule["rule_id"]
        if rid not in seen:
            seen.add(rid)
            results.append(RetrievedRule(
                rule_id=rid,
                standard=rule["standard"],
                section=rule["section"],
                dal_level=rule["dal_level"],
                title=rule["title"],
                full_text=rule["full_text"],
                relevance_score=0.0,
            ))

    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    return results[:10]
