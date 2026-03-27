"""
Seed data ingestion script.
Run with: python -m app.data.ingest
"""
import asyncio
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.data.seed_rules import SEED_RULES
from app.services.embedding_service import get_embeddings_batch
from app.services.pinecone_service import upsert_vectors
from app.services.mongodb_service import insert_rules, create_indexes, get_rules_collection


async def ingest():
    print(f"[Ingest] Starting ingestion of {len(SEED_RULES)} rules...")

    # Step 1: Create MongoDB indexes
    print("[Ingest] Creating MongoDB indexes...")
    await create_indexes()

    # Step 2: Clear existing DO-178B data for idempotent re-runs
    coll = await get_rules_collection()
    result = await coll.delete_many({"standard": "DO-178B"})
    print(f"[Ingest] Cleared {result.deleted_count} existing DO-178B rules from MongoDB.")

    # Step 3: Insert rules into MongoDB
    await insert_rules(SEED_RULES)
    print(f"[Ingest] Inserted {len(SEED_RULES)} rules into MongoDB.")

    # Step 4: Generate embeddings
    texts = [f"{rule['title']}. {rule['full_text']}" for rule in SEED_RULES]
    print("[Ingest] Generating embeddings via Google embedding-001...")
    embeddings = await get_embeddings_batch(texts)
    print(f"[Ingest] Generated {len(embeddings)} embeddings ({len(embeddings[0])} dims each).")

    # Step 5: Upsert to Pinecone
    vectors = [
        {
            "id": rule["rule_id"],
            "values": embedding,
            "metadata": {
                "standard": rule["standard"],
                "section": rule["section"],
                "dal_level": rule["dal_level"],
                "title": rule["title"],
            },
        }
        for rule, embedding in zip(SEED_RULES, embeddings)
    ]
    print("[Ingest] Upserting vectors to Pinecone...")
    await upsert_vectors(vectors)
    print(f"[Ingest] Upserted {len(vectors)} vectors to Pinecone.")

    print("[Ingest] Done.")


if __name__ == "__main__":
    asyncio.run(ingest())
