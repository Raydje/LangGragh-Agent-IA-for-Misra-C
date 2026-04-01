import re
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from app.config import get_settings

_INDEX_FIELDS = [("rule_type", 1), ("section", 1), ("rule_number", 1)]
_ID_RE = re.compile(r'^MISRA_(RULE|DIR)_(\d+)\.(\d+)$')

_service: "MongoDBService | None" = None


def get_mongodb_service() -> "MongoDBService":
    global _service
    if _service is None:
        _service = MongoDBService()
    return _service


class MongoDBService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        self.collection: AsyncIOMotorCollection = self.db[settings.mongodb_collection]

    def close(self) -> None:
        self.client.close()

    async def get_rules_by_ids(self, rule_ids: list[str]) -> list[dict]:
        cursor = self.collection.find({"rule_id": {"$in": rule_ids}}, {"_id": 0})
        return await cursor.to_list(length=100)

    async def get_misra_rules_by_pinecone_ids(self, rule_ids: list[str]) -> list[dict]:
        """
        Resolves Pinecone IDs (e.g. 'MISRA_RULE_15.1', 'MISRA_DIR_4.1') to MongoDB documents.
        MongoDB stores rules with separate 'rule_type', 'section' and 'rule_number' int fields.
        Returns each doc annotated with a 'rule_id' key matching the original Pinecone ID.
        """
        or_conditions = []
        id_map: dict[tuple, str] = {}  # (rule_type, section, rule_number) -> original Pinecone ID

        for rid in rule_ids:
            m = _ID_RE.match(rid)
            if m:
                rule_type, section, rule_number = m.group(1), int(m.group(2)), int(m.group(3))
                or_conditions.append({"rule_type": rule_type, "section": section, "rule_number": rule_number})
                id_map[(rule_type, section, rule_number)] = rid

        if not or_conditions:
            return []

        cursor = self.collection.find({"$or": or_conditions}, {"_id": 0})
        docs = await cursor.to_list(length=100)

        for doc in docs:
            key = (doc.get("rule_type"), doc.get("section"), doc.get("rule_number"))
            doc["rule_id"] = id_map.get(key, "")

        return docs

    async def get_rules_by_metadata(self, filters: dict) -> list[dict]:
        """Find MISRA rules by metadata fields.
        Example: {"section": 3, "rule_number": 4} for MISRA 3.4 (ints, not strings).
        """
        cursor = self.collection.find(filters, {"_id": 0})
        return await cursor.to_list(length=100)

    async def insert_rules(self, rules: list[dict]) -> None:
        if rules:
            await self.collection.insert_many(rules)

    async def create_indexes(self) -> None:
        try:
            await self.collection.drop_index("section_1_rule_number_1")
        except Exception:
            pass  # Index doesn't exist — that's fine
        await self.collection.create_index(_INDEX_FIELDS, unique=True)
