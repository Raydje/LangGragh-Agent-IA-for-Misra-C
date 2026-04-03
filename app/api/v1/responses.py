from pydantic import BaseModel
from typing import Any, Optional

class CritiqueDetail(BaseModel):
    iteration: int
    issues_found: list[str]
    approved: bool

class MetadataUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    orchestrator_tokens: Optional[int] = None
    validation_tokens: Optional[int] = None
    critique_tokens: Optional[int] = None
    remediation_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None


class ComplianceQueryResponse(BaseModel):
    thread_id: str
    intent: str
    final_response: str
    is_compliant: Optional[bool] = None
    confidence_score: Optional[float] = None
    cited_rules: list[str] = []
    critique_iterations: int = 0
    critique_passed: bool = True
    critique_history: list[CritiqueDetail] = []
    retrieved_rule_ids: list[str] = []
    error: Optional[str] = None
    fixed_code_snippet: Optional[str] = None
    remediation_explanation: Optional[str] = None
    total_tokens_usage: MetadataUsage


class HealthResponse(BaseModel):
    status: str
    mongodb_connected: bool
    pinecone_connected: bool


class IngestResponse(BaseModel):
    message: str
    rules_ingested: int
    vectors_upserted: int


class ThreadHistoryEntry(BaseModel):
    checkpoint_id: Optional[str]
    next_node: tuple
    values: dict[str, Any]


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    history: list[ThreadHistoryEntry]
