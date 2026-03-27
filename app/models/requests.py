from pydantic import BaseModel, Field
from typing import Optional


class ComplianceQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language compliance question")
    code_snippet: Optional[str] = Field(
        None, description="Code or requirement text to validate"
    )
    standard: str = Field("DO-178B", description="Technical standard to query against")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Does this function meet MC/DC coverage requirements for Level A?",
                "code_snippet": "def calculate_altitude(pressure: float) -> float:\n    return 44330 * (1 - (pressure / 1013.25) ** 0.1903)",
                "standard": "DO-178B",
            }
        }
    }


class IngestRuleRequest(BaseModel):
    rule_id: str
    standard: str
    section: str
    dal_level: str = Field(..., pattern="^[A-E]$")
    title: str
    full_text: str
    keywords: list[str] = []
