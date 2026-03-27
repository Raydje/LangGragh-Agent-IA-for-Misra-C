from app.models.state import ComplianceState
from app.services.llm_service import get_llm
from app.utils import parse_json_response


VALIDATION_PROMPT = """You are a DO-178B compliance validation expert. Your task is to
determine whether the provided artifact (code snippet or requirement) satisfies the
compliance rules retrieved from the standard.

## Compliance Rules (from DO-178B):
{rules_text}

## Artifact Under Review:
```
{code_snippet}
```

## User's Question:
{query}

{critique_context}

## Instructions:
1. For each relevant rule, state whether the artifact SATISFIES, PARTIALLY SATISFIES,
   or DOES NOT SATISFY the rule.
2. Cite specific rule IDs (e.g., [DO178B-REQ-003]) for every claim.
3. If the artifact is incomplete or ambiguous, state what is missing.
4. Provide a confidence score from 0.0 to 1.0 reflecting your certainty.

Respond with a JSON object:
{{
    "analysis": "<detailed multi-paragraph compliance analysis>",
    "is_compliant": <true|false>,
    "confidence_score": <0.0 to 1.0>,
    "cited_rules": ["<rule_id_1>", "<rule_id_2>"],
    "gaps": ["<gap description 1>", "<gap description 2>"]
}}
"""


async def validation_node(state: ComplianceState) -> dict:
    llm = get_llm(temperature=0.1)

    rules_text = _format_rules(state.get("retrieved_rules", []))
    code_snippet = state.get("code_snippet", "(No code snippet provided)")

    critique_context = ""
    if state.get("iteration_count", 0) > 0 and state.get("critique_feedback"):
        critique_context = (
            f"\n## IMPORTANT - Previous Review Feedback:\n"
            f"A quality reviewer found issues with your previous analysis. "
            f"Address these specifically:\n{state['critique_feedback']}\n"
        )

    prompt = VALIDATION_PROMPT.format(
        rules_text=rules_text,
        code_snippet=code_snippet,
        query=state["query"],
        critique_context=critique_context,
    )

    response = await llm.ainvoke(prompt)
    parsed = parse_json_response(response.content)

    return {
        "validation_result": parsed["analysis"],
        "is_compliant": parsed["is_compliant"],
        "confidence_score": parsed["confidence_score"],
        "cited_rules": parsed["cited_rules"],
    }


def _format_rules(rules: list[dict]) -> str:
    parts = []
    for r in rules:
        parts.append(
            f"[{r['rule_id']}] {r['title']}\n"
            f"  Section: {r['section']} | DAL Level: {r['dal_level']}\n"
            f"  {r['full_text']}\n"
        )
    return "\n".join(parts) if parts else "(No rules retrieved)"
