from app.models.state import ComplianceState, CritiqueEntry
from app.services.llm_service import get_llm
from app.utils import parse_json_response


CRITIQUE_PROMPT = """You are a rigorous quality assurance reviewer for a compliance
analysis system. Your sole job is to find errors, hallucinations, and unsupported
claims in the analysis below.

## Retrieved Compliance Rules (ground truth):
{rules_text}

## Analysis Under Review:
{validation_result}

## Cited Rule IDs: {cited_rules}
## Claimed Compliance: {is_compliant}
## Claimed Confidence: {confidence_score}

## Your Review Criteria:
1. **Citation Accuracy**: Does every claim reference a real rule ID from the retrieved
   rules? Flag any rule IDs mentioned in the analysis that do NOT appear in the
   retrieved rules list.
2. **Factual Grounding**: Does the analysis accurately reflect what the rules actually
   say? Flag any paraphrasing that distorts the rule's meaning.
3. **Logical Consistency**: Is the compliance verdict consistent with the detailed
   analysis? (e.g., if the analysis describes gaps, is_compliant should be false)
4. **Confidence Calibration**: Is the confidence score appropriate given the analysis?
   A high-confidence "compliant" verdict with noted gaps is suspicious.
5. **Completeness**: Did the analysis address all relevant retrieved rules, or did it
   ignore some?

Respond with a JSON object:
{{
    "approved": <true|false>,
    "issues": ["<issue 1>", "<issue 2>"],
    "suggestion": "<specific instruction for the validator to fix the issues>",
    "severity": "<none|low|medium|high>"
}}

If you find NO issues, respond with: {{"approved": true, "issues": [], "suggestion": "", "severity": "none"}}
Be thorough but fair. Only flag genuine errors, not stylistic preferences.
"""


async def critique_node(state: ComplianceState) -> dict:
    llm = get_llm(temperature=0.0)

    rules_text = _format_rules(state.get("retrieved_rules", []))
    cited_rules = ", ".join(state.get("cited_rules", []))

    prompt = CRITIQUE_PROMPT.format(
        rules_text=rules_text,
        validation_result=state.get("validation_result", ""),
        cited_rules=cited_rules,
        is_compliant=state.get("is_compliant", False),
        confidence_score=state.get("confidence_score", 0.0),
    )

    response = await llm.ainvoke(prompt)
    parsed = parse_json_response(response.content)

    current_iteration = state.get("iteration_count", 0) + 1

    critique_entry: CritiqueEntry = {
        "iteration": current_iteration,
        "issues_found": parsed["issues"],
        "suggestion": parsed["suggestion"],
        "approved": parsed["approved"],
    }

    return {
        "critique_feedback": parsed["suggestion"],
        "critique_approved": parsed["approved"],
        "iteration_count": current_iteration,
        "critique_history": [critique_entry],
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
