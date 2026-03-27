from app.models.state import ComplianceState
from app.services.llm_service import get_llm
from app.utils import parse_json_response


CLASSIFICATION_PROMPT = """You are an intent classifier for a regulatory compliance system.
Given a user query about technical standards (e.g., DO-178B, DO-254, ARP4754A),
classify it into exactly ONE of these intents:

- "search": The user wants to FIND or LOOK UP specific compliance rules,
  requirements, or sections. They are asking "what does the standard say about X?"
- "validate": The user wants to CHECK whether a specific code snippet,
  requirement, or design artifact MEETS a compliance rule. They are providing
  something to be evaluated against the standard.
- "explain": The user wants an EXPLANATION or INTERPRETATION of a compliance
  concept. They are asking "what does X mean?" or "why is X required?"

User query: {query}
Code snippet provided: {has_code}

Respond with ONLY a JSON object:
{{"intent": "<search|validate|explain>", "reasoning": "<one sentence explanation>"}}
"""


async def orchestrator_node(state: ComplianceState) -> dict:
    llm = get_llm(temperature=0.0)
    has_code = "Yes" if state.get("code_snippet") else "No"

    prompt = CLASSIFICATION_PROMPT.format(
        query=state["query"],
        has_code=has_code,
    )
    response = await llm.ainvoke(prompt)
    parsed = parse_json_response(response.content)

    intent = parsed["intent"]
    if state.get("code_snippet") and intent != "validate":
        intent = "validate"
        parsed["reasoning"] += " (Overridden to 'validate' because code snippet was provided.)"

    return {
        "intent": intent,
        "orchestrator_reasoning": parsed["reasoning"],
        "standard": state.get("standard", "DO-178B"),
        "max_iterations": state.get("max_iterations", 3),
        "iteration_count": 0,
    }
