from langgraph.graph import StateGraph, START, END
from app.models.state import ComplianceState
from app.graph.nodes.orchestrator import orchestrator_node
from app.graph.nodes.rag import rag_node
from app.graph.nodes.validation import validation_node
from app.graph.nodes.critique import critique_node
from app.graph.edges import route_after_rag, should_loop_or_finish


async def assemble_node(state: ComplianceState) -> dict:
    intent = state.get("intent", "search")
    rules = state.get("retrieved_rules", [])
    validation = state.get("validation_result", "")
    approved = state.get("critique_approved", True)
    iterations = state.get("iteration_count", 0)

    if intent == "search":
        parts = [f"Found {len(rules)} relevant compliance rule(s):\n"]
        for r in rules:
            parts.append(
                f"- [{r['rule_id']}] {r['title']} (Section {r['section']}, "
                f"Level {r['dal_level']})\n  {r['full_text'][:300]}"
            )
        final = "\n".join(parts)
    elif intent == "validate":
        final = validation
        if not approved:
            final += (
                f"\n\n[Note: This analysis underwent {iterations} review "
                f"iteration(s) but did not fully pass quality review. "
                f"Please verify critical findings independently.]"
            )
    else:  # explain
        if validation:
            final = validation
        else:
            parts = []
            for r in rules:
                parts.append(f"**[{r['rule_id']}] {r['title']}**\n{r['full_text']}")
            final = "\n\n".join(parts) if parts else "No relevant rules found."

    return {"final_response": final}


def build_graph() -> StateGraph:
    builder = StateGraph(ComplianceState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("critique_node", critique_node)
    builder.add_node("assemble_node", assemble_node)

    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "rag_node")

    builder.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {"validation_node": "validation_node", "assemble_node": "assemble_node"},
    )

    builder.add_edge("validation_node", "critique_node")

    builder.add_conditional_edges(
        "critique_node",
        should_loop_or_finish,
        {"validation_node": "validation_node", "assemble_node": "assemble_node"},
    )

    builder.add_edge("assemble_node", END)

    return builder.compile()
