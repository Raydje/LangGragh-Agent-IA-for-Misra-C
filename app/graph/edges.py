from app.models.state import ComplianceState


def route_after_rag(state: ComplianceState) -> str:
    if state["intent"] == "validate":
        return "validation_node"
    return "assemble_node"


def should_loop_or_finish(state: ComplianceState) -> str:
    if state["critique_approved"]:
        return "assemble_node"
    if state["iteration_count"] < state["max_iterations"]:
        return "validation_node"
    return "assemble_node"
