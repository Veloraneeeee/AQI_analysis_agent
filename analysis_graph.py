from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def build_analysis_graph(
    state_schema,
    nodes,
    confirmation_router,
    validation_router,
):
    builder = StateGraph(state_schema)

    for node_name, node_function in nodes.items():
        builder.add_node(node_name, node_function)

    builder.add_edge(START, "extract_event_types")

    builder.add_edge("inspect_files", "identify_columns")
    builder.add_edge("identify_columns", "ask_for_confirmation")

    builder.add_conditional_edges(
        "ask_for_confirmation",
        confirmation_router,
        {
            "extract_event_types": "extract_event_types",
            "revise_columns": "revise_columns",
        },
    )

    builder.add_edge("revise_columns", "ask_for_confirmation")
    builder.add_edge("extract_event_types", "select_event_type")
    builder.add_edge("select_event_type", "validate_data")

    builder.add_conditional_edges(
        "validate_data",
        validation_router,
        {
            "select_event_type": "select_event_type",
            "explain_missing_data": "explain_missing_data",
            "match_and_calculate": "match_and_calculate",
        },
    )

    builder.add_edge("explain_missing_data", END)
    builder.add_edge("match_and_calculate", "summarize_results")
    builder.add_edge("summarize_results", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(
        checkpointer=InMemorySaver(),
    )



