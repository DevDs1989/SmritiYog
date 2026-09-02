from langgraph.graph import END, START, StateGraph

from app.agents.content import content_node
from app.agents.monitor import monitor_node
from app.agents.planner import planner_node
from app.agents.state import PatientState


def build_graph():
    builder = StateGraph(PatientState)
    builder.add_node("planner", planner_node)
    builder.add_node("content", content_node)
    builder.add_node("monitor", monitor_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "content")
    builder.add_edge("content", "monitor")
    builder.add_edge("monitor", END)
    return builder.compile()


graph = build_graph()
