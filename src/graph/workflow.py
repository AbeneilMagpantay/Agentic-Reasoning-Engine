from langgraph.graph import END, StateGraph
from src.graph.state import AgentState
from src.graph.nodes.retriever import retrieve
from src.graph.nodes.grader import grade_documents
from src.graph.nodes.generator import generate
from src.graph.nodes.query_refiner import refine_query
from src.graph.nodes.hallucination_monitor import check_hallucination
from src.graph.nodes.web_search import web_search

def decide_to_generate_or_fallback(state):
    """
    Determines whether to generate, refine, or fallback to web search.
    """
    print("---DECIDE TO GENERATE OR FALLBACK---")
    documents = state.get("documents", [])
    route = state.get("route", "vectorstore") # Default to vectorstore
    retry_count = state.get("retry_count", 0)
    
    if documents:
        # We have relevant documents!
        print("---DECISION: GENERATE (Data Found)---")
        return "generate"
    
    # No relevant documents found.
    # If we were searching the Vector Store, fallback to Web Search.
    if route == "vectorstore":
        print("---DECISION: VECTORSTORE EMPTY/IRRELEVANT -> FALLBACK TO WEB SEARCH---")
        return "web_search"
    
    # If we were already searching the web and found nothing...
    if retry_count > 1:
        print(f"---DECISION: MAX RETRIES ({retry_count}). GENERATING WITH WHAT WE HAVE.---")
        return "generate"
    else:
        print("---DECISION: WEB SEARCH FAILED -> REFINE QUERY---")
        return "refine_query"

def grade_generation_v_documents_and_question(state):
    print("---GRADE GENERATION---")
    hallucination_grade = state.get("hallucination_grade")
    retry_count = state.get("retry_count", 0)
    
    if hallucination_grade == "useful":
        print("---DECISION: USEFUL---")
        return "useful"
    elif retry_count > 3:
        print(f"---DECISION: MAX RETRIES ({retry_count}). DONE.---")
        return "useful"
    else:
        print(f"---DECISION: NOT USEFUL (Attempt {retry_count}). RETRYING...---")
        return "not useful"

def check_hallucination_skipped(state):
    print("---CHECK POST-GENERATION---")
    # Always check hallucination for robustness in this enterprise version
    print("---DECISION: CHECK HALLUCINATION---")
    return "hallucination_monitor"

def route_refinement(state):
    print("---ROUTE REFINEMENT---")
    route = state.get("route")
    if route == "web_search":
        print("---DECISION: RETRY WEB SEARCH---")
        return "web_search"
    else:
        # If we failed vectorstore, we likely already fell back to web search in the previous step.
        # But if we refine from a vectorstore failure (before fallback triggered?), this logic handles it.
        # Ideally, we refine for the CURRENT route.
        print("---DECISION: RETRY VECTORSTORE---")
        return "retrieve"

def compile_graph():
    """
    Compiles the state graph with Dynamic Fallback logic.
    Entry -> Retrieve -> Grade -> (No Docs?) -> Web Search -> Grade -> Generate
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("hallucination_monitor", check_hallucination)

    # Entry Point: Always try Vector Store First (Lookup-First Strategy)
    workflow.set_entry_point("retrieve")
    
    # Retrieve -> Grade
    workflow.add_edge("retrieve", "grade_documents")
    
    # Web Search -> Grade (Re-use the same grader)
    workflow.add_edge("web_search", "grade_documents")
    
    # Grade Level Conditional Logic (The Brain)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_or_fallback,
        {
            "generate": "generate",
            "web_search": "web_search",         # Fallback path
            "refine_query": "refine_query",     # Give up / Retry path
        },
    )
    
    # Refine -> Loop back (Logic inside route_refinement determines where)
    workflow.add_conditional_edges(
        "refine_query",
        route_refinement,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        }
    )
    
    # Generate -> Hallucination Monitor
    workflow.add_edge("generate", "hallucination_monitor")
    
    # Hallucination Monitor -> End or Retry
    workflow.add_conditional_edges(
        "hallucination_monitor",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "refine_query",
        },
    )

    app = workflow.compile()
    return app

app = compile_graph()
