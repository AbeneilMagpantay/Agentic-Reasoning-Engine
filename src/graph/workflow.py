from langgraph.graph import END, StateGraph
from src.graph.state import AgentState
from src.graph.nodes.retriever import retrieve
from src.graph.nodes.grader import grade_documents
from src.graph.nodes.generator import generate
from src.graph.nodes.query_refiner import refine_query
from src.graph.nodes.hallucination_monitor import check_hallucination

def decide_to_generate(state):
    """
    Determines whether to generate (relevant docs found) or re-generate query.
    """
    print("---DECIDE TO GENERATE---")
    retry_count = state.get("retry_count", 0)
    
    if state["documents"]:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    elif retry_count > 1: # Strict limit for refinement
        print(f"---DECISION: MAX REFINE RETRIES ({retry_count}) REACHED. GENERATING WITHOUT DOCS.---")
        return "generate"
    else:
        # No relevant documents found, refine query
        print("---DECISION: REFINE QUERY---")
        return "refine_query"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---GRADE GENERATION---")
    hallucination_grade = state.get("hallucination_grade")
    retry_count = state.get("retry_count", 0)
    
    if hallucination_grade == "useful":
        print("---DECISION: USEFUL---")
        return "useful"
    elif retry_count > 3:
        print(f"---DECISION: MAX RETRIES ({retry_count}) REACHED. STOPPING LOOP.---")
        return "useful" # Accept it to prevent crash
    else:
        print(f"---DECISION: NOT USEFUL (Attempt {retry_count}). RETRYING...---")
        return "not useful"

from src.graph.nodes.web_search import web_search
from src.graph.nodes.router import route_question

def route_decision(state):
    print("---ROUTE DECISION---")
    route = state.get("route")
    if route == "general":
        print("---DECISION: GENERAL CHAT (FAST)---")
        return "generate"
    elif route == "web_search":
         print("---DECISION: WEB SEARCH---")
         return "web_search"
    else:
        print("---DECISION: VECTORSTORE---")
        return "retrieve"

def check_hallucination_skipped(state):
    """
    Determines if we should check for hallucinations or skip (fast mode).
    """
    print("---CHECK POST-GENERATION---")
    route = state.get("route")
    if route == "general":
        print("---DECISION: SKIP HALLUCINATION CHECK (FAST MODE)---")
        return "end"
    else:
        print("---DECISION: CHECK HALLUCINATION---")
        return "hallucination_monitor"

def route_refinement(state):
    """
    Routes the refined query back to the original source.
    """
    print("---ROUTE REFINEMENT---")
    route = state.get("route")
    if route == "web_search":
        print("---DECISION: RETRY WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: RETRY VECTORSTORE---")
        return "retrieve"

def compile_graph():
    """
    Compiles the state graph.
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("router", route_question)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("refine_query", refine_query)
    workflow.add_node("hallucination_monitor", check_hallucination)

    # Define edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "web_search": "web_search",
        }
    )
    
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "grade_documents")
    
    # Conditional edge: Grade -> Generate OR Refine
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "refine_query": "refine_query",
            "generate": "generate",
        },
    )
    
    # Conditional edge: Refine -> Retrieve OR Web Search
    workflow.add_conditional_edges(
        "refine_query",
        route_refinement,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        }
    )
    
    # Conditional edge after generate: Monitor OR End
    workflow.add_conditional_edges(
        "generate",
        check_hallucination_skipped,
        {
            "end": END,
            "hallucination_monitor": "hallucination_monitor"
        }
    )
    
    # Conditional edge: Monitor -> END OR Generate (Retry)
    workflow.add_conditional_edges(
        "hallucination_monitor",
        grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not useful": "refine_query", # Loop back to refine query (better than just retrying generation)
        },
    )

    # Compile
    app = workflow.compile()
    return app

app = compile_graph()
