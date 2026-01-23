from typing import Literal
from pydantic import BaseModel, Field
from src.graph.state import AgentState
from src.llm import llm

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search", "general"] = Field(
        ...,
        description="Given a user question choose to route it to web search, vectorstore, or general/chitchat.",
    )

def route_question(state: AgentState):
    """
    Routes the question to the appropriate data source.
    """
    print("---ROUTE QUESTION---")
    
    # Manual Override Check (Enterprise Feature)
    if state.get("route") in ["vectorstore", "web_search", "general"]:
         print(f"---ROUTE PRE-SELECTED: {state['route']}---")
         return {"route": state["route"]}

    question = state["question"]
    
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    # Prompt the router
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents about:
    1. 'Agentic Reasoning', 'LangGraph', 'Adversarial Attacks'
    2. 'Antigravity Corp', 'Financial Reports', 'Revenue', 'Q3 2025' (Internal Data)
    
    Use the vectorstore for questions on these topics.
    Use 'general' for greetings, chitchat, or simple questions that don't need retrieval (e.g. "Hello", "How are you").
    Otherwise, use web-search (e.g. for current world events, weather, or generic companies like Microsoft/Apple that aren't us)."""
    
    route_prompt = f"System: {system}\nQuestion: {question}"
    
    try:
        source = structured_llm_router.invoke(route_prompt)
        print(f"---ROUTED TO: {source.datasource}---")
        # Update state with decision
        return {"route": source.datasource}
    except Exception as e:
        print(f"Routing Error: {e}, defaulting to/vectorstore")
        return {"route": "vectorstore"}
