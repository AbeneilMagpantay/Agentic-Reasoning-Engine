from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
    Represents the internal state of the agent.
    """
    question: str
    generation: Optional[str]
    documents: List[str] # List of retrieved document contents
    step: str # Current step in the graph
    hallucination_grade: Optional[str] # 'useful' or 'not useful'
    retry_count: int = 0 # Track correction attempts
    route: Optional[str] # 'vectorstore', 'web_search', 'general'
