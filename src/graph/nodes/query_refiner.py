from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import AgentState
from src.llm import llm

def refine_query(state: AgentState) -> AgentState:
    """
    Refines the question to improve retrieval.
    """
    print("---REFINE QUERY---")
    question = state["question"]
    
    # Prompt
    system = """You are an expert at optimizing queries for vector search. 
    Look at the initial question and formulate a better question that expresses the same intent but is more likely to retrieve relevant documents.
    Return ONLY the refined question."""
    
    prompt = PromptTemplate(
        template=f"{system}\n\nInitial Question: {{question}}\nRefined Question:",
        input_variables=["question"],
    )
    
    # Chain
    chain = prompt | llm | StrOutputParser()
    
    refined_question = chain.invoke({"question": question})
    print(f"---REFINED QUESTION: {refined_question}---")
    
    retry_count = state.get("retry_count", 0)
    return {"question": refined_question, "retry_count": retry_count + 1}
