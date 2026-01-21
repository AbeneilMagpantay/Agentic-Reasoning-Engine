from typing import Literal
from pydantic import BaseModel, Field
from src.graph.state import AgentState
from src.llm import llm

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state: AgentState) -> AgentState:
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK RELEVANCE---")
    documents = state["documents"]
    question = state["question"]
    
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    route = state.get("route")
    
    if route == "web_search":
        system = """You are a grader assessing relevance of a web search result snippet to a user question. \n
        The snippet might be short/incomplete. If the snippet mentions keywords related to the question or seems to talk about the right topic, grade it as 'yes'. \n
        Give a binary score 'yes' or 'no'."""
    else:
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    filtered_docs = []
    for doc in documents:
        prompt = f"System: {system}\nQuestion: {question}\nDocument: {doc}"
        grade = structured_llm_grader.invoke(prompt)
        score = grade.binary_score
        if score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    return {"documents": filtered_docs, "question": question}
