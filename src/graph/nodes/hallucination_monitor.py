from typing import Literal
from pydantic import BaseModel, Field
from src.graph.state import AgentState
from src.llm import llm

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check in generation text."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to check check if the answer resolves the question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer resolves the question, 'yes' or 'no'"
    )

def check_hallucination(state: AgentState) -> dict:
    """
    Checks if the generation is a hallucination or not supported by documents.
    Returns a status key in the state.
    """
    print("---CHECK HALLUCINATION---")
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]
    
    route = state.get("route", "vectorstore")
    
    # 1. Check Groundedness
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    if route == "web_search":
        print("---HALLUCINATION CHECK: WEB SEARCH MODE (LENIENT)---")
        system = """You are a lenient grader assessing whether an LLM generation is grounded in a set of web search snippets.
        The snippets may be partial or incomplete. 
        If the answer is reasonable given the context, grade it as 'yes'.
        Only grade 'no' if the answer directly contradicts the snippets or is completely unrelated."""
    else:
        print("---HALLUCINATION CHECK: DOCUMENT MODE (STRICT)---")
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in and supported by the set of facts."""
    
    hallucination_prompt = f"System: {system}\nSet of Facts: {documents}\nLLM Generation: {generation}"
    
    try:
        grade = structured_llm_grader.invoke(hallucination_prompt)
        score = grade.binary_score
    except Exception as e:
        print(f"Hallucination grading error: {e}")
        score = "no"

    retry_count = state.get("retry_count", 0)
    
    if score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # 2. Check Answer Quality
        structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)
        
        if route == "web_search":
             system_answer = """You are a lenient grader. The snippets might not contain the full answer.
             If the generation addresses the question partially or provides a summary based on available info, grade as 'yes'."""
        else:
            system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question."""
            
        answer_prompt = f"System: {system_answer}\nUser Question: {question}\nLLM Generation: {generation}"
        
        grade_answer = structured_llm_grader_answer.invoke(answer_prompt)
        if grade_answer.binary_score == "yes":
             print("---DECISION: GENERATION ADDRESSES QUESTION---")
             return {"hallucination_grade": "useful", "retry_count": 0} # Reset on success? Or keep?
        else:
             print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
             return {"hallucination_grade": "not useful", "retry_count": retry_count + 1}
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS (HALLUCINATION)---")
        return {"hallucination_grade": "not useful", "retry_count": retry_count + 1} 
