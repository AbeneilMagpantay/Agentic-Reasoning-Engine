from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.state import AgentState
from src.llm import llm

def generate(state: AgentState) -> AgentState:
    """
    Generates an answer using the retrieved documents.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("documents", [])
    route = state.get("route", "vectorstore")
    
    if route == "general":
        # General chat prompt
        prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Answer the user's question or greeting naturally.
            
            Question: {question}
            
            Answer:""",
            input_variables=["question"],
        )
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"question": question})
    else:
        # RAG prompt
        prompt = PromptTemplate(
            template="""You are an expert research assistant. Use the provided context to answer the question.
            
            **Style Instructions:**
            1. **Structure**: Use Markdown headers (###) to organize the answer into logical sections.
            2. **Lists**: Use bullet points for lists of items (e.g. cities, features).
            3. **Conciseness**: Be direct and professional. Avoid fluff.
            4. **Citations**: 
               - Use numbered citations inline like [1], [2].
               - DO NOT put long URLs in the text.
            5. **References Section**:
               - At the bottom, list sources as clickable Markdown links.
               - Format: 1. [Title](URL)
            
            **Context:**
            {context}
            
            **Question:** {question} 
            
            **Answer:**
            (Provide the structured answer here, followed by a 'References' section)
            
            ### References
            1. [Title](URL)
            2. ...
            """,
            input_variables=["question", "context"],
        )
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        
    return {"documents": documents, "question": question, "generation": generation}
