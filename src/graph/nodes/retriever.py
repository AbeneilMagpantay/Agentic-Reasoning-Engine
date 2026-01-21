from src.graph.state import AgentState
from src.vectorstore import get_retriever

def retrieve(state: AgentState) -> AgentState:
    """
    Retrieves documents from the vector store.
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    retriever = get_retriever()
    try:
        documents = retriever.invoke(question)
        doc_contents = []
        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            content = f"Content: {doc.page_content}\nSource: {source}"
            doc_contents.append(content)
        return {"documents": doc_contents, "question": question}
    except Exception as e:
        print(f"Retrieval Error: {e}")
        return {"documents": [], "question": question}
