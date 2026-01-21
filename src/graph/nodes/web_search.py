from ddgs import DDGS
from src.graph.state import AgentState

def web_search(state: AgentState) -> AgentState:
    """
    Web search based on the re-phrased question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    
    # 1. Generate optimized search query
    from src.llm import llm
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = PromptTemplate(
        template="""You are an expert at converting conversational user questions into effective web search queries.
        Rules:
        1. If the user asks for a specific fact (e.g. "CEO of...", "Population of..."), search for that exact term.
        2. If the user asks for a **LIST** or "every" item (e.g. "List cities", "All pokemon"), you MUST include words like "list of", "table", or "wikipedia" to find structured data instead of blogs.
        
        Examples:
        "u dont know about the bicol region? can u search for it" -> Bicol region overview
        "list every place in [X]" -> list of cities and municipalities in [X] wikipedia
        "what are the [items] in [category]" -> list of [items] in [category]
        "tell me about python 3.12" -> Python 3.12 features
        
        Question: {question}
        Search Query:""",
        input_variables=["question"],
    )
    chain = prompt | llm | StrOutputParser()
    try:
        search_query = chain.invoke({"question": question}).strip()
        print(f"---OPTIMIZED SEARCH QUERY: {search_query}---")
    except Exception as e:
        print(f"Query Gen Error: {e}")
        search_query = question

    # 2. Execute Search
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def run_search(query):
        try:
            results = DDGS().text(query, max_results=3)
            return results
        except Exception as e:
            print(f"  [Search Failed]: {e}")
            return None

    print(f"---ATTEMPT 1: '{search_query}'---")
    results = run_search(search_query)
    
    # Fallback: If optimized query fails, try a simpler version
    if not results:
        print("---NO RESULTS. TRYING FALLBACK...---")
        # Heuristic: just the last few words or the raw question? 
        # Let's try to ask Gemini for a BROADER query or just use the question.
        fallback_query = question 
        print(f"---ATTEMPT 2: '{fallback_query}'---")
        results = run_search(fallback_query)

    # Format results
    if results:
        content = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}" for r in results])
        documents = [content]
        print(f"---WEB SEARCH RESULTS: Found {len(results)} docs---")
        for i, r in enumerate(results):
            print(f"  [{i}] {r['title']}: {r['body'][:100]}...")
    else:
        documents = ["System: The web search returned no results. The agent tried searching but found nothing."]
        print("---WEB SEARCH RESULTS: No results after retries---")

    return {"documents": documents, "question": question}
