import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from typing import List

from ..state import LawyerAgentState


# Maximum L2 distance for a FAISS match to count as relevant (measured on this index).
RELEVANCE_MAX_DISTANCE = 1.25

# Low-quality / user-generated web domains that should never be cited as legal sources.
BLOCKED_WEB_DOMAINS = (
    "instagram.com", "facebook.com", "twitter.com", "x.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "pinterest.com", "quora.com",
    "scribd.com", "linkedin.com", "threads.net", "whatsapp.com",
)


def legal_database_search(query: str):
    """Search FAISS. Returns (content_string, list_of_source_dicts)."""
    try:
        FAISS_INDEX_PATH = "data/faiss_index"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        db = FAISS.load_local(
            FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )

        retrieved_docs = db.similarity_search_with_score(query, k=8)
        # Absolute relevance cutoff: drop off-topic matches, keep none if nothing fits.
        content = ""
        sources = []
        for doc, dist in retrieved_docs:
            if dist <= RELEVANCE_MAX_DISTANCE:
                content += doc.page_content + "\n\n"
                if doc.metadata:
                    sources.append({"type": "document", "metadata": doc.metadata})
        return content, sources
    except Exception as e:
        return f"Error during legal database search: {e}", []


def web_search(query: str):
    """Perform web search using Tavily. Returns (content_string, list_of_source_dicts)."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    web_search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)
    results = web_search_tool.invoke(query)

    # TavilySearch returns a dict with a "results" list; normalize both shapes.
    if isinstance(results, dict):
        items = results.get("results", [])
    elif isinstance(results, list):
        items = results
    else:
        items = []

    content = ""
    sources = []
    for item in items:
        if isinstance(item, dict):
            url = item.get("url", "")
            # Skip social media / user-generated content - not citable legal sources.
            if url and any(bad in url.lower() for bad in BLOCKED_WEB_DOMAINS):
                continue
            content += f"{item.get('title', '')}\n{item.get('content', '')}\n\n"
            sources.append({"type": "web", "title": item.get("title", ""), "url": url})
        else:
            content += str(item) + "\n\n"
            sources.append({"type": "web", "content": str(item)})
    return content, sources


def retriever(state: LawyerAgentState) -> dict:
    """Retrieve context for each sub-query and merge into retrieved_context + sources."""
    print("---RETRIEVING CONTEXT---")
    sub_queries = state["sub_queries"]

    retrieved_context = []
    all_sources = []
    for sub_query in sub_queries:
        print(f"Retrieving for: {sub_query}")
        faiss_result, faiss_sources = legal_database_search(sub_query)
        web_result, web_sources = web_search(sub_query)
        combined = f"Sub-query: {sub_query}\nFAISS: {faiss_result}\nWeb: {web_result}\n"
        retrieved_context.append(combined)
        all_sources.extend(faiss_sources)
        all_sources.extend(web_sources)

    return {"retrieved_context": retrieved_context, "sources": all_sources}
