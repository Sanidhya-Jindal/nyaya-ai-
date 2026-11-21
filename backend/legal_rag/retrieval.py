# legal_rag/retrieval.py

import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Any
import operator

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch

# -------------------------
# Load environment variables
# -------------------------
backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(backend_dir / ".env")

# -------------------------
# GLOBAL LAZY-LOADED OBJECTS
# -------------------------
_embedding_model = None
_faiss_db = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Must match the model the FAISS index was built with
        # (see data/faiss_index/faiss_indexer.py). Using a different model
        # here silently degrades retrieval relevance.
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def get_faiss_db():
    global _faiss_db
    if _faiss_db is None:
        # Resolve from this file so the app works whatever folder it is started from.
        FAISS_INDEX_PATH = str(Path(__file__).resolve().parents[1] / "data" / "faiss_index")
        _faiss_db = FAISS.load_local(
            FAISS_INDEX_PATH,
            get_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    return _faiss_db


# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]
    sources: Annotated[List[dict], operator.add]


# -------------------------
# FAISS Legal DB Tool (LAZY SAFE)
# -------------------------
# Maximum L2 distance for a FAISS match to count as relevant (see legal_database_search).
RELEVANCE_MAX_DISTANCE = 1.25

# Low-quality / user-generated web domains that should never be cited as legal sources.
BLOCKED_WEB_DOMAINS = (
    "instagram.com", "facebook.com", "twitter.com", "x.com", "tiktok.com",
    "youtube.com", "youtu.be", "reddit.com", "pinterest.com", "quora.com",
    "scribd.com", "linkedin.com", "threads.net", "whatsapp.com",
)


@tool
def legal_database_search(query: str) -> List[Document]:
    """
    Search against a pre-indexed FAISS vector store of Indian laws and cases.
    Returns a list of Document objects, filtered to the most relevant matches.
    """
    global faiss_db
    try:
        db = get_faiss_db()
        retrieved = db.similarity_search_with_score(query, k=8)
        if not retrieved:
            return []

        # similarity_search_with_score returns L2 distance (lower = more similar).
        # Empirically measured on this index (all-MiniLM-L6-v2): genuinely relevant
        # legal matches score ~0.7-1.15, while off-topic / out-of-corpus queries
        # (e.g. a current-events protest) score >1.3. We apply an ABSOLUTE cutoff
        # and deliberately keep NO minimum: if nothing is relevant, return an empty
        # list so unrelated documents never get surfaced as "sources".
        retrieved.sort(key=lambda x: x[1])
        kept = [doc for doc, dist in retrieved if dist <= RELEVANCE_MAX_DISTANCE]
        return kept[:5]

    except FileNotFoundError:
        return [Document(page_content="FAISS index not found.")]
    except Exception as e:
        return [Document(page_content=f"Error during legal database search: {e}")]


# -------------------------
# Research Function
# -------------------------
def perform_research(state: AgentState) -> dict:
    print("---PERFORMING RESEARCH---")
    query = state["query"]

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set.")

    web_search_tool = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)

    rag_chain = RunnableParallel(
        {
            "faiss_search_results": lambda x: legal_database_search.invoke(x["query"]),
            "web_search_results": lambda x: web_search_tool.invoke(x["query"]),
        }
    )

    results = rag_chain.invoke({"query": query})

    faiss_docs = results.get("faiss_search_results", [])
    web_results = results.get("web_search_results", [])

    sources = []
    faiss_content = ""

    for doc in faiss_docs:
        faiss_content += doc.page_content + "\n\n"
        if doc.metadata:
            sources.append({"type": "document", "metadata": doc.metadata})

    # TavilySearch returns a dict like {"results": [{"title","url","content"}, ...]}.
    # Older code iterated the dict directly, which yielded its keys ("query",
    # "answer", ...) instead of the actual results. Normalize both shapes here.
    if isinstance(web_results, dict):
        web_items = web_results.get("results", [])
    elif isinstance(web_results, list):
        web_items = web_results
    else:
        web_items = []

    web_content = ""
    for item in web_items:
        if isinstance(item, dict):
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("content", "")
            # Skip social media / user-generated content - not citable legal sources.
            if url and any(bad in url.lower() for bad in BLOCKED_WEB_DOMAINS):
                continue
            web_content += f"{title}\n{snippet}\n\n"
            sources.append({"type": "web", "title": title, "url": url})
        else:
            web_content += str(item) + "\n\n"
            sources.append({"type": "web", "content": str(item)})

    print("---RESEARCH COMPLETE---")

    return {
        "faiss_search_results": faiss_content,
        "web_search_results": web_content,
        "sources": sources,
        "intermediate_steps": [
            f"FAISS Results: {faiss_content}",
            f"Web Results: {web_content}",
        ],
    }
