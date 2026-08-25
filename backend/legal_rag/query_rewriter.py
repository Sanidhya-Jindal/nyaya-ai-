# LARA/legal_rag/query_rewriter.py

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    role: str
    chat_history: List[Any]


def _format_history(chat_history, max_turns: int = 6) -> str:
    """Render recent conversation turns as plain text for prompting."""
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]
    lines = []
    for m in recent:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or "user"
            content = m.get("content", "")
        else:
            role = getattr(m, "type", "user")
            content = getattr(m, "content", str(m))
        who = "User" if str(role).lower() in ("user", "human") else "L.A.R.A"
        content = str(content).strip()
        if content:
            lines.append(f"{who}: {content[:500]}")
    return "\n".join(lines)


def rewrite_query(state: AgentState) -> dict:
    """
    Rewrites the user's latest message into a precise, SELF-CONTAINED legal search
    query, resolving follow-up references ("it", "that", "the punishment", ...)
    using the conversation history so retrieval stays on-topic across turns.
    """
    print("---REWRITING QUERY---")
    query = state["query"]
    role = state.get("role", "Common Citizen")
    history_text = _format_history(state.get("chat_history", []))
    history_block = history_text if history_text else "(no prior messages)"

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_retries=5,
    )

    if role == "Lawyer":
        audience = "a lawyer seeking precedents, relevant acts and arguments"
    else:
        audience = "a common citizen seeking clear legal information"

    rewrite_prompt_template = """You are a legal research assistant specializing in Indian law, helping {audience}.
Using the conversation for context, rewrite the user's LATEST message into a precise, SELF-CONTAINED, legally-focused search query.

CRITICAL: Resolve any references such as "it", "that", "this", "more", "the section", "the punishment", "the act" using the earlier conversation, so the query makes complete sense on its own.
Example: if the user earlier asked about "Section 302 IPC" and now says "what is the punishment for it", rewrite to "punishment for murder under Section 302 of the Indian Penal Code".
If the latest message is already self-contained, just tighten it into a good search query.

Conversation so far:
{history}

User's latest message: {query}

Output MUST be JSON in this format:
{{ "rewritten_query": "<your query here>" }}
JSON Output:"""

    rewrite_prompt = PromptTemplate(
        template=rewrite_prompt_template,
        input_variables=["audience", "history", "query"],
    )

    try:
        rewriter_chain = rewrite_prompt | llm | JsonOutputParser()
        parsed = rewriter_chain.invoke(
            {"audience": audience, "history": history_block, "query": query}
        )
        rewritten_query = parsed.get("rewritten_query", query) if isinstance(parsed, dict) else str(parsed)
    except Exception as e:
        print(f"Query rewrite failed, using original query. Error: {e}")
        rewritten_query = query

    # Fall back to the original query if the rewrite came back empty
    if not rewritten_query or not str(rewritten_query).strip():
        rewritten_query = query

    print(f"Rewritten Query: {rewritten_query}")
    # Update the shared state's `query` field so downstream nodes (retriever)
    # actually use the rewritten query instead of the raw user input.
    return {"query": str(rewritten_query)}
