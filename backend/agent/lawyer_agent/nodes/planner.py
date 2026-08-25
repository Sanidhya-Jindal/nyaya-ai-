import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import json

from ..state import LawyerAgentState

load_dotenv()


def get_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=groq_api_key, max_retries=5
    )


def safe_invoke(llm, prompt, vars):
    chain = prompt | llm
    result = chain.invoke(vars)
    return getattr(result, "content", str(result))


def planner(state: LawyerAgentState) -> dict:
    """Convert issues and facts into 3-6 retrieval sub-queries."""
    print("---PLANNING RETRIEVAL SUB-QUERIES---")
    issues = state["issues"]
    facts = state["facts"]
    query = state["query"]

    issues_str = "\n".join(issues)
    facts_str = "\n".join(facts)

    llm = get_llm()
    prompt = PromptTemplate(
        template="""You are a legal research planner. Based on the legal issues and facts, generate 3-6 specific sub-queries for retrieving relevant legal information from databases and web searches.

Issues:
{issues}

Facts:
{facts}

Original Query: {query}

Generate sub-queries that focus on:
- Relevant statutes and sections
- Case precedents
- Legal principles
- Recent judgments

Output ONLY a JSON array of plain-string questions and NOTHING else (no prose, no markdown fences, no keys).
Example: ["What defenses are available under Section 138 NI Act?", "Key Supreme Court precedents on cheque bounce"]

JSON array:""",
        input_variables=["issues", "facts", "query"],
    )

    response = safe_invoke(
        llm, prompt, {"issues": issues_str, "facts": facts_str, "query": query}
    )

    sub_queries = _parse_sub_queries(response, fallback=query)
    print(f"Planned {len(sub_queries)} sub-queries: {sub_queries}")
    return {"sub_queries": sub_queries}


def _parse_sub_queries(response: str, fallback: str) -> list:
    """
    Robustly extract a clean list of sub-query strings from the LLM output.

    The model often wraps the JSON in prose or ```json fences, or returns a JSON
    object like {"sub_query1": "...", ...} instead of a plain list. The previous
    line-splitting fallback turned '[', '{', '```json' into "sub-queries", which
    polluted retrieval. This handles all those shapes and filters out junk.
    """
    candidates = []

    # 1) Try to find a JSON array anywhere in the response (even inside ``` fences).
    array_match = re.search(r"\[.*\]", response, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group(0))
            if isinstance(parsed, list):
                candidates = [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    # 2) Otherwise try a JSON object and take its string values.
    if not candidates:
        obj_match = re.search(r"\{.*\}", response, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
                if isinstance(parsed, dict):
                    candidates = [str(v) for v in parsed.values()]
            except json.JSONDecodeError:
                pass

    # 3) Last resort: split into lines and keep only sentence-like queries,
    #    dropping JSON syntax, fences, keys and boilerplate.
    if not candidates:
        for line in response.split("\n"):
            line = line.strip().strip(",").strip()
            # Drop obvious non-query lines
            if not line or line in ("[", "]", "{", "}") or line.startswith("```"):
                continue
            if line.lower().startswith(("here are", "sub-queries", "based on")):
                continue
            # Strip surrounding quotes / leading "sub_queryN:" style keys
            line = re.sub(r'^"?sub[_\s]?query\s*\d*"?\s*[:\-]\s*', "", line, flags=re.IGNORECASE)
            line = line.strip().strip('"').strip()
            if len(line) > 8:  # ignore stray fragments
                candidates.append(line)

    # Clean, de-duplicate, cap to 6. If nothing usable, fall back to the raw query.
    cleaned = []
    seen = set()
    for c in candidates:
        c = c.strip().strip('"').strip()
        if len(c) > 8 and c not in seen:
            seen.add(c)
            cleaned.append(c)

    return cleaned[:6] if cleaned else [fallback]
