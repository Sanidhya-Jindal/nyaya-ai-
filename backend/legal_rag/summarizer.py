import os
import operator
import asyncio  # noqa: F401
import json
from typing import TypedDict, Annotated, List, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage  # noqa: F401

# Imports for Hybrid Evaluation
from sentence_transformers import SentenceTransformer, util

load_dotenv()

# ------------------------------
# Config
# ------------------------------
FAST_MODE = True  # ✅ Toggle True = faster (trims), False = detailed chunking
CHUNK_SIZE = 1200
MAX_CHUNKS = 3

# ------------------------------
# Global Embedding Model (for evaluation)
# ------------------------------
# Load the model once when the server starts
try:
    EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model for evaluation loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    EMBEDDING_MODEL = None
# ------------------------------


# ------------------------------
# Agent State (Full definition for reference)
# ------------------------------
class AgentState(TypedDict):
    query: str
    intermediate_steps: Annotated[List[Any], operator.add]
    web_search_results: str
    faiss_search_results: str
    final_analysis: str
    research_complete: bool
    chat_history: List[BaseMessage]
    sources: Annotated[List[dict], operator.add]
    role: str
    research_cycles: Annotated[int, operator.add]
    evaluation_score: str # This will store the formatted evaluation string


# ------------------------------
# Utility Functions
# ------------------------------
def get_llm():
    """Initialize Groq LLM with env key."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=groq_api_key, max_retries=5
    )


def safe_invoke(llm, prompt, vars):
    """Run a prompt safely and return text content."""
    chain = prompt | llm
    result = chain.invoke(vars)
    return getattr(result, "content", str(result))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, max_chunks: int = MAX_CHUNKS):
    """Split text into word chunks, capped to max_chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if len(chunks) >= max_chunks:
            break
    return chunks


def summarize_long_text(text: str, label: str, query: str) -> str:
    """Summarize text with fast or detailed strategy."""
    if not text:
        return f"No {label} found."

    llm = get_llm()

    # ✅ Fast mode: just trim input (kept small to respect Groq free-tier TPM limits)
    if FAST_MODE:
        trimmed = " ".join(text.split()[:900])
        prompt = PromptTemplate(
            template=f"""Summarize the following {label} (<200 words), focusing on acts, sections, judgments.

            Query: {{query}}
            Text: {{text}}

            Summary:""",
            input_variables=["query", "text"],
        )
        return safe_invoke(llm, prompt, {"query": query, "text": trimmed})

    # ✅ Detailed mode: chunk + merge
    chunk_summaries = []
    for chunk in chunk_text(text):
        prompt = PromptTemplate(
            template=f"""Summarize this {label} chunk (<120 words),
            focusing only on acts, sections, judgments.

            Query: {{query}}
            {label} chunk: {{chunk}}

            Summary:""",
            input_variables=["query", "chunk"],
        )
        chunk_summaries.append(
            safe_invoke(llm, prompt, {"query": query, "chunk": chunk})
        )

    merge_prompt = PromptTemplate(
        template=f"""Combine the following {label} summaries into one concise digest (<250 words):

        Query: {{query}}
        Summaries:
        {{summaries}}

        Final Digest:""",
        input_variables=["query", "summaries"],
    )
    return safe_invoke(
        llm, merge_prompt, {"query": query, "summaries": "\n".join(chunk_summaries)}
    )


# ------------------------------
# Citizen-focused Functions
# ------------------------------
def summarize_and_reflect(state: AgentState) -> dict:
    """Summarizes the findings and reflects on the research to identify gaps."""
    print("---SUMMARIZING & REFLECTING---")
    query = state["query"]

    faiss_summary = summarize_long_text(
        state["faiss_search_results"], "FAISS results", query
    )
    web_summary = summarize_long_text(state["web_search_results"], "Web results", query)

    llm = get_llm()
    summary_prompt = PromptTemplate(
        template="""Based on the original query and the following summarized search results,
        provide a concise reflection.

        Original Query: {query}
        FAISS Summary: {faiss_summary}
        Web Summary: {web_summary}

        Reflection:
        1. Key findings:
        2. Knowledge Gaps:
        3. Research complete? ('YES' or 'NO')""",
        input_variables=["query", "faiss_summary", "web_summary"],
    )

    summary = safe_invoke(
        llm,
        summary_prompt,
        {"query": query, "faiss_summary": faiss_summary, "web_summary": web_summary},
    )

    is_complete = "YES" in summary.upper()

    return {
        "intermediate_steps": state["intermediate_steps"] + [summary],
        "research_complete": is_complete,
    }


def format_chat_history(chat_history, max_turns: int = 6) -> str:
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
        who = "User" if str(role).lower() in ("user", "human") else "Nyaya AI"
        content = str(content).strip()
        if content:
            lines.append(f"{who}: {content[:500]}")
    return "\n".join(lines)


def generate_final_analysis(state: AgentState) -> dict:
    """Generates the final, conversational answer, grounded in the research."""
    print("---GENERATING FINAL ANALYSIS---")
    query = state["query"]

    # If BOTH the legal document search and the web search came back empty, there
    # is nothing to ground an answer on. Rather than let the model invent plausible
    # (but fabricated) facts, return an honest message. This is common for very
    # recent / current-events topics that aren't in the corpus.
    faiss_res = (state.get("faiss_search_results") or "").strip()
    web_res = (state.get("web_search_results") or "").strip()
    if not faiss_res and not web_res:
        print("---NO GROUNDING FOUND: returning honest fallback---")
        return {
            "final_analysis": (
                "I couldn't find reliable information on this in my legal document set "
                "or through a web search just now. This can happen with very recent or "
                "current-events topics that aren't in my sources. For up-to-date "
                "developments, please check reputable news outlets or official government "
                "notifications. If it helps, I can still explain the underlying legal "
                "principles involved - just ask."
            )
        }

    all_steps = "\n".join(state["intermediate_steps"])

    # Compress steps if too long
    if len(all_steps.split()) > 1500:
        all_steps = summarize_long_text(all_steps, "research steps", query)

    history_text = format_chat_history(state.get("chat_history", []))
    history_block = history_text if history_text else "(This is the first message in the conversation.)"

    llm = get_llm()
    analysis_prompt = PromptTemplate(
        template="""You are Nyaya AI, a friendly and knowledgeable legal assistant for Indian law, having an ongoing chat with a user.

Conversation so far:
{history}

The user's latest message: {query}

Research findings gathered for this message (this is your ONLY source of factual grounding):
{all_steps}

Write your reply following these rules carefully:
1. CONTINUITY: Continue the conversation naturally. Only greet or introduce yourself if this is the first message. If there are earlier messages, do NOT say "Hello" or re-introduce yourself - just answer.
2. FOLLOW-UPS: If the latest message is a follow-up ("explain more", "what about X", "why?"), interpret it using the earlier messages.
3. GROUNDING: State legal facts, act names, section numbers, article numbers, and case names ONLY if they actually appear in the research findings above. Do NOT rely on your own memory for citations.
4. NO HALLUCINATION: Never invent or guess a case name, citation, section, or article. If the findings lack a specific precedent or provision, say so plainly (e.g. "I couldn't find a specific case on this in my sources"). It is better to admit a gap than to fabricate.
5. STYLE: Clear, plain, conversational language. No rigid report headers. As concise as the question warrants.

Your reply:""",
        input_variables=["history", "query", "all_steps"],
    )

    final_analysis = safe_invoke(
        llm,
        analysis_prompt,
        {"history": history_block, "query": query, "all_steps": all_steps},
    )

    return {"final_analysis": final_analysis}


# ------------------------------
# New Lawyer-focused Functions
# ------------------------------
def summarize_and_reflect_lawyer(state: AgentState) -> dict:
    """
    Summarizes findings and reflects on the research from a lawyer's perspective.
    Identifies if enough legal precedents and arguments have been found.
    """
    print("---SUMMARIZING & REFLECTING FOR LAWYER---")
    query = state["query"]

    faiss_summary = summarize_long_text(
        state["faiss_search_results"], "FAISS results", query
    )
    web_summary = summarize_long_text(state["web_search_results"], "Web results", query)

    llm = get_llm()
    summary_prompt = PromptTemplate(
        template="""You are a legal research assistant for a lawyer. Based on the original query
        and the following summarized search results, provide a professional reflection.

        Original Query: {query}
        FAISS Summary: {faiss_summary}
        Web Summary: {web_summary}

        Reflection:
        1. **Key Legal Findings**: Highlight relevant statutes, case names, and legal principles.
        2. **Gaps in Research**: Identify missing information, such as conflicting judgments or lack of recent precedents.
        3. **Is the research complete?** ('YES' or 'NO')""",
        input_variables=["query", "faiss_summary", "web_summary"],
    )

    summary = safe_invoke(
        llm,
        summary_prompt,
        {"query": query, "faiss_summary": faiss_summary, "web_summary": web_summary},
    )

    is_complete = "YES" in summary.upper()

    return {
        "intermediate_steps": state["intermediate_steps"] + [summary],
        "research_complete": is_complete,
    }


def generate_lawyer_analysis(state: AgentState) -> dict:
    """Generates a structured legal analysis report for a lawyer."""
    print("---GENERATING LAWYER ANALYSIS REPORT---")
    query = state["query"]
    all_steps = "\n".join(state["intermediate_steps"])

    # Compress steps if too long
    if len(all_steps.split()) > 1500:
        all_steps = summarize_long_text(all_steps, "research steps", query)

    llm = get_llm()
    analysis_prompt = PromptTemplate(
        template="""You are an expert legal assistant. Based on the lawyer's case details
        and the research steps below, generate a professional legal analysis.

        - **Original Case Details**: A summary of the query provided by the lawyer.
        - **Relevant Statutes & Acts**: List of key legal provisions from Indian Law.
        - **Past Case Precedents & Judgments**: A detailed summary of related case studies with names and citations.
        - **Key Legal Arguments & Points**: Actionable points and arguments derived from the research.
        - **Sources**: A clear list of all web pages and internal documents used.

        Case Details: {query}
        Research Steps: {all_steps}

        Final Legal Analysis:""",
        input_variables=["query", "all_steps"],
    )

    final_analysis = safe_invoke(
        llm, analysis_prompt, {"query": query, "all_steps": all_steps}
    )

    return {"final_analysis": final_analysis}


# ----------------------------------------------------
# HYBRID EVALUATION HELPER FUNCTION
# ----------------------------------------------------
def evaluate_analysis(llm, query: str, all_steps: str, analysis: str) -> str:
    """
    Evaluates the generated analysis using a hybrid method:
    - Gets LLM-based scores (Relevance, Context Faithfulness, Clarity)
    - Computes semantic similarity between query/context and analysis
    - Combines them into a final confidence score
    """

    print("---EVALUATING FINAL ANALYSIS (HYBRID CONFIDENCE METHOD)---")

    # Step 1: Ask LLM to provide structured numeric scores
    eval_prompt = PromptTemplate(
        template="""You are a strict legal analysis evaluator.
        Evaluate the [Final Analysis] based on the [Original Query] and [Research Context].
        Provide only numeric values in JSON format.

        [Original Query]: {query}
        [Research Context]: {all_steps}
        [Final Analysis]: {analysis}

        Return strictly in JSON:
        {{
            "relevance_score": (1-5),
            "context_faithfulness_score": (1-5),
            "clarity_score": (1-5),
            "justification": "Short justification for the scores."
        }}
        """,
        input_variables=["query", "all_steps", "analysis"],
    )

    evaluation = safe_invoke(
        llm,
        eval_prompt,
        {"query": query, "all_steps": all_steps, "analysis": analysis},
    )

    # Parse JSON safely
    try:
        eval_data = json.loads(evaluation)
        relevance = float(eval_data.get("relevance_score", 3))
        faithfulness = float(eval_data.get("context_faithfulness_score", 3))
        clarity = float(eval_data.get("clarity_score", 3))
        justification = eval_data.get("justification", "")
    except Exception:
        relevance = faithfulness = clarity = 3.0
        justification = "Failed to parse evaluation; fallback scores applied."

    # Step 2: Compute semantic similarities (Using GLOBAL model for performance)
    if EMBEDDING_MODEL:
        emb_query = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
        emb_context = EMBEDDING_MODEL.encode(all_steps, convert_to_tensor=True)
        emb_analysis = EMBEDDING_MODEL.encode(analysis, convert_to_tensor=True)

        relevance_sim = util.cos_sim(emb_analysis, emb_query).item()
        context_sim = util.cos_sim(emb_analysis, emb_context).item()
        
        # Step 3: Compute normalized semantic similarity (scale 1–5)
        semantic_confidence = ((relevance_sim + context_sim) / 2) * 5
    else:
        semantic_confidence = 3.0 # Fallback score if model failed to load

    # Step 4: Weighted hybrid score
    llm_score = (0.4 * relevance) + (0.4 * faithfulness) + (0.2 * clarity)
    final_confidence = (0.6 * llm_score + 0.4 * semantic_confidence)
    final_confidence = round(min(final_confidence, 5.0), 2)

    # Step 5: Return formatted result
    result = f"""--- Response Quality (auto-assessed) ---
🔹 Relevance: {relevance}/5
🔹 Context Faithfulness: {faithfulness}/5
🔹 Clarity: {clarity}/5
🔹 LLM Weighted Average: {round(llm_score, 2)}/5
🔸 Semantic Similarity (Query + Context): {round(semantic_confidence, 2)}/5

⭐ Overall Quality Score: {final_confidence} / 5

💬 Justification: {justification}

⚠️ Note: This score reflects how relevant, faithful and clear the response is - it is NOT a guarantee of legal accuracy. Always verify any cited acts, sections and cases against official sources."""
    return result


# ----------------------------------------------------
# HYBRID EVALUATION GRAPH NODE
# ----------------------------------------------------
def evaluate_hybrid_response(state: AgentState) -> dict:
    """
    Graph node to evaluate the final analysis using the hybrid method.
    Calls the 'evaluate_analysis' helper function.
    """
    print("---STARTING HYBRID EVALUATION NODE---")
    
    query = state["query"]
    final_analysis = state["final_analysis"]
    
    # Re-create 'all_steps' from intermediate steps
    all_steps = "\n".join(state["intermediate_steps"])
    
    # Compress steps if too long (matching logic in generate_final_analysis)
    if len(all_steps.split()) > 1500:
        all_steps = summarize_long_text(all_steps, "research steps", query)

    if not final_analysis:
        print("---EVALUATION: No final analysis to evaluate.---")
        return {"evaluation_score": "Error: No analysis generated."}

    llm = get_llm() # Get the LLM instance
    
    # Call the helper function to get the formatted string
    evaluation_summary = evaluate_analysis(
        llm, 
        query=query, 
        all_steps=all_steps, 
        analysis=final_analysis
    )    
    # Return the raw summary string to be stored in the state
    return {"evaluation_score": evaluation_summary}


# ----------------------------------------------------
# FINAL COMBINATION GRAPH NODE
# ----------------------------------------------------
def combine_analysis_and_evaluation(state: AgentState) -> dict:
    """
    Appends the formatted evaluation summary to the end of the final analysis.
    This is the last step before sending the response to the user.
    """
    print("---COMBINING ANALYSIS AND EVALUATION---")

    # The conversational answer (final_analysis) and the evaluation summary are
    # now returned to the frontend as SEPARATE fields, so the chat bubble stays
    # clean and the evaluation is shown behind an expandable button instead.
    # This node is kept as a no-op to preserve the existing graph wiring.
    return {}