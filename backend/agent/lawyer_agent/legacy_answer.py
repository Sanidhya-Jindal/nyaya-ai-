import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()


def get_llm():
    """Initialize Groq LLM with env key."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0.2, groq_api_key=groq_api_key, max_retries=5
    )


def safe_invoke(llm, prompt, vars):
    """Run a prompt safely and return text content."""
    chain = prompt | llm
    result = chain.invoke(vars)
    return getattr(result, "content", str(result))


def summarize_long_text(text: str, label: str, query: str) -> str:
    """Summarize text with fast strategy."""
    if not text:
        return f"No {label} found."

    llm = get_llm()

    # Fast mode: just trim input
    trimmed = " ".join(text.split()[:2000])
    prompt = PromptTemplate(
        template="""Summarize the following {label} (<200 words), focusing on acts, sections, judgments.

        Query: {query}
        Text: {text}

        Summary:""",
        input_variables=["query", "text", "label"],
    )
    return safe_invoke(llm, prompt, {"query": query, "text": trimmed, "label": label})


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
        who = "Lawyer" if str(role).lower() in ("user", "human") else "Nyaya AI"
        content = str(content).strip()
        if content:
            lines.append(f"{who}: {content[:500]}")
    return "\n".join(lines)


def generate_legacy_lawyer_answer(query: str, all_steps: str, chat_history=None) -> str:
    """Generates a conversational, grounded legal briefing for a lawyer."""
    print("---GENERATING LEGACY LAWYER ANALYSIS REPORT---")

    # Compress steps if too long
    if len(all_steps.split()) > 1500:
        all_steps = summarize_long_text(all_steps, "research steps", query)

    history_text = _format_history(chat_history)
    history_block = history_text if history_text else "(This is the first message in the conversation.)"

    llm = get_llm()
    analysis_prompt = PromptTemplate(
        template="""You are Nyaya AI, an expert legal assistant helping a lawyer with Indian law, in an ongoing chat.

Conversation so far:
{history}

The lawyer's latest message: {query}

Research findings gathered for this message (your ONLY source of factual grounding):
{all_steps}

Write your reply following these rules:
1. CONTINUITY: Continue naturally. Only greet or introduce yourself on the first message; otherwise do not greet.
2. FOLLOW-UPS: If the latest message is a follow-up, interpret it using the earlier messages.
3. GROUNDING: Cite statutes, sections, articles and case names ONLY if they appear in the research findings. Do not rely on your own memory for citations.
4. NO HALLUCINATION: Never invent or guess a case name, citation, section or article. If the findings lack something, say so plainly rather than fabricating.
5. STYLE: Clear professional prose, no rigid report headers. Offer actionable arguments where useful; as detailed as the case warrants without padding.

Your conversational legal briefing:""",
        input_variables=["history", "query", "all_steps"],
    )

    final_analysis = safe_invoke(
        llm,
        analysis_prompt,
        {"history": history_block, "query": query, "all_steps": all_steps},
    )

    return final_analysis
