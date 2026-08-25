import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agent.citizen_agent import app as citizen_app
from agent.lawyer_agent import lawyer_app as lawyer_app
from db import get_thread_messages

load_dotenv()


def _format_history(chat_history, max_turns=6):
    """Render recent conversation turns as plain text for prompting."""
    if not chat_history:
        return ""
    recent = chat_history[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = m.get("role") or m.get("type") or "user"
        who = "User" if role in ("user", "human") else "L.A.R.A"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{who}: {content[:500]}")
    return "\n".join(lines)


# --- Intent Gate ---
def handle_non_legal_message(user_query: str, chat_history=None):
    """
    Classify the user's message before running the heavy research pipeline.

    Greetings, thanks, and other small talk should NOT trigger a full legal
    analysis (which otherwise hallucinates a legal query out of a simple
    "hi how are you"). Returns a short conversational reply for such messages,
    or None if this is a genuine legal question that should go to the agents.

    The recent conversation is passed in so that follow-ups in an ongoing legal
    chat (e.g. "explain more", "why?") are treated as legal, not small talk.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_retries=5,
    )

    history_text = _format_history(chat_history) or "(no prior messages)"

    prompt = PromptTemplate(
        template=(
            "You are L.A.R.A, an AI assistant that ONLY covers INDIAN law.\n"
            "Classify the user's LATEST message using the conversation for context, into one category:\n"
            "- 'greeting': a greeting, thanks, or small talk that is not a legal question.\n"
            "- 'out_of_scope': a legal question about the law of ANOTHER country/jurisdiction "
            "(e.g. US, California, UK, EU law) with no Indian-law angle, OR a request that is clearly "
            "outside legal help entirely. (Questions about Indian law that merely mention a foreign "
            "country, like extradition to India, are NOT out_of_scope - they are indian_law.)\n"
            "- 'indian_law': a question about Indian law, or a follow-up to an ongoing legal discussion "
            "(e.g. 'explain more', 'what about section 5', 'why?').\n\n"
            "For 'greeting': write a short, warm, natural reply (do NOT greet again if the conversation "
            "already has prior messages).\n"
            "For 'out_of_scope': write a brief, polite reply explaining that you focus specifically on "
            "Indian law and cannot reliably advise on that jurisdiction, and invite an Indian-law question.\n"
            "For 'indian_law': leave reply empty.\n\n"
            'Respond ONLY as JSON: {{"category": "greeting|out_of_scope|indian_law", "reply": "..."}}\n\n'
            "Conversation so far:\n{history}\n\n"
            "User's latest message: {query}\n"
            "JSON:"
        ),
        input_variables=["history", "query"],
    )

    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"history": history_text, "query": user_query})
        if isinstance(result, dict):
            category = str(result.get("category", "indian_law")).strip().lower()
            if category in ("greeting", "out_of_scope"):
                reply = (result.get("reply") or "").strip()
                if not reply:
                    if category == "out_of_scope":
                        reply = (
                            "I focus specifically on Indian law, so I can't reliably advise on "
                            "that. If you have a question about Indian law, I'd be glad to help."
                        )
                    else:
                        reply = (
                            "I'm here to help with Indian law - ask me anything about acts, "
                            "sections, cases, rights or procedures."
                        )
                return reply
    except Exception as e:
        # If classification fails, fall through and treat it as a legal query.
        print(f"Intent classification failed, proceeding as legal query. Error: {e}")

    return None


# --- Routing Logic ---
def route_query(role: str, user_query: str, thread_id: str):
    """
    Routes the user's query to the correct agent based on their selected role.

    Args:
        role (str): The role selected by the user ("Common Citizen" or "Lawyer").
        user_query (str): The user's input query.
        thread_id (str): The unique identifier for the conversation thread.

    Returns:
        The response from the invoked agent.
    """
    # Load existing chat history for the thread (used for continuity AND for the
    # context-aware intent gate).
    existing_messages = get_thread_messages(thread_id)
    chat_history = []
    for msg in existing_messages:
        if msg['role'] == 'user':
            chat_history.append({"role": "user", "content": msg['content']})
        elif msg['role'] == 'bot':
            chat_history.append({"role": "assistant", "content": msg['content']})

    # --- Intent gate: short-circuit greetings / small talk ---
    casual_reply = handle_non_legal_message(user_query, chat_history)
    if casual_reply is not None:
        print("---INTENT: Non-legal / conversational message. Skipping research pipeline.---")
        return {"final_analysis": casual_reply}

    # Create the initial state with the query and role
    input_state = {
        "query": user_query,
        "chat_history": chat_history,
        "intermediate_steps": [],
        "role": role,  # <-- Pass the role into the state
    }

    # IMPORTANT: use a fresh, unique checkpointer thread for every run.
    # The graph's `sources` / `intermediate_steps` / `research_cycles` fields use
    # `operator.add` reducers; reusing the same thread_id across turns made the
    # MemorySaver checkpointer ACCUMULATE the previous question's sources into the
    # next answer (e.g. child-marriage sources showing up under a protest query).
    # Conversational continuity is preserved separately via `chat_history`.
    run_config = {
        "configurable": {"thread_id": f"{thread_id}::{uuid.uuid4()}"},
        "recursion_limit": 50,
    }

    # Normalize role to avoid case-sensitivity issues between frontend and backend
    normalized_role = (role or '').strip().lower()

    if normalized_role == "lawyer":
        print("Routing to Lawyer Agent...")
        result = lawyer_app.invoke(input_state, config=run_config)
    elif normalized_role == "citizen":
        print("Routing to Citizen Agent...")
        result = citizen_app.invoke(input_state, config=run_config)
    else:
        # If role is unrecognized, default to Citizen behavior but log a warning.
        print(f"Warning: Unrecognized role '{role}' received. Defaulting to Citizen Agent.")
        result = citizen_app.invoke(input_state, config=run_config)

    # NOTE: message persistence is handled by the API layer (app.py), which saves
    # exactly one user message and one bot message per turn. The previous re-save
    # loop here duplicated the entire history on every turn.
    return result
