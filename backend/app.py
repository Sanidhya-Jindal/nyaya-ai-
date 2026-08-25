import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Import your agent router ---
from agent.router import route_query
from db import (
    save_thread,
    save_message,
    get_user_threads,
    get_thread_messages,
    delete_thread,
)

# ============================
# 1. FASTAPI INITIALIZATION
# ============================

app = FastAPI(
    title="L.A.R.A. Backend API",
    description="API for the Legal Analysis & Research Assistant",
    version="1.0.0",
)

# ============================
# 2. CORS CONFIGURATION (FIXED)
# ============================

# IMPORTANT:
# - 5174 is your frontend port
# - both localhost and 127.0.0.1 MUST be allowed
# - exact match is required by browser

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8010",
        "http://127.0.0.1:8010",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 3. PYDANTIC MODELS
# ============================

class QueryRequest(BaseModel):
    user_query: str
    role: str
    thread_id: str


class QueryResponse(BaseModel):
    final_analysis: str
    thread_id: str
    evaluation: str | None = None
    sources: list[str] | None = None


class ChatHistoryRequest(BaseModel):
    user_id: str


class ThreadMessagesRequest(BaseModel):
    thread_id: str


class SaveThreadRequest(BaseModel):
    user_id: str
    thread_id: str
    title: str | None = None


# ============================
# 4. API ENDPOINTS
# ============================

def _extract_evaluation(result: dict) -> str | None:
    """Pull a human-readable evaluation summary from either agent's result."""
    # Citizen agent: pre-formatted evaluation string
    evaluation = result.get("evaluation_score")
    if evaluation and str(evaluation).strip():
        return str(evaluation).strip()

    # Lawyer agent: a numeric 0-1 quality score
    score = result.get("answer_score")
    if isinstance(score, (int, float)):
        return (
            f"⭐ Overall Quality Score: {round(score * 5, 2)} / 5\n\n"
            "⚠️ Note: This reflects response quality, not verified legal accuracy. "
            "Always verify cited acts, sections and cases against official sources."
        )

    return None


def _extract_sources(result: dict) -> list[str] | None:
    """Normalize the different source shapes into a clean list of display strings."""
    display: list[str] = []

    # Citizen agent: list of {"type": "document"/"web", ...}
    for s in result.get("sources", []) or []:
        if not isinstance(s, dict):
            display.append(str(s)[:250])
            continue
        if s.get("type") == "document":
            md = s.get("metadata", {}) or {}
            name = md.get("case_name") or md.get("source") or "Legal document"
            display.append(f"📄 {name}")
        elif s.get("type") == "web":
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            label = title or (str(s.get("content") or "")[:80]).strip() or "Web result"
            display.append(f"🌐 {label}" + (f" — {url}" if url else ""))
        else:
            display.append(str(s)[:250])

    # Lawyer agent: retrieved_context is a list of raw context strings
    if not display:
        for ctx in result.get("retrieved_context", []) or []:
            snippet = str(ctx).strip().replace("\n", " ")
            if snippet:
                display.append(f"📄 {snippet[:250]}")

    # De-duplicate while preserving order, and cap to keep the UI readable.
    seen = set()
    deduped = []
    for item in display:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:12] or None


@app.get("/")
def read_root():
    return {"message": "LARA backend is running."}


@app.post("/process_query", response_model=QueryResponse)
async def process_legal_query(request: QueryRequest):
    """
    Main endpoint called by frontend
    """
    try:
        result = route_query(
            role=request.role,
            user_query=request.user_query,
            thread_id=request.thread_id,
        )

        final_analysis = result.get(
            "final_analysis",
            "Sorry, no analysis could be generated.",
        )

        # --- Explainable-AI fields (only present for real legal queries) ---
        evaluation = _extract_evaluation(result)
        sources = _extract_sources(result)

        # Save chat (store only the clean conversational answer)
        save_message(request.thread_id, "user", request.user_query)
        save_message(request.thread_id, "bot", final_analysis)

        return QueryResponse(
            final_analysis=final_analysis,
            thread_id=request.thread_id,
            evaluation=evaluation,
            sources=sources,
        )

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/get_chat_history")
async def get_chat_history(request: ChatHistoryRequest):
    try:
        threads = get_user_threads(request.user_id)
        return {"threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_thread_messages")
async def get_thread_messages_endpoint(request: ThreadMessagesRequest):
    try:
        messages = get_thread_messages(request.thread_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_thread")
async def save_thread_endpoint(request: SaveThreadRequest):
    try:
        save_thread(request.user_id, request.thread_id, request.title)
        return {"message": "Thread saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_thread/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    try:
        delete_thread(thread_id)
        return {"message": "Thread deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# 5. LOCAL SERVER ENTRY POINT
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
    )
