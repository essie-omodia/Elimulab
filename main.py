"""ElimuLab AI service.

A small OpenAI-compatible proxy. The ElimuLab page calls this service, and this
service calls the real model (Ollama on your laptop, or any hosted provider that
offers an OpenAI-compatible API). The model key stays here, never in the page.

It stores nothing and does not log what students or teachers send.
"""
import os
import time
import collections

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---- settings (set these as environment variables on Render) ----
UPSTREAM_BASE = os.getenv("UPSTREAM_BASE", "http://localhost:11434/v1").rstrip("/")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "")        # not needed for Ollama
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "")    # forces one model for every call
EVENT_TOKEN = os.getenv("EVENT_TOKEN", "")          # optional shared password
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
RATE_MAX = int(os.getenv("RATE_MAX", "20"))         # requests per IP ...
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "600"))  # ... per this many seconds
MAX_CHARS = int(os.getenv("MAX_CHARS", "24000"))    # total characters accepted
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3500"))   # cap on the reply length
STRICT = os.getenv("STRICT", "1") == "1"            # only accept ElimuLab's own tasks

# The three system prompts the ElimuLab page sends. Anything else is refused,
# so the service cannot be used as a free general chatbot.
ALLOWED_SYSTEM = (
    "You convert textbook practicals into lab files",
    "You mark one answer",
    "You are a patient physics lab partner",
)

app = FastAPI(title="ElimuLab AI service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_hits = collections.defaultdict(collections.deque)


def _limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_MAX:
        return True
    q.append(now)
    return False


@app.get("/health")
def health():
    return {"ok": True, "model": UPSTREAM_MODEL or "chosen by the page"}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    if EVENT_TOKEN and req.headers.get("authorization", "") != f"Bearer {EVENT_TOKEN}":
        raise HTTPException(401, "Missing or wrong event token")

    ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[0].strip()
    if _limited(ip):
        raise HTTPException(429, "Too many requests. Please try again in a few minutes.")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "The request body must be JSON")

    msgs = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(msgs, list) or not msgs or len(msgs) > 6:
        raise HTTPException(400, "messages must be a short list")
    total = 0
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            raise HTTPException(400, "Each message needs a role and text content")
        total += len(m["content"])
    if total > MAX_CHARS:
        raise HTTPException(413, "That text is too long")

    if STRICT:
        system_text = next((m["content"] for m in msgs if m["role"] == "system"), "")
        if not system_text.startswith(ALLOWED_SYSTEM):
            raise HTTPException(403, "This service only handles ElimuLab lab tasks")

    try:
        wanted = int(body.get("max_tokens") or MAX_TOKENS)
    except (TypeError, ValueError):
        wanted = MAX_TOKENS
    try:
        temp = min(max(float(body.get("temperature", 0.3)), 0.0), 1.0)
    except (TypeError, ValueError):
        temp = 0.3

    payload = {
        "model": UPSTREAM_MODEL or body.get("model") or "llama3.2",
        "messages": msgs,
        "stream": False,
        "max_tokens": min(wanted, MAX_TOKENS),
        "temperature": temp,
    }
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_KEY}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{UPSTREAM_BASE}/chat/completions", json=payload, headers=headers)
    except httpx.HTTPError:
        raise HTTPException(502, "The AI model could not be reached")
    if r.status_code >= 400:
        raise HTTPException(502, f"The AI model returned an error ({r.status_code})")
    try:
        return JSONResponse(r.json())
    except Exception:
        raise HTTPException(502, "The AI model sent a reply that could not be read")
