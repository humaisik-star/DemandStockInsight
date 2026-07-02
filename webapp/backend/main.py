"""FastAPI backend for the Demand & Stock Assistant.

Exposes a /chat endpoint that runs the Azure OpenAI function-calling loop over
the tools in tools.py. CORS is open so a Vercel-hosted frontend can call it.

Env vars (set in Azure Container Apps / locally in .env):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_API_VERSION (optional)
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tools import TOOL_SPECS, dispatch

STATIC_DIR = Path(__file__).parent / "static"

SYSTEM_PROMPT = """You are a demand-planning and inventory assistant for a retail chain.
Answer questions about product demand forecasts and stock recommendations by calling
the provided tools — never invent numbers. Store IDs look like S001..S005 and product
IDs like P0001..P0020. When the user is vague, call list_series or inventory_summary to
orient yourself. Keep answers concise and business-focused: state the number, the unit,
and a one-line recommendation. Reply in the user's language."""

app = FastAPI(title="Demand & Stock Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_client = None


def client():
    """Lazily build the Azure OpenAI client so /health works without creds."""
    global _client
    if _client is None:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return _client


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str] = []


def run_turn(messages):
    """Resolve tool calls until the model returns a text answer."""
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    tools_used = []
    while True:
        resp = client().chat.completions.create(
            model=deployment, messages=messages, tools=TOOL_SPECS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, tools_used

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tools_used.append(tc.function.name)
            result = dispatch(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Demand & Stock Assistant API. POST /chat to talk."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    """Return JSON (not a raw HTML 500) so the frontend can always parse it."""
    return JSONResponse(
        status_code=200,
        content={
            "answer": "Şu an bir sorun oluştu, lütfen birkaç saniye sonra tekrar deneyin.",
            "tools_used": [],
            "error": str(exc),
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": t.role, "content": t.content} for t in req.history]
        messages.append({"role": "user", "content": req.message})
        answer, tools_used = run_turn(messages)
        return ChatResponse(answer=answer or "", tools_used=tools_used)
    except Exception as e:
        # Never surface a raw 500; hand the frontend a clean JSON message.
        return ChatResponse(
            answer=f"İstek işlenemedi ({type(e).__name__}). Model uyanıyor olabilir, tekrar deneyin.",
            tools_used=[],
        )
