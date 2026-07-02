# Web App — Demand & Stock Assistant

A chat UI (Vercel) talking to a FastAPI backend (Azure Container Apps) that runs
the Azure OpenAI function-calling assistant over the model outputs.

```
[Vercel: frontend/index.html]  →  [Container Apps: backend/main.py]  →  [Azure OpenAI gpt-5-mini + tools]
```

## backend/ — FastAPI

- `main.py` — `/chat` (POST) runs the tool-calling loop; `/health` (GET).
- `tools.py` — the tools the LLM can call (reads `data/*.csv`).
- `Dockerfile`, `requirements.txt` — container build.

Run locally:
```bash
cd backend
pip install -r requirements.txt
export AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
uvicorn main:app --reload
# POST http://127.0.0.1:8000/chat  {"message": "stok özetini ver"}
```

Deploy to Azure Container Apps (scale-to-zero):
```bash
cd backend
az containerapp up -n demand-assistant-api -g demand-stock-rg -l germanywestcentral \
  --source . --ingress external --target-port 8000 \
  --env-vars AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
# Set the API key as a SECRET (not a plain env var):
az containerapp secret set -n demand-assistant-api -g demand-stock-rg --secrets openai-key=<KEY>
az containerapp update -n demand-assistant-api -g demand-stock-rg \
  --set-env-vars AZURE_OPENAI_API_KEY=secretref:openai-key
```

## frontend/ — static chat UI (Vercel)

`index.html` is a self-contained chat page. Before deploying, replace
`__BACKEND_URL__` with your Container Apps URL (or pass `?api=https://...`).

Deploy:
```bash
cd frontend
npx vercel --prod        # or: import the folder in the Vercel dashboard
```
