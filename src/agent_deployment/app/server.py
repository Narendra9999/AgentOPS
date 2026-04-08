"""
AgentOPS Databricks App — FastAPI server.

Exposes /invocations (ChatAgent-compatible) and / (health check).
Runs on uvicorn at 0.0.0.0:8000.
"""

import uvicorn
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agent_app import run_agent

app = FastAPI(title="AgentOPS Docs Chatbot", version="1.0.0")

CHAT_UI = Path(__file__).parent / "chat.html"


class ChatRequest(BaseModel):
    """Compatible with both ChatAgent and direct API calls."""
    messages: list[dict]
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    # Also accept custom_inputs for backward compatibility with Model Serving clients
    custom_inputs: Optional[dict] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    return CHAT_UI.read_text()


@app.get("/health")
async def health():
    import os
    import mlflow
    return {
        "status": "ok",
        "agent": "agentops-docs-chatbot",
        "mlflow_experiment_id": os.getenv("MLFLOW_EXPERIMENT_ID", "NOT SET"),
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "lakebase_project": os.getenv("LAKEBASE_AUTOSCALING_PROJECT", "NOT SET"),
        "serving_endpoint": os.getenv("SERVING_ENDPOINT_NAME", "NOT SET"),
        "vs_index": os.getenv("VS_INDEX", "NOT SET"),
    }


@app.post("/invocations")
async def invocations(request: ChatRequest):
    # Extract thread_id/user_id from either top-level or custom_inputs
    thread_id = request.thread_id
    user_id = request.user_id
    if request.custom_inputs:
        thread_id = thread_id or request.custom_inputs.get("thread_id")
        user_id = user_id or request.custom_inputs.get("user_id")

    try:
        result = await run_agent(
            messages=request.messages,
            thread_id=thread_id,
            user_id=user_id,
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()[-1000:]},
        )

    # Return in ChatAgent-compatible format
    return {
        "messages": [{"role": "assistant", "content": result["output"]}],
        "custom_outputs": {
            "thread_id": result["thread_id"],
            "user_id": result.get("user_id", ""),
        },
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
