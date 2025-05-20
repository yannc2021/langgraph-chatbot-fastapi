from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from agent.graph import graph

app = FastAPI()

class ChatRequest(BaseModel):
    messages: Any  # Accepts list of dicts or string
    thread_id: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        # Support both string and list of messages
        input_data = {"messages": request.messages, "thread_id": request.thread_id}
        result = await graph.ainvoke(input_data, config)

        messages = result.get("messages", [])
       
        running_summary = result.get("context", {}).get("running_summary", None)
        summary = running_summary.summary if running_summary else None

        return {
            "response": messages[-1].content if messages else None,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
