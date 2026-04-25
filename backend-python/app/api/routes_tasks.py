# app/api/routes_tasks.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_langchain import shatter_task, generate_reframe

router = APIRouter()

# Schemas for incoming requests
class TaskRequest(BaseModel):
    task_description: str | None = None
    task: str | None = None

class WorryRequest(BaseModel):
    worry_text: str | None = None
    worry: str | None = None

@router.post("/shatter-task")
async def api_shatter_task(request: TaskRequest):
    try:
        large_task = (request.task_description or request.task or "").strip()
        if not large_task:
            raise HTTPException(status_code=400, detail="task_description or task is required")

        result = shatter_task(large_task)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phoenix-reframe")
async def api_phoenix_reframe(request: WorryRequest):
    try:
        worry_text = (request.worry_text or request.worry or "").strip()
        if not worry_text:
            raise HTTPException(status_code=400, detail="worry_text or worry is required")

        reframe = generate_reframe(worry_text)
        return {"success": True, "reframe": reframe}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
