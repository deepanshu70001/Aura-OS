# app/services/llm_langchain.py
from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field

from app.core.config import settings

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_groq import ChatGroq
except Exception as e:  # pragma: no cover
    print(f"Failed to import Langchain components: {e}")
    PromptTemplate = None
    JsonOutputParser = None
    ChatGroq = None


class MicroQuest(BaseModel):
    title: str = Field(description="A very short, 2-5 word title for the micro-task")
    action: str = Field(
        description="A simple, non-intimidating action step taking less than 2 minutes"
    )


class TaskList(BaseModel):
    tasks: list[MicroQuest] = Field(description="List of shattered micro-quests")


task_parser = JsonOutputParser(pydantic_object=TaskList) if JsonOutputParser else None


@lru_cache(maxsize=1)
def _get_llm():
    if ChatGroq is None or not settings.GROQ_API_KEY:
        return None

    try:
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.6,
            max_tokens=500,
        )
    except Exception:
        return None


def _fallback_microquests(large_task: str) -> dict:
    task = large_task.strip()
    head = task.split(" ", 1)[0] if task else "task"
    return {
        "tasks": [
            {
                "title": "Open workspace",
                "action": f"Open the folder or app where you will do: {task[:80] or head}.",
            },
            {
                "title": "Small first move",
                "action": "Do one 2-minute starter action and ignore everything else for now.",
            },
            {
                "title": "Lock next step",
                "action": "Write the immediate next action on paper so restarting is easier.",
            },
        ]
    }


def shatter_task(large_task: str) -> dict:
    """Breaks down an overwhelming task into bite-sized micro-quests."""
    if not large_task or not large_task.strip():
        return {"tasks": []}

    llm = _get_llm()
    if llm is None or PromptTemplate is None or task_parser is None:
        return _fallback_microquests(large_task)

    prompt = PromptTemplate(
        template=(
            "You are an empathetic ADHD initiation coach. The user is experiencing "
            "'Executive Freeze' and cannot start the following task: '{large_task}'. "
            "Break this task down into 3 to 5 extremely small, highly actionable, "
            "and non-intimidating micro-quests. The first step should take less than "
            "1 minute to complete to trigger a dopamine response.\n\n"
            "{format_instructions}"
        ),
        input_variables=["large_task"],
        partial_variables={"format_instructions": task_parser.get_format_instructions()},
    )

    try:
        chain = prompt | llm | task_parser
        return chain.invoke({"large_task": large_task})
    except Exception:
        return _fallback_microquests(large_task)


def generate_reframe(worry_text: str) -> str:
    """Turns a shattered worry block into a positive affirmation."""
    if not worry_text:
        return "One breath at a time. You are doing enough for this moment."

    llm = _get_llm()
    if llm is None or PromptTemplate is None:
        return "This moment is hard, but you are still moving forward."

    prompt = PromptTemplate(
        template=(
            "The user just physically shattered a block containing this worry: "
            "'{worry_text}'. Provide a single, short, grounding, and empathetic "
            "positive affirmation (max 15 words)."
        ),
        input_variables=["worry_text"],
    )

    try:
        chain = prompt | llm
        result = chain.invoke({"worry_text": worry_text})
        return str(result.content).strip()
    except Exception:
        return "You are not behind; you are regrouping, and that still counts."


def generate_clinical_summary(session_data: str) -> str:
    """Generates a professional summary for guardians."""
    if not session_data:
        return "Session data was empty. Recommend a calm check-in and hydration break."

    llm = _get_llm()
    if llm is None or PromptTemplate is None:
        return (
            "The user experienced elevated stress and used AuraOS tools for regulation. "
            "A supportive check-in is recommended, followed by one small next action."
        )

    prompt = PromptTemplate(
        template=(
            "Analyze this AuraOS session data from a user who experienced an anxiety spike:\n"
            "{session_data}\n\n"
            "Write a brief, professional, clinical 3-sentence summary for a guardian. "
            "Focus on trigger, regulation actions taken, and present state."
        ),
        input_variables=["session_data"],
    )

    try:
        chain = prompt | llm
        result = chain.invoke({"session_data": session_data})
        return str(result.content).strip()
    except Exception:
        return (
            "The session indicates acute stress with active self-regulation attempts. "
            "The user engaged coping tools and appears to be stabilizing. "
            "Recommend reduced task pressure and supportive follow-up."
        )
