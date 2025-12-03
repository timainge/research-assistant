"""Data models for research orchestration."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Status of a research task."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ClarificationRequest(BaseModel):
    """Request for user clarification before proceeding."""

    questions: list[str] = Field(
        description="Questions to ask the user for clarification"
    )
    context: str = Field(
        default="", description="Context explaining why clarification is needed"
    )


class ResearchTask(BaseModel):
    """A single deep research task."""

    id: str = Field(description="Unique task identifier")
    query: str = Field(description="The research query/question")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=0, description="Higher = more important")
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of tasks this depends on"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    """A citation/source reference."""

    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Source title")
    snippet: str = Field(default="", description="Relevant excerpt")


class ResearchResult(BaseModel):
    """Result from a deep research task."""

    task_id: str = Field(description="ID of the task this result is for")
    content: str = Field(description="The research report content")
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class PlanningOutput(BaseModel):
    """Output from the planning step."""

    needs_clarification: bool = Field(
        default=False, description="Whether user clarification is needed"
    )
    clarification: Optional[ClarificationRequest] = Field(
        default=None, description="Clarification request if needed"
    )
    tasks: list[ResearchTask] = Field(
        default_factory=list, description="Research tasks to execute"
    )
    reasoning: str = Field(
        default="", description="Explanation of the planning decision"
    )


class OrchestratorState(BaseModel):
    """Current state of the research orchestrator."""

    original_question: str = Field(description="The user's original question")
    tasks: dict[str, ResearchTask] = Field(default_factory=dict)
    results: dict[str, ResearchResult] = Field(default_factory=dict)
    iteration: int = Field(default=0)
    status: str = Field(default="planning")

