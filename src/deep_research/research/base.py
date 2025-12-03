"""Base classes for research providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class ResearchCitation(BaseModel):
    """A citation from research results."""

    url: str
    title: str = ""
    snippet: str = ""


class WebSearchCall(BaseModel):
    """A web search call made during research."""

    query: str = Field(description="The search query")
    status: str = Field(default="completed", description="completed, failed, etc.")


class ReasoningSummary(BaseModel):
    """A reasoning summary from the research process."""

    text: str = Field(description="The reasoning text")


class ResearchResponse(BaseModel):
    """Standardized response from a research provider."""

    content: str
    citations: list[ResearchCitation] = []
    model: str = ""
    usage: dict[str, int] = {}

    # Intermediate outputs for evaluation
    web_searches: list[WebSearchCall] = Field(
        default_factory=list, description="Web searches performed during research"
    )
    reasoning_summaries: list[ReasoningSummary] = Field(
        default_factory=list, description="Reasoning steps from the model"
    )
    tool_call_count: int = Field(
        default=0, description="Total number of tool calls made"
    )
    duration_seconds: float = Field(
        default=0.0, description="Time taken for the research call"
    )

    raw_response: Any = None


class ResearchProvider(ABC):
    """Abstract base class for deep research providers."""

    @abstractmethod
    async def research(
        self,
        query: str,
        *,
        system_instructions: str | None = None,
        **kwargs: Any,
    ) -> ResearchResponse:
        """
        Execute a deep research task.

        Args:
            query: The research question/query.
            system_instructions: Optional instructions for the research agent.
            **kwargs: Provider-specific options.

        Returns:
            ResearchResponse with the research report and citations.
        """
        pass

    async def research_stream(
        self,
        query: str,
        *,
        system_instructions: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream research progress (optional, not all providers support this).

        Default implementation just yields the final result.
        """
        response = await self.research(
            query, system_instructions=system_instructions, **kwargs
        )
        yield response.content

