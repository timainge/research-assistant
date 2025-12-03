"""Base classes for research providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel


class ResearchCitation(BaseModel):
    """A citation from research results."""

    url: str
    title: str = ""
    snippet: str = ""


class ResearchResponse(BaseModel):
    """Standardized response from a research provider."""

    content: str
    citations: list[ResearchCitation] = []
    model: str = ""
    usage: dict[str, int] = {}
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

