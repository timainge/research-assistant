"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = {}
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
        output_schema: dict | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt/input.
            system: Optional system/developer message.
            model: Override the default model.
            reasoning_effort: For reasoning models (none/low/medium/high).
            verbosity: Output verbosity (low/medium/high).
            output_schema: JSON schema for structured output.
            **kwargs: Provider-specific options.

        Returns:
            LLMResponse with the generated content.
        """
        pass

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Generate a structured completion that conforms to a Pydantic model.

        Args:
            prompt: The user prompt/input.
            response_model: Pydantic model class for the response.
            system: Optional system/developer message.
            model: Override the default model.
            **kwargs: Provider-specific options.

        Returns:
            Instance of response_model populated with the response.
        """
        pass

