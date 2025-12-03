"""OpenAI LLM provider implementation using GPT-5.1."""

import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI provider using the Responses API for GPT-5.1."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-5.1",
        default_reasoning_effort: str = "medium",
        default_verbosity: str = "medium",
    ):
        self.client = AsyncOpenAI(api_key=api_key, timeout=300.0)
        self.default_model = default_model
        self.default_reasoning_effort = default_reasoning_effort
        self.default_verbosity = default_verbosity

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
        """Generate a completion using GPT-5.1 Responses API."""
        model = model or self.default_model
        reasoning = reasoning_effort or self.default_reasoning_effort
        verb = verbosity or self.default_verbosity

        # Build input messages
        input_messages = []
        if system:
            input_messages.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system}],
                }
            )
        input_messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )

        # Build request params
        request_params: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "reasoning": {"effort": reasoning},
            "text": {"verbosity": verb},
        }

        # Add structured output if schema provided
        if output_schema:
            request_params["text"]["format"] = {
                "type": "json_schema",
                "json_schema": output_schema,
            }

        response = await self.client.responses.create(**request_params)

        # Extract text from response
        content = ""
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        content += block.text

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            raw_response=response,
        )

    async def complete_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured completion conforming to a Pydantic model."""
        # Build JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        output_schema = {
            "name": response_model.__name__,
            "strict": True,
            "schema": schema,
        }

        response = await self.complete(
            prompt,
            system=system,
            model=model,
            output_schema=output_schema,
            **kwargs,
        )

        # Parse JSON response into model
        data = json.loads(response.content)
        return response_model.model_validate(data)

