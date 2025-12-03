"""OpenAI Deep Research provider implementation."""

from typing import Any

from openai import AsyncOpenAI

from .base import ResearchCitation, ResearchProvider, ResearchResponse


class OpenAIDeepResearchProvider(ResearchProvider):
    """OpenAI Deep Research API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "o4-mini-deep-research-2025-06-26",
    ):
        self.client = AsyncOpenAI(api_key=api_key, timeout=600.0)
        self.model = model

    async def research(
        self,
        query: str,
        *,
        system_instructions: str | None = None,
        **kwargs: Any,
    ) -> ResearchResponse:
        """Execute a deep research task using OpenAI's Deep Research API."""
        system_msg = system_instructions or (
            "You are a professional researcher. "
            "Produce a structured, citation-rich report with headings, "
            "bullet points, and clear recommendations."
        )

        input_messages = [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_msg}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            },
        ]

        response = await self.client.responses.create(
            model=self.model,
            input=input_messages,
            reasoning={"summary": "auto"},
            tools=[{"type": "web_search_preview"}],
        )

        # Extract content from response
        content = ""
        citations: list[ResearchCitation] = []

        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        content += block.text
                    # Extract citations from annotations if present
                    if hasattr(block, "annotations"):
                        for ann in block.annotations:
                            if getattr(ann, "type", None) == "url_citation":
                                citations.append(
                                    ResearchCitation(
                                        url=getattr(ann, "url", ""),
                                        title=getattr(ann, "title", ""),
                                        snippet="",
                                    )
                                )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        return ResearchResponse(
            content=content,
            citations=citations,
            model=self.model,
            usage=usage,
            raw_response=response,
        )

