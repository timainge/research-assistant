"""OpenAI Deep Research provider implementation."""

import logging
import time
from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import ResearchCitation, ResearchProvider, ResearchResponse

logger = logging.getLogger("deep_research.research.openai")

# Deep Research can take 15-30+ minutes for complex queries
# Set generous timeouts for connect, read, write, and pool
DEEP_RESEARCH_TIMEOUT = httpx.Timeout(
    connect=30.0,      # 30s to establish connection
    read=1800.0,       # 30 minutes to wait for response
    write=60.0,        # 60s to send request
    pool=30.0,         # 30s to acquire connection from pool
)


class OpenAIDeepResearchProvider(ResearchProvider):
    """OpenAI Deep Research API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "o4-mini-deep-research-2025-06-26",
    ):
        self.client = AsyncOpenAI(api_key=api_key, timeout=DEEP_RESEARCH_TIMEOUT)
        self.model = model

    async def research(
        self,
        query: str,
        *,
        system_instructions: str | None = None,
        **kwargs: Any,
    ) -> ResearchResponse:
        """Execute a deep research task using OpenAI's Deep Research API."""
        logger.debug(f"Starting OpenAI Deep Research with model {self.model}")
        logger.debug(f"Query: {query[:100]}...")
        start = time.time()

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

        logger.debug("Calling OpenAI Deep Research API...")
        response = await self.client.responses.create(
            model=self.model,
            input=input_messages,
            reasoning={"summary": "auto"},
            tools=[{"type": "web_search_preview"}],
        )
        logger.debug(f"OpenAI Deep Research API call completed in {time.time() - start:.1f}s")

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

