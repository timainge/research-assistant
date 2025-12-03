"""OpenAI Deep Research provider implementation."""

import logging
import time
from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import (
    ReasoningSummary,
    ResearchCitation,
    ResearchProvider,
    ResearchResponse,
    WebSearchCall,
)

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
        duration = time.time() - start
        logger.debug(f"OpenAI Deep Research API call completed in {duration:.1f}s")

        # Extract content, citations, and intermediate outputs from response
        content = ""
        citations: list[ResearchCitation] = []
        web_searches: list[WebSearchCall] = []
        reasoning_summaries: list[ReasoningSummary] = []
        tool_call_count = 0

        for item in response.output:
            item_type = getattr(item, "type", None)

            # Extract web search calls
            if item_type == "web_search_call":
                tool_call_count += 1
                search_query = getattr(item, "query", "") or ""
                status = getattr(item, "status", "completed")
                web_searches.append(WebSearchCall(query=search_query, status=status))
                logger.debug(f"  Web search: {search_query[:50]}...")

            # Extract reasoning summaries
            elif item_type == "reasoning":
                summary_items = getattr(item, "summary", []) or []
                for summary in summary_items:
                    if hasattr(summary, "text"):
                        reasoning_summaries.append(ReasoningSummary(text=summary.text))

            # Extract message content and citations
            elif item_type == "message":
                if hasattr(item, "content") and item.content:
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

            # Fallback for other item types that might have content
            elif hasattr(item, "content") and item.content:
                for block in item.content:
                    if hasattr(block, "text"):
                        content += block.text

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }

        logger.info(
            f"Research complete: {len(web_searches)} searches, "
            f"{len(citations)} citations, {len(content)} chars"
        )

        return ResearchResponse(
            content=content,
            citations=citations,
            model=self.model,
            usage=usage,
            web_searches=web_searches,
            reasoning_summaries=reasoning_summaries,
            tool_call_count=tool_call_count,
            duration_seconds=duration,
            raw_response=response,
        )

