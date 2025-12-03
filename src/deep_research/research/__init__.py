"""Research provider abstraction layer."""

from .base import (
    ReasoningSummary,
    ResearchCitation,
    ResearchProvider,
    ResearchResponse,
    WebSearchCall,
)
from .openai_deep_research import OpenAIDeepResearchProvider

__all__ = [
    "ReasoningSummary",
    "ResearchCitation",
    "ResearchProvider",
    "ResearchResponse",
    "WebSearchCall",
    "OpenAIDeepResearchProvider",
]

