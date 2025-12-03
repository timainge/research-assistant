"""Research provider abstraction layer."""

from .base import ResearchProvider, ResearchResponse
from .openai_deep_research import OpenAIDeepResearchProvider

__all__ = ["ResearchProvider", "ResearchResponse", "OpenAIDeepResearchProvider"]

