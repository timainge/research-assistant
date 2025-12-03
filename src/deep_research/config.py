"""Configuration management using Pydantic settings."""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # LLM settings
    llm_provider: Literal["openai"] = Field(
        default="openai", description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-5.1", description="Model name for LLM calls"
    )
    llm_reasoning_effort: Literal["none", "low", "medium", "high"] = Field(
        default="medium", description="Reasoning effort for GPT-5.1"
    )
    llm_verbosity: Literal["low", "medium", "high"] = Field(
        default="medium", description="Output verbosity for GPT-5.1"
    )

    # Research provider settings
    research_provider: Literal["openai_deep_research"] = Field(
        default="openai_deep_research", description="Deep research provider"
    )
    research_model: str = Field(
        default="o4-mini-deep-research-2025-06-26",
        description="Model for deep research tasks",
    )

    # Orchestrator limits
    max_parallel_tasks: int = Field(
        default=3, description="Max parallel deep research tasks"
    )
    max_iterations: int = Field(
        default=5, description="Max planning/execution iterations"
    )
    max_runtime_seconds: int = Field(
        default=900, description="Max total runtime in seconds"
    )

    # Paths
    prompts_dir: str = Field(
        default="prompts", description="Directory containing prompt templates"
    )


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

