# Provider Abstraction Pattern

Use abstract base classes for swappable providers (LLM, research, storage, etc.).

## Structure

```
src/package/
├── llm/
│   ├── __init__.py      # Exports base + implementations
│   ├── base.py          # Abstract base class
│   ├── openai_provider.py
│   └── anthropic_provider.py
└── research/
    ├── __init__.py
    ├── base.py
    ├── openai_deep_research.py
    └── perplexity_provider.py
```

## Base Class Pattern

```python
# base.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ProviderResponse(BaseModel):
    """Standardized response model."""
    content: str
    usage: dict = {}
    raw_response: Any = None

class Provider(ABC):
    """Abstract base for providers."""
    
    @abstractmethod
    async def execute(self, query: str, **kwargs) -> ProviderResponse:
        """Execute the provider's main function."""
        pass
```

## Key Principles

1. **Async by default** - Use `async/await` for all provider methods
2. **Pydantic responses** - Standardized response models across providers
3. **Raw response access** - Include `raw_response` for debugging/advanced use
4. **Separate modules** - One file per provider implementation
5. **Factory function** - Create providers via config, not direct instantiation

## Factory Pattern

```python
# __init__.py
def get_provider(provider_type: str, **kwargs) -> Provider:
    """Factory to create provider instances."""
    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
```

## Config-Driven Selection

```python
# config.py
class Settings(BaseSettings):
    llm_provider: str = "openai"
    research_provider: str = "openai_deep_research"

# orchestrator.py
def _create_llm_provider(self) -> LLMProvider:
    if self.settings.llm_provider == "openai":
        return OpenAIProvider(api_key=self.settings.openai_api_key)
    # Add more providers here
```

## Benefits

- Easy to add new providers without changing orchestration code
- Testable with mock providers
- Clear contracts via abstract methods
- Consistent response handling across providers

