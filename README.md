# Deep Research Orchestrator

AI-powered research automation that decomposes complex questions into parallel research tasks and synthesizes comprehensive answers.

## Features

- **Smart Planning**: Analyzes research questions and decides whether to clarify or decompose into tasks
- **Parallel Execution**: Runs independent research tasks concurrently
- **Multi-Provider**: Abstraction layer supports swapping LLM and research providers
- **Citation Tracking**: Preserves source references throughout the research process
- **Synthesis**: Combines findings from multiple tasks into coherent answers

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Set your API key
export OPENAI_API_KEY="your-key-here"

# Run a research task
deep-research "What are the key differences between React, Vue, and Svelte for building modern web applications?"
```

## Usage

### CLI

```bash
# Basic research
deep-research "Your research question here"

# With additional context
deep-research "Your question" --context "Focus on enterprise use cases"

# Non-interactive mode (no clarification prompts)
deep-research "Your question" --non-interactive

# List available prompt templates
deep-research list-prompts
```

### Python API

```python
import asyncio
from deep_research.orchestrator import ResearchOrchestrator
from deep_research.config import get_settings

async def main():
    settings = get_settings()
    orchestrator = ResearchOrchestrator(settings=settings)
    
    result = await orchestrator.run(
        "What are the emerging trends in AI research for 2025?"
    )
    
    print(result["answer"])

asyncio.run(main())
```

## Configuration

Set these environment variables or create a `.env` file:

```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional (defaults shown)
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.1
LLM_REASONING_EFFORT=medium
LLM_VERBOSITY=medium
RESEARCH_PROVIDER=openai_deep_research
RESEARCH_MODEL=o4-mini-deep-research-2025-06-26
MAX_PARALLEL_TASKS=3
MAX_ITERATIONS=5
MAX_RUNTIME_SECONDS=900
```

## Architecture

```
src/deep_research/
├── __init__.py
├── config.py          # Pydantic settings
├── models.py          # Data models (tasks, results, etc.)
├── prompts.py         # Prompt template loader
├── orchestrator.py    # Main orchestration logic
├── cli.py             # CLI interface
├── llm/               # LLM provider abstraction
│   ├── base.py        # Abstract base class
│   └── openai_provider.py
└── research/          # Research provider abstraction
    ├── base.py        # Abstract base class
    └── openai_deep_research.py

prompts/               # Prompt templates (Markdown + YAML frontmatter)
├── planner.md
├── research_instructions.md
└── synthesize.md
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run tests
pytest
```

## Notes

- See `notes/design.md` for architecture decisions and detailed design
- See `notes/todo.md` for backlog and next actions
