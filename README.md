# Deep Research Orchestrator

AI-powered research automation that decomposes complex questions into parallel research tasks and synthesizes comprehensive answers.

## Features

- **Smart Planning**: Analyzes research questions and decides whether to clarify or decompose into tasks
- **Parallel Execution**: Runs independent research tasks concurrently (respects dependencies)
- **Multi-Provider**: Abstraction layer supports swapping LLM and research providers
- **Citation Tracking**: Preserves source references throughout the research process
- **Research Metrics**: Captures web searches, reasoning steps, and tool call counts for evaluation
- **Verbose Logging**: Progress tracking with `--verbose` flag

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

# With verbose logging (see progress)
deep-research -v "Your research question here"

# With additional context
deep-research "Your question" --context "Focus on enterprise use cases"

# Non-interactive mode (no clarification prompts)
deep-research "Your question" --non-interactive

# Save full results to JSON file (includes tasks, metrics, citations)
deep-research "Your question" --save results.json

# Combine flags (verbose + non-interactive + save)
deep-research -v -n -o output.json "Your question"

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
    orchestrator = ResearchOrchestrator(settings=settings, verbose=True)
    
    result = await orchestrator.run(
        "What are the emerging trends in AI research for 2025?"
    )
    
    print(result["answer"])
    
    # Access research metrics
    for task_id, task_result in result["results"].items():
        print(f"Task {task_id}: {task_result['tool_call_count']} tool calls")

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
├── __main__.py        # python -m deep_research
├── config.py          # Pydantic settings
├── models.py          # Data models (tasks, results, metrics)
├── prompts.py         # Prompt template loader (YAML + Mustache)
├── orchestrator.py    # Main orchestration logic
├── cli.py             # CLI interface (Typer + Rich)
├── llm/               # LLM provider abstraction
│   ├── base.py        # Abstract base class
│   └── openai_provider.py  # GPT-5.1 via Responses API
└── research/          # Research provider abstraction
    ├── base.py        # Abstract base class
    └── openai_deep_research.py  # Deep Research API

prompts/               # Prompt templates (Markdown + YAML frontmatter)
├── planner.md         # Task decomposition
├── research_instructions.md  # Query expansion
└── synthesize.md      # Multi-result synthesis
```

## Research Output

Each research run returns:

```python
{
    "status": "complete",
    "answer": "...",  # Final synthesized answer
    "tasks": [...],   # List of research tasks executed
    "results": {      # Per-task results with metrics
        "task-id": {
            "content": "...",
            "citations": [...],
            "web_searches": [...],      # Queries performed
            "reasoning_summaries": [...], # Model reasoning
            "tool_call_count": 8,
            "duration_seconds": 45.2,
            "usage": {"input_tokens": 1234, "output_tokens": 5678}
        }
    },
    "reasoning": "..."  # Planning rationale
}
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

- Deep Research API calls can take 5-20+ minutes for complex queries
- Use `--verbose` to monitor progress during long runs
- See `notes/design.md` for architecture decisions
- See `notes/todo.md` for backlog and next actions
