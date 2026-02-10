# Deep Research Orchestrator – Design Document

## Current State

A working MVP that orchestrates OpenAI Deep Research API calls to answer complex research questions through task decomposition and parallel execution.

---

## 1. Goals

Build a **deep research orchestrator** that:

* Takes a **high-level research question**
* **Plans** and decomposes into focused sub-questions
* Executes research tasks **in parallel** where possible
* Uses **OpenAI Deep Research** for web-heavy research
* **Synthesizes** findings into a coherent answer with citations
* Is implemented as **clear, testable Python** with swappable providers

---

## 2. Architecture

### 2.1 Package Structure (Implemented)

```
src/deep_research/
├── __init__.py
├── __main__.py        # python -m deep_research
├── config.py          # Pydantic settings from env
├── models.py          # ResearchTask, ResearchResult, etc.
├── prompts.py         # YAML frontmatter + Mustache loader
├── orchestrator.py    # ResearchOrchestrator class
├── cli.py             # Typer CLI with Rich output
├── llm/
│   ├── base.py        # LLMProvider ABC
│   └── openai_provider.py  # GPT-5.1 Responses API
└── research/
    ├── base.py        # ResearchProvider ABC
    └── openai_deep_research.py  # Deep Research API
```

### 2.2 Key Design Decisions

1. **No LangChain** – We built a simpler, custom implementation without LangChain abstractions. Direct OpenAI SDK calls give us full control over the Responses API.

2. **Provider Abstraction** – Both LLM and Research providers use abstract base classes, making it easy to swap OpenAI for Perplexity, local models, etc.

3. **Prompt Templates** – Prompts are Markdown files with YAML frontmatter (model settings, output schemas) and Mustache template syntax. Keeps prompts separate from code.

4. **Structured Outputs** – GPT-5.1's JSON schema mode ensures planning output conforms to expected structure. Required `additionalProperties: false` on all objects.

5. **Parallel Execution** – Tasks without dependencies run in parallel via `asyncio.gather`. Respects `max_parallel_tasks` setting.

6. **Research Metrics** – Capture web searches, reasoning summaries, tool call counts, and timing for evaluation purposes.

---

## 3. Research Flow

```
User Question
     │
     ▼
┌─────────────────┐
│  Plan (GPT-5.1) │ → Clarify OR Decompose into tasks
└─────────────────┘
     │
     ▼ (if tasks)
┌─────────────────────────────────────┐
│  Execute Tasks (parallel batches)   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐│
│  │ Task 1  │ │ Task 2  │ │ Task 3  ││
│  │ (DR API)│ │ (DR API)│ │ (DR API)││
│  └─────────┘ └─────────┘ └─────────┘│
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────┐
│   Synthesize    │ → Combine results into final answer
└─────────────────┘
     │
     ▼
  Final Answer + Citations
```

### 3.1 Planning Phase

The planner (GPT-5.1 with structured output) decides:
- **Clarify**: Ask user 2-3 questions if scope unclear
- **Decompose**: Create focused research tasks

Each task has:
- `id`: Short slug (e.g., "market-size")
- `query`: The specific research question
- `priority`: Higher = run first
- `depends_on`: Task IDs that must complete first

### 3.2 Execution Phase

Tasks are executed in batches:
1. Find all tasks with satisfied dependencies
2. Take up to `max_parallel_tasks` highest priority
3. Execute in parallel via Deep Research API
4. Repeat until all tasks complete

### 3.3 Synthesis Phase

For single-task results, return directly. For multiple tasks:

1. **Batch synthesis** (if total content < 250k tokens): Synthesize all at once
2. **Rolling synthesis** (if content exceeds limits):
   - Start with first result as draft
   - Merge each subsequent result into the draft
   - Compress draft if needed before merging
   - Maintains citations and resolves contradictions incrementally

This prevents context overflow errors when Deep Research returns large reports.

---

## 4. Configuration

```python
# config.py - Pydantic Settings
class Settings(BaseSettings):
    openai_api_key: str
    
    llm_provider: str = "openai"
    llm_model: str = "gpt-5.1"
    llm_reasoning_effort: str = "medium"
    
    research_provider: str = "openai_deep_research"
    research_model: str = "o4-mini-deep-research-2025-06-26"
    
    max_parallel_tasks: int = 3
    max_iterations: int = 5
    max_runtime_seconds: int = 900
```

---

## 5. Prompt Templates

Templates use YAML frontmatter + Mustache syntax:

```markdown
---
name: planner
model: gpt-5.1
reasoning_effort: medium
output_schema:
  name: PlanningOutput
  strict: true
  schema: { ... }
---

You are a research planning assistant...

## User's Research Question

{{question}}

{{#context}}
## Additional Context
{{context}}
{{/context}}
```

**Note**: Use Mustache syntax `{{#var}}...{{/var}}` for conditionals, not `{{#if var}}`.

---

## 6. API Timeouts

Deep Research API calls can take 15-30+ minutes. We use explicit httpx timeouts:

```python
DEEP_RESEARCH_TIMEOUT = httpx.Timeout(
    connect=30.0,    # 30s to connect
    read=1800.0,     # 30 min to wait for response
    write=60.0,      # 60s to send
    pool=30.0,       # 30s for pool
)
```

### 6.1 Runtime and Reliability Guards

To prevent runaway loops and unbounded runtime/cost, the orchestrator now enforces:

- **Workflow runtime budget**: A hard deadline based on `max_runtime_seconds` is checked before scheduling, task preparation, instruction generation, deep-research calls, and synthesis operations.
- **Clarification loop cap**: Interactive clarification can re-enter planning, but is now capped by `max_iterations`.
- **Compression pass cap**: Oversized synthesis inputs use iterative compression with a bounded number of passes (`max_iterations * 2`) and a truncation fallback if still oversized.
- **Partial-failure handling**: Failed tasks are excluded from synthesis and surfaced in the final answer as a partial-failure notice.

---

## 7. Research Metrics

Each `ResearchResult` captures:

- `web_searches`: List of search queries performed
- `reasoning_summaries`: Model's reasoning steps
- `tool_call_count`: Total web searches made
- `duration_seconds`: API call time
- `usage`: Token counts (input/output)

Useful for evaluation, debugging, and cost tracking.

---

## 8. Phase 2 (Future)

Not yet implemented:

- **Judge/Eval Loop**: Evaluate synthesis quality, trigger re-planning if gaps found
- **Hierarchical Summarization**: Compress results to fit in context
- **Alternative Providers**: Perplexity, local LLMs (Ollama)
- **Internal KB RAG**: Query local markdown via vector store
- **MCP Integration**: Let Deep Research query internal tools
- **Persistence**: Save/resume research sessions
- **Streaming**: Stream partial results during long waits
- **Cost Tracking**: Budget limits and cost estimation

---

## 9. Known Limitations

1. **No Re-planning**: Single planning pass only (no judge loop)
2. **OpenAI Only**: No alternative providers implemented yet
3. **No Caching**: Repeated queries re-run full research
4. **Token Estimation**: Uses rough 4 chars/token heuristic (could use tiktoken for precision)
5. **Non-interruptible Provider Calls**: Once a deep research request is in-flight, it cannot be preempted mid-call
