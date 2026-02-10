Process Rule (pin)
------------------
Stay focused on the current task. When new tasks or opportunities arise, log them below and defer until the active task is complete.

---

MVP Sprint: Deep Research Orchestrator
======================================

### Completed ✅
- [x] Scaffold `src/deep_research/` package structure
- [x] Create prompt templates (YAML frontmatter + Mustache markdown)
- [x] Implement prompt loader utility
- [x] Implement LLM abstraction layer (OpenAI GPT-5.1 first, swappable)
- [x] Implement research provider abstraction (OpenAI Deep Research API first)
- [x] Implement config (Pydantic settings)
- [x] Implement models (ResearchTask, ResearchResult, ClarificationRequest)
- [x] Implement orchestrator (clarify-or-dispatch logic)
- [x] Implement CLI entry point
- [x] Update requirements.txt with dependencies
- [x] Add verbose logging (`--verbose` / `-v` flag)
- [x] Fix Mustache template syntax (use `{{#var}}` not `{{#if var}}`)
- [x] Fix JSON schema for GPT-5.1 structured outputs (additionalProperties: false)
- [x] Fix httpx timeouts for long-running Deep Research API calls (30min read timeout)
- [x] Capture intermediate outputs (web searches, reasoning, tool call counts) for evaluation
- [x] Add `--save` / `-o` option to save full JSON results to file
- [x] Add rolling synthesis for large results (context overflow protection)
- [x] Fix verbose logging to suppress httpx/httpcore debug spam
- [x] Handle OpenAI 10MB per-string limit by chunking oversized content
- [x] Add workflow runtime budget guard (`max_runtime_seconds`) across execution and synthesis
- [x] Add clarification-loop safety guard using `max_iterations`
- [x] Add compression-pass ceiling and truncation fallback (prevent unbounded recursive compression)
- [x] Exclude failed tasks from synthesis and surface partial-failure notice in final output
- [x] Add CLI handling for runtime/guard errors with clear user-facing messages

### Active
- [ ] Test synthesis with very large research outputs
- [ ] Add deterministic tests for guard behavior (runtime, clarification cap, compression cap)

### Deferred (Phase 2)
- [ ] Multi-task roll-up with hierarchical summarization
- [ ] Judge / eval chain before finalizing
- [ ] Re-plan loop (spawn new branches based on judge feedback)
- [ ] Perplexity provider
- [ ] Local LLM provider (Ollama, llama.cpp)
- [ ] Internal KB RAG tool (Chroma / PGVector)
- [ ] MCP integration for Deep Research to query internal KB
- [ ] Persistence & resume (save/load research state)
- [ ] Streaming output during long research tasks
- [ ] CI (formatting, linting, tests)
- [ ] Cost tracking and budget limits
- [ ] graceful handling of connection errors/offline-mode and option to use local-llm and local-kb
- [ ] refactor context management and other non-orchestration logic into a utils module
---

Backlog (general)
-----------------
- Add CI (formatting, linting, tests)
- Document environment management and secrets handling
- Add example research outputs to docs

---

Known Issues
------------
- [ ] No cost estimation or budget enforcement yet
- [ ] Token estimation uses rough heuristic (4 chars/token) - could use tiktoken for precision
- [ ] Deep research calls cannot be interrupted mid-request once an external API call has started

---

Next Actions
------------
1. Add unit tests with mocked providers for runtime and guard behavior
2. Test end-to-end with various question types, including induced task failures
3. Add final output section for structured errors/omissions in saved JSON
