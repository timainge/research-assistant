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

### Active
- [ ] Synthesize function: add token limit guards to avoid context overflow

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

---

Backlog (general)
-----------------
- Add CI (formatting, linting, tests)
- Document environment management and secrets handling
- Add example research outputs to docs

---

Known Issues
------------
- [ ] Synthesize fn combines all reports without token limit guards (could overflow context)
- [ ] No cost estimation or budget enforcement yet

---

Next Actions
------------
1. Add token limit guards to synthesis
2. Test end-to-end with various question types
3. Consider adding streaming for better UX during long waits
