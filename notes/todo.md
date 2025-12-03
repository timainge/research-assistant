Process Rule (pin)
------------------
Stay focused on the current task. When new tasks or opportunities arise, log them below and defer until the active task is complete.

---

MVP Sprint: Deep Research Orchestrator
======================================

### Active
- [ ] Scaffold `src/deep_research/` package structure
- [ ] Create prompt templates (YAML frontmatter + Handlebars markdown)
- [ ] Implement prompt loader utility
- [ ] Implement LLM abstraction layer (OpenAI GPT-5.1 first, swappable)
- [ ] Implement research provider abstraction (OpenAI Deep Research API first)
- [ ] Implement config (Pydantic settings)
- [ ] Implement models (ResearchTask, ResearchResult, ClarificationRequest)
- [ ] Implement orchestrator (clarify-or-dispatch logic)
- [ ] Implement CLI entry point
- [ ] Update requirements.txt with dependencies

### Deferred (Phase 2)
- [ ] Multi-task roll-up and synthesis
- [ ] Judge / eval chain before finalising
- [ ] Re-plan loop (spawn new branches based on judge feedback)
- [ ] Perplexity provider
- [ ] Local LLM provider (Ollama, llama.cpp)
- [ ] Internal KB RAG tool (Chroma / PGVector)
- [ ] MCP integration for Deep Research to query internal KB
- [ ] Persistence & resume (save/load research tree)
- [ ] Rich logging / LangSmith tracing
- [ ] CI (formatting, linting, tests)
- [ ] Document environment management and secrets handling

---

Backlog (general)
-----------------
- Add CI (formatting, linting, tests)
- Document environment management and secrets handling

---

Next Actions
------------
1. Scaffold package and prompt templates
2. Implement abstractions (LLM, research provider)
3. Wire up orchestrator + CLI
4. Manual end-to-end test with a real question


### Human issues/reminders

- [ ] synthesize fn combines all reports without token limit guards