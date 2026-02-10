"""Research orchestrator - coordinates planning, execution, and synthesis."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .llm import LLMProvider, OpenAIProvider
from .models import (
    ClarificationRequest,
    OrchestratorState,
    PlanningOutput,
    ResearchResult,
    ResearchTask,
    TaskStatus,
)
from .prompts import PromptLoader, get_prompt_loader
from .research import OpenAIDeepResearchProvider, ResearchProvider

logger = logging.getLogger("deep_research.orchestrator")


class ResearchOrchestrator:
    """Orchestrates deep research workflows."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
        research_provider: ResearchProvider | None = None,
        prompt_loader: PromptLoader | None = None,
        verbose: bool = False,
    ):
        self.settings = settings or get_settings()
        self.verbose = verbose
        self.llm = llm_provider or self._create_llm_provider()
        self.research = research_provider or self._create_research_provider()
        self.prompts = prompt_loader or get_prompt_loader(
            Path(__file__).parent.parent.parent / "prompts"
        )
        self._last_planner_prompt: str = ""

    def _check_runtime(self, deadline: float, stage: str) -> None:
        """Raise if the workflow runtime budget has been exceeded."""
        if time.time() > deadline:
            raise TimeoutError(
                f"Runtime budget exceeded during {stage}. "
                f"Configured max_runtime_seconds={self.settings.max_runtime_seconds}."
            )

    def _create_llm_provider(self) -> LLMProvider:
        """Create the configured LLM provider."""
        if self.settings.llm_provider == "openai":
            return OpenAIProvider(
                api_key=self.settings.openai_api_key,
                default_model=self.settings.llm_model,
                default_reasoning_effort=self.settings.llm_reasoning_effort,
                default_verbosity=self.settings.llm_verbosity,
            )
        raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    def _create_research_provider(self) -> ResearchProvider:
        """Create the configured research provider."""
        if self.settings.research_provider == "openai_deep_research":
            return OpenAIDeepResearchProvider(
                api_key=self.settings.openai_api_key,
                model=self.settings.research_model,
            )
        raise ValueError(
            f"Unknown research provider: {self.settings.research_provider}"
        )

    async def plan(
        self,
        question: str,
        context: str = "",
    ) -> PlanningOutput:
        """
        Analyze a research question and create a plan.

        Returns either a clarification request or a list of research tasks.
        """
        logger.info("📋 Planning research approach...")
        start = time.time()

        prompt = self.prompts.render(
            "planner",
            question=question,
            context=context,
        )
        self._last_planner_prompt = prompt

        template = self.prompts.load("planner")
        system = (
            "You are a research planning assistant. "
            "Respond with valid JSON matching the schema."
        )

        logger.debug("Calling LLM for planning...")
        response = await self.llm.complete(
            prompt,
            system=system,
            reasoning_effort=template.reasoning_effort,
            output_schema=template.output_schema,
        )
        logger.debug(f"Planning LLM call completed in {time.time() - start:.1f}s")

        # Parse response into PlanningOutput
        import json

        data = json.loads(response.content)

        tasks = []
        for t in data.get("tasks", []):
            tasks.append(
                ResearchTask(
                    id=t["id"],
                    query=t["query"],
                    priority=t.get("priority", 0),
                    depends_on=t.get("depends_on", []),
                )
            )

        clarification = None
        clarification_questions = data.get("clarification_questions", [])
        if clarification_questions:
            clarification = ClarificationRequest(
                questions=clarification_questions,
                context=data.get("clarification_context", ""),
            )

        plan = PlanningOutput(
            needs_clarification=data.get("needs_clarification", False),
            clarification=clarification,
            tasks=tasks,
            reasoning=data.get("reasoning", ""),
        )

        if plan.needs_clarification:
            logger.info("❓ Clarification needed from user")
        else:
            logger.info(f"✅ Plan created with {len(tasks)} task(s)")
            for t in tasks:
                logger.debug(f"  - [{t.id}] {t.query[:60]}...")

        return plan

    async def execute_task(
        self,
        task: ResearchTask,
        original_question: str,
        deadline: float,
        completed_results: dict[str, ResearchResult] | None = None,
    ) -> ResearchResult:
        """Execute a single research task."""
        self._check_runtime(deadline, f"task preparation [{task.id}]")
        logger.info(f"🔬 Starting task [{task.id}]: {task.query[:50]}...")
        task_start = time.time()

        # Build context from completed tasks if any
        completed_context = ""
        if completed_results:
            parts = []
            for tid, result in completed_results.items():
                parts.append(f"### {tid}\n{result.content[:2000]}...")
            completed_context = "\n\n".join(parts)

        # Generate detailed research instructions
        logger.debug(f"[{task.id}] Generating research instructions...")
        self._check_runtime(deadline, f"instruction generation [{task.id}]")
        instructions_prompt = self.prompts.render(
            "research_instructions",
            original_question=original_question,
            task_query=task.query,
            completed_tasks=completed_context if completed_context else None,
        )

        instructions_response = await self.llm.complete(
            instructions_prompt,
            reasoning_effort="low",
        )
        logger.debug(f"[{task.id}] Instructions generated")

        # Execute the research
        self._check_runtime(deadline, f"deep research call [{task.id}]")
        logger.info(f"🌐 [{task.id}] Executing deep research (this may take several minutes)...")
        research_start = time.time()
        research_response = await self.research.research(
            instructions_response.content,
        )
        research_duration = time.time() - research_start
        logger.info(f"✅ [{task.id}] Deep research completed in {research_duration:.1f}s")

        result = ResearchResult(
            task_id=task.id,
            content=research_response.content,
            citations=[
                {"url": c.url, "title": c.title, "snippet": c.snippet}
                for c in research_response.citations
            ],
            confidence=0.8,  # TODO: implement confidence scoring
            web_searches=[
                {"query": s.query, "status": s.status}
                for s in research_response.web_searches
            ],
            reasoning_summaries=[r.text for r in research_response.reasoning_summaries],
            tool_call_count=research_response.tool_call_count,
            duration_seconds=research_response.duration_seconds,
            usage=research_response.usage,
            instruction_generation_prompt=instructions_prompt,
            research_prompt=instructions_response.content,
        )

        logger.info(
            f"✅ [{task.id}] Task completed in {time.time() - task_start:.1f}s "
            f"({len(result.content)} chars, {result.tool_call_count} tool calls)"
        )
        return result

    async def execute_tasks(
        self,
        tasks: list[ResearchTask],
        original_question: str,
        deadline: float,
    ) -> dict[str, ResearchResult]:
        """Execute multiple research tasks, respecting dependencies."""
        logger.info(f"🚀 Executing {len(tasks)} research task(s)...")
        results: dict[str, ResearchResult] = {}
        pending = {t.id: t for t in tasks}
        batch_num = 0

        while pending:
            self._check_runtime(deadline, "task scheduling")
            batch_num += 1
            # Find tasks ready to run (all dependencies satisfied)
            ready = [
                t
                for t in pending.values()
                if all(dep in results for dep in t.depends_on)
            ]

            if not ready:
                # No tasks ready but still pending = circular dependency
                raise RuntimeError(
                    f"Circular dependency detected. Pending: {list(pending.keys())}"
                )

            # Sort by priority and take up to max_parallel
            ready.sort(key=lambda t: -t.priority)
            batch = ready[: self.settings.max_parallel_tasks]

            logger.info(f"📦 Batch {batch_num}: Running {len(batch)} task(s) in parallel: {[t.id for t in batch]}")

            # Mark as running
            for t in batch:
                t.status = TaskStatus.RUNNING

            # Execute in parallel
            batch_results = await asyncio.gather(
                *[
                    self.execute_task(t, original_question, deadline, results)
                    for t in batch
                ],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    logger.error(f"❌ [{task.id}] Failed: {result}")
                    # Create a failed result
                    results[task.id] = ResearchResult(
                        task_id=task.id,
                        content=f"Task failed: {result}",
                        confidence=0.0,
                    )
                else:
                    task.status = TaskStatus.DONE
                    results[task.id] = result

                del pending[task.id]

            logger.info(f"📦 Batch {batch_num} complete. {len(pending)} task(s) remaining.")

        return results

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return len(text) // 4

    async def synthesize(
        self,
        original_question: str,
        results: dict[str, ResearchResult],
        deadline: float,
    ) -> str:
        """Synthesize multiple research results into a final answer.
        
        Uses rolling synthesis if total content exceeds context limits.
        Compresses oversized individual results to stay within API string limits.
        """
        logger.info(f"📝 Synthesizing {len(results)} research result(s)...")
        start = time.time()
        self._check_runtime(deadline, "synthesis")

        # GPT-5.1 has 400k context, but we need room for output (128k max)
        # and system prompt overhead. Use 250k as safe input limit.
        MAX_INPUT_TOKENS = 250_000
        
        # OpenAI has a per-string limit of ~10MB. Use 8MB as safe limit
        # to leave room for prompt template overhead.
        MAX_STRING_CHARS = 8_000_000
        
        results_list = []
        for r in results.values():
            content = r.content
            
            # Check if this individual result is too large
            if len(content) > MAX_STRING_CHARS:
                logger.warning(
                    f"⚠️ Task [{r.task_id}] result is {len(content):,} chars "
                    f"(>{MAX_STRING_CHARS:,}), compressing..."
                )
                content = await self._compress_large_content(
                    original_question,
                    r.task_id,
                    content,
                    MAX_STRING_CHARS // 2,  # Target 4MB after compression
                    deadline=deadline,
                )
            
            results_list.append({
                "task_id": r.task_id,
                "content": content,
                "citations": [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in r.citations
                ],
            })

        # Estimate total tokens
        total_content = "".join(r["content"] for r in results_list)
        estimated_tokens = self._estimate_tokens(total_content)
        
        if estimated_tokens < MAX_INPUT_TOKENS:
            # All results fit - synthesize in one go
            logger.debug(f"All results fit in context (~{estimated_tokens} tokens)")
            return await self._synthesize_batch(original_question, results_list, deadline)
        else:
            # Need rolling synthesis
            logger.info(f"⚠️ Results too large (~{estimated_tokens} tokens), using rolling synthesis")
            return await self._synthesize_rolling(
                original_question,
                results_list,
                MAX_INPUT_TOKENS,
                deadline,
            )

    async def _synthesize_batch(
        self,
        original_question: str,
        results_list: list[dict],
        deadline: float,
    ) -> str:
        """Synthesize all results in a single call."""
        self._check_runtime(deadline, "batch synthesis prompt")
        prompt = self.prompts.render(
            "synthesize",
            original_question=original_question,
            results=results_list,
        )

        response = await self.llm.complete(
            prompt,
            reasoning_effort="high",
        )

        return response.content

    async def _synthesize_rolling(
        self,
        original_question: str,
        results_list: list[dict],
        max_tokens: int,
        deadline: float,
    ) -> str:
        """Incrementally synthesize results to stay within context limits.
        
        Strategy:
        1. Start with first result as the draft
        2. For each subsequent result, merge it into the draft
        3. Keep the running draft under control by summarizing as we go
        """
        start = time.time()
        
        # Target draft size - leave room for new content + overhead
        TARGET_DRAFT_TOKENS = max_tokens // 3  # ~80k tokens for draft
        
        # Initialize with first result
        current_draft = results_list[0]["content"]
        merged_task_ids = [results_list[0]["task_id"]]
        all_citations = results_list[0].get("citations", [])
        
        logger.debug(f"Starting rolling synthesis with task [{results_list[0]['task_id']}]")
        
        # Merge in remaining results one at a time
        for result in results_list[1:]:
            self._check_runtime(deadline, "rolling synthesis merge")
            task_id = result["task_id"]
            new_content = result["content"]
            new_citations = result.get("citations", [])
            
            logger.info(f"🔄 Merging task [{task_id}] into synthesis...")
            
            # Check if we need to compress the draft first
            draft_tokens = self._estimate_tokens(current_draft)
            new_tokens = self._estimate_tokens(new_content)
            
            if draft_tokens + new_tokens > max_tokens:
                # Compress draft before merging
                logger.debug(f"Compressing draft ({draft_tokens} tokens) before merge")
                current_draft = await self._compress_draft(
                    original_question, 
                    current_draft, 
                    merged_task_ids,
                    TARGET_DRAFT_TOKENS,
                    deadline,
                )
            
            # Merge new result into draft
            current_draft = await self._merge_result(
                original_question,
                current_draft,
                merged_task_ids,
                result,
                deadline,
            )
            
            merged_task_ids.append(task_id)
            all_citations.extend(new_citations)
        
        logger.info(f"✅ Rolling synthesis completed in {time.time() - start:.1f}s")
        return current_draft

    async def _compress_large_content(
        self,
        original_question: str,
        task_id: str,
        content: str,
        target_chars: int,
        deadline: float,
    ) -> str:
        """Compress oversized content by chunking and summarizing.
        
        When content exceeds string limits, we split it into chunks,
        summarize each chunk, then combine the summaries.
        """
        # Max chars we can fit in a single API call (leave room for prompt)
        CHUNK_SIZE = 6_000_000  # 6MB per chunk, leaves room for prompt overhead
        max_passes = max(self.settings.max_iterations, 1) * 2
        current_content = content

        for attempt in range(1, max_passes + 1):
            self._check_runtime(deadline, f"compression pass {attempt} [{task_id}]")
            if len(current_content) <= target_chars:
                return current_content

            if len(current_content) <= CHUNK_SIZE:
                # Can compress in one call
                prompt = f"""Summarize this research content concisely while preserving key findings and citations.

## Original Question
{original_question}

## Task: {task_id}

{current_content}

## Your Task
Create a comprehensive but condensed summary (target: ~{target_chars:,} characters) that:
1. Preserves all key findings, data points, and insights
2. Maintains important citations and sources
3. Keeps the logical structure
4. Removes redundancy and verbose explanations

Output ONLY the summary, no preamble."""

                response = await self.llm.complete(prompt, reasoning_effort="medium")
                logger.debug(
                    f"Compressed [{task_id}] from {len(current_content):,} "
                    f"to {len(response.content):,} chars"
                )
                current_content = response.content
                continue

            # Need to chunk the content
            logger.info(
                f"📄 Chunking [{task_id}] ({len(current_content):,} chars) into segments..."
            )
            chunks = []
            for i in range(0, len(current_content), CHUNK_SIZE):
                chunks.append(current_content[i : i + CHUNK_SIZE])

            logger.debug(f"Split into {len(chunks)} chunks")

            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                self._check_runtime(
                    deadline, f"chunk summarization {i + 1}/{len(chunks)} [{task_id}]"
                )
                logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}...")
                prompt = f"""Summarize this segment of research content.

## Context
This is part {i+1} of {len(chunks)} from a research task about: {original_question}

## Content
{chunk}

## Your Task
Create a concise summary preserving key findings, data, and citations.
Output ONLY the summary."""

                response = await self.llm.complete(prompt, reasoning_effort="low")
                chunk_summaries.append(response.content)

            current_content = "\n\n---\n\n".join(chunk_summaries)

        logger.warning(
            f"⚠️ Compression passes exhausted for [{task_id}] after {max_passes} attempts. "
            f"Returning truncated content."
        )
        if len(current_content) > target_chars:
            return current_content[:target_chars]
        return current_content

    async def _compress_draft(
        self,
        original_question: str,
        draft: str,
        task_ids: list[str],
        target_tokens: int,
        deadline: float,
    ) -> str:
        """Compress a draft to fit within target token count."""
        target_chars = target_tokens * 4
        self._check_runtime(deadline, "draft compression")
        
        # Check if draft itself exceeds string limit
        MAX_SAFE_CHARS = 6_000_000
        if len(draft) > MAX_SAFE_CHARS:
            logger.debug(f"Draft too large ({len(draft):,} chars), pre-compressing...")
            draft = await self._compress_large_content(
                original_question,
                f"draft-{'-'.join(task_ids[:2])}",
                draft,
                MAX_SAFE_CHARS // 2,
                deadline,
            )
        
        prompt = f"""You are helping synthesize a multi-part research report.

The current draft (synthesizing tasks: {', '.join(task_ids)}) is too long and needs to be condensed.

## Original Question
{original_question}

## Current Draft
{draft}

## Your Task
Condense this draft to approximately {target_chars:,} characters while:
1. Preserving all key findings and insights
2. Maintaining the most important citations
3. Keeping the logical structure
4. Removing redundancy and verbose explanations

Output ONLY the condensed draft, no preamble."""

        response = await self.llm.complete(
            prompt,
            reasoning_effort="medium",
        )
        
        logger.debug(f"Compressed draft from {self._estimate_tokens(draft)} to {self._estimate_tokens(response.content)} tokens")
        return response.content

    async def _merge_result(
        self,
        original_question: str,
        current_draft: str,
        existing_task_ids: list[str],
        new_result: dict,
        deadline: float,
    ) -> str:
        """Merge a new research result into the current draft."""
        self._check_runtime(deadline, f"merge task [{new_result['task_id']}]")
        MAX_SAFE_CHARS = 6_000_000  # Per-component limit
        
        draft = current_draft
        new_content = new_result['content']
        task_id = new_result['task_id']
        
        # Ensure neither component exceeds safe string size
        if len(draft) > MAX_SAFE_CHARS:
            logger.debug(f"Draft too large for merge ({len(draft):,} chars), compressing...")
            draft = await self._compress_large_content(
                original_question,
                f"draft-merge",
                draft,
                MAX_SAFE_CHARS // 2,
                deadline,
            )
        
        if len(new_content) > MAX_SAFE_CHARS:
            logger.debug(f"New content too large for merge ({len(new_content):,} chars), compressing...")
            new_content = await self._compress_large_content(
                original_question,
                task_id,
                new_content,
                MAX_SAFE_CHARS // 2,
                deadline,
            )
        
        prompt = f"""You are synthesizing research findings into a comprehensive report.

## Original Question
{original_question}

## Current Draft
(Synthesized from tasks: {', '.join(existing_task_ids)})

{draft}

## New Research to Integrate
Task: {task_id}

{new_content}

## Your Task
Integrate the new research into the existing draft to create an updated, comprehensive synthesis.

Guidelines:
1. Preserve key insights from both the draft and new research
2. Resolve any contradictions by noting both perspectives
3. Maintain citations and source references
4. Use clear structure with headings and bullet points
5. Eliminate redundancy between old and new content
6. The result should read as a cohesive document, not a collection of summaries

Output ONLY the updated synthesis, no preamble."""

        response = await self.llm.complete(
            prompt,
            reasoning_effort="high",
        )
        
        return response.content

    async def run(
        self,
        question: str,
        context: str = "",
        on_clarification: Any = None,
        clarification_attempt: int = 0,
        workflow_start: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the full research workflow.

        Args:
            question: The research question.
            context: Additional context or previous clarification answers.
            on_clarification: Async callback for handling clarification requests.
                              Should return the user's answers as a string.

        Returns:
            Dict with 'answer', 'results', 'tasks', and metadata.
        """
        logger.info("🔍 Starting research workflow...")
        if workflow_start is None:
            workflow_start = time.time()
        deadline = workflow_start + self.settings.max_runtime_seconds
        self._check_runtime(deadline, "workflow start")
        state = OrchestratorState(original_question=question)

        # Step 1: Plan
        plan = await self.plan(question, context)
        state.status = "planned"

        # Step 2: Handle clarification if needed
        if plan.needs_clarification and plan.clarification:
            if on_clarification:
                if clarification_attempt >= self.settings.max_iterations:
                    raise RuntimeError(
                        "Clarification loop exceeded max_iterations "
                        f"({self.settings.max_iterations})."
                    )
                # Get clarification from user
                answers = await on_clarification(plan.clarification)
                # Re-plan with the additional context
                combined_context = f"{context}\n\nUser clarification:\n{answers}"
                return await self.run(
                    question,
                    combined_context,
                    on_clarification,
                    clarification_attempt=clarification_attempt + 1,
                    workflow_start=workflow_start,
                )
            else:
                # Return the clarification request
                return {
                    "status": "needs_clarification",
                    "clarification": plan.clarification.model_dump(),
                    "reasoning": plan.reasoning,
                }

        # Step 3: Execute tasks
        state.tasks = {t.id: t for t in plan.tasks}
        state.status = "executing"

        results = await self.execute_tasks(plan.tasks, question, deadline)
        state.results = results
        state.status = "synthesizing"

        failed_task_ids = [t.id for t in plan.tasks if t.status == TaskStatus.FAILED]
        successful_results = {
            task_id: result
            for task_id, result in results.items()
            if task_id not in failed_task_ids
        }

        # Step 4: Synthesize (skip if only one task)
        if not successful_results:
            final_answer = (
                "Research could not complete because all tasks failed.\n\n"
                f"Failed tasks: {', '.join(failed_task_ids)}"
            )
        elif len(successful_results) == 1:
            final_answer = list(successful_results.values())[0].content
        else:
            final_answer = await self.synthesize(question, successful_results, deadline)

        if failed_task_ids:
            final_answer = (
                f"{final_answer}\n\n---\n\n"
                f"## Partial failure notice\n\n"
                f"The following task(s) failed and were excluded from synthesis: "
                f"{', '.join(failed_task_ids)}."
            )

        state.status = "complete"
        elapsed = time.time() - workflow_start
        logger.info(f"🎉 Research workflow completed in {elapsed:.1f}s")

        return {
            "status": "complete",
            "answer": final_answer,
            "tasks": [t.model_dump() for t in plan.tasks],
            "results": {k: v.model_dump() for k, v in results.items()},
            "failed_tasks": failed_task_ids,
            "reasoning": plan.reasoning,
            "planning_prompt": self._last_planner_prompt,
            "elapsed_seconds": elapsed,
        }
