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
        completed_results: dict[str, ResearchResult] | None = None,
    ) -> ResearchResult:
        """Execute a single research task."""
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
    ) -> dict[str, ResearchResult]:
        """Execute multiple research tasks, respecting dependencies."""
        logger.info(f"🚀 Executing {len(tasks)} research task(s)...")
        results: dict[str, ResearchResult] = {}
        pending = {t.id: t for t in tasks}
        batch_num = 0

        while pending:
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
                    self.execute_task(t, original_question, results)
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

    async def synthesize(
        self,
        original_question: str,
        results: dict[str, ResearchResult],
    ) -> str:
        """Synthesize multiple research results into a final answer."""
        logger.info(f"📝 Synthesizing {len(results)} research result(s)...")
        start = time.time()

        results_list = [
            {
                "task_id": r.task_id,
                "content": r.content,
                "citations": [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in r.citations
                ],
            }
            for r in results.values()
        ]

        prompt = self.prompts.render(
            "synthesize",
            original_question=original_question,
            results=results_list,
        )

        response = await self.llm.complete(
            prompt,
            reasoning_effort="high",
        )

        logger.info(f"✅ Synthesis completed in {time.time() - start:.1f}s")
        return response.content

    async def run(
        self,
        question: str,
        context: str = "",
        on_clarification: Any = None,
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
        workflow_start = time.time()
        state = OrchestratorState(original_question=question)

        # Step 1: Plan
        plan = await self.plan(question, context)
        state.status = "planned"

        # Step 2: Handle clarification if needed
        if plan.needs_clarification and plan.clarification:
            if on_clarification:
                # Get clarification from user
                answers = await on_clarification(plan.clarification)
                # Re-plan with the additional context
                combined_context = f"{context}\n\nUser clarification:\n{answers}"
                return await self.run(question, combined_context, on_clarification)
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

        results = await self.execute_tasks(plan.tasks, question)
        state.results = results
        state.status = "synthesizing"

        # Step 4: Synthesize (skip if only one task)
        if len(results) == 1:
            final_answer = list(results.values())[0].content
        else:
            final_answer = await self.synthesize(question, results)

        state.status = "complete"
        logger.info(f"🎉 Research workflow completed in {time.time() - workflow_start:.1f}s")

        return {
            "status": "complete",
            "answer": final_answer,
            "tasks": [t.model_dump() for t in plan.tasks],
            "results": {k: v.model_dump() for k, v in results.items()},
            "reasoning": plan.reasoning,
        }
