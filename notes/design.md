Here you go — a single, self-contained “design doc” you can paste straight into your coding assistant as context.

---

# Deep Research Orchestrator – Design & Integration with OpenAI Deep Research + LangChain

## 0. Assumptions

* **LLM provider:** OpenAI (standard chat models + Deep Research models).
* **Framework:** LangChain (Python).
* **Research engine:** OpenAI Deep Research API used as a *heavy* research tool.
* **Internal KB:** Markdown + hyperlinks, exposed via a LangChain RAG tool (vector store).
* **Usage mode:** Single-user CLI prototype.
* **Autonomy:** Agent runs long-form research until:

  * High-confidence answer,
  * Or clearly stuck/low-confidence,
  * Or guardrails hit (max iterations / tokens / cost / time).

---

## 1. Goals

Build a **deep research orchestrator** that:

* Takes a **high-level research question**.
* **Plans** and executes multiple rounds of research.
* Can **branch into parallel sub-questions** when possible.
* Uses:

  * **OpenAI Deep Research** for large, external/web-heavy sub-questions.
  * **LangChain RAG** for internal markdown KB queries.
* **Synthesizes** findings into a coherent, structured final answer with citations.
* Is implemented as **clear, testable Python** that *uses* LangChain, but does not get trapped in its abstractions.

---

## 2. High-Level Architecture

### 2.1 Major components

**Top-level package structure:**

```text
deep_research/
  __init__.py
  config.py        # Pydantic settings
  models.py        # Research tree / node / plan / judge outputs
  tools.py         # LangChain tools: deep research, web search, KB RAG, code eval
  memory.py        # Vector stores, caches, global summary
  chains.py        # LangChain LLM chains for planning / branches / synthesis / judge
  orchestrator.py  # ResearchOrchestrator class (control loop)
  cli.py           # CLI entry point
```

**Core idea:**

* The **orchestrator** is a **plain Python class** (`ResearchOrchestrator`) that:

  * Owns the **research tree** (nodes/branches).
  * Runs **planning**, **branch execution**, **synthesis**, and **judge** chains.
  * Applies **parallelisation rules**, **guardrails**, and **context management**.
* **LangChain** is used as:

  * A convenient way to call models (`ChatOpenAI`, etc.).
  * A way to define **tools** (Deep Research tool wrapper, RAG tool, etc.).
  * A way to build prompt→LLM→structured-output **chains**.

---

## 3. Research Tree & Planning Model

### 3.1 Data models (simplified)

```python
# models.py
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"

class Citation(BaseModel):
    source_id: str    # URL or doc-id
    snippet: str
    score: float
    branch_id: str

class ResearchNode(BaseModel):
    id: str
    parent_id: Optional[str]
    question: str
    status: NodeStatus = NodeStatus.PENDING
    depth: int = 0
    can_parallelize: bool = True
    depends_on: List[str] = []

    created_at: datetime = Field(default_factory=datetime.utcnow)

    notes: str = ""           # branch-local summary / notes
    citations: List[Citation] = []
    confidence: float = 0.0   # 0–1
    children_ids: List[str] = []

class ResearchTree(BaseModel):
    root_id: str
    nodes: Dict[str, ResearchNode]

    def get_executable_nodes(self, max_parallel: int) -> List[ResearchNode]:
        """Return nodes that are ready to run now."""
        candidates = [
            n for n in self.nodes.values()
            if n.status == NodeStatus.PENDING
            and all(
                self.nodes[dep_id].status == NodeStatus.DONE
                for dep_id in n.depends_on
            )
        ]
        # basic prioritisation: shallow depth first; you can add explicit priority later
        candidates.sort(key=lambda n: n.depth)
        return candidates[:max_parallel]
```

### 3.2 Planning output

Planning step converts the user’s high-level question into **sub-tasks**:

```python
class SubTask(BaseModel):
    id: str
    parent_id: str
    description: str
    depends_on: List[str] = []
    parallelizable: bool = True
    priority: int = 0   # higher = more important

class PlanningOutput(BaseModel):
    new_nodes: List[SubTask]
    updates: List[dict]  # e.g. change status, confidence of existing nodes
```

### 3.3 Heuristics for *parallel* vs *sequential*

The planner must explicitly mark `parallelizable: true/false`. Guidance to the planning chain:

> Mark `parallelizable: true` when this sub-question depends only on:
>
> * The **original question**, and/or
> * The **global summary**,
>   not on the answer to any specific other sub-task.

Examples:

* **Parallel**:

  * “Deep research React for these axes…”
  * “Deep research Vue for these axes…”
  * “Deep research Svelte for these axes…”
* **Sequential**:

  * “Rank the top 3 frameworks based on the findings from all framework-specific branches.”
  * “Given the pros/cons already identified, design mitigations for top 3 risks.”

Implementation rule of thumb:

* Planner sets `depends_on` for nodes that must wait for specific others.
* Orchestrator only runs nodes where `depends_on` are all `DONE`.
* Among those, only runs nodes with `parallelizable=True` in parallel; others can be run one-by-one or in smaller groups.

---

## 4. Parallel Execution Strategy

### 4.1 Selection loop

Pseudo-code for selecting and running branches:

```python
# orchestrator.py (inside main loop)
ready = tree.get_executable_nodes(settings.max_parallel_branches)

if not ready:
    # either done or stuck
    break

for node in ready:
    node.status = NodeStatus.RUNNING

results = await asyncio.gather(
    *[self.execute_branch(node, tree) for node in ready]
)

# results will update each node’s notes, confidence, citations, etc.
```

### 4.2 When to parallelise

**Parallelise**:

* Per-entity deep-dive under a shared analysis template:

  * Compare multiple frameworks/libraries/competitors.
  * Research multiple product features independently.
* Data flows from these branches mostly into the **final synthesis**, not into each other.

**Keep sequential**:

* When a task asks “given the results of X, do Y”.
* For deep hypothesis refinement loops:

  * E.g. “design architecture → evaluate → refine → evaluate...”
* When your cost or token budget is tight: you may choose to collapse parallelism and run tasks sequentially to allow early stopping.

---

## 5. Memory & Context Management

### 5.1 Three layers of memory

1. **Short-term / call-local context**

   * What’s passed into a single LLM call (current question, branch notes, small slice of global summary).
   * Built ad-hoc per call.

2. **Branch-local memory**

   * Lives on each `ResearchNode`:

     * `notes`: a rolling summary of what this branch has discovered.
     * `citations`: relevant sources used in this branch.
   * May also have branch-specific embeddings/namespace in vector store.

3. **Global research memory**

   * A **rolling global summary** string `G` of the entire project.
   * A **vector store** with chunks from:

     * Deep Research reports,
     * Internal KB results,
     * Branch summaries.
   * All chunks tagged with metadata: `{branch_id, node_id, source_type, url/doc-id, confidence}`.

### 5.2 Rx2 pattern (rolling global summary)

To avoid R×N context explosion:

* Let `R` be your target report length (say 4–6k tokens).
* Maintain global summary `G` as a capped-length text.

Whenever a new branch completes:

```text
New_G = Summarise(GlobalSummary = G_prev, BranchSummary = B_i) → truncated back to ~R tokens
```

Implementation:

* Use a **cheaper, smaller model** (e.g. gpt-4.1-mini) to do:

  * “Combine current global summary and this branch summary into a new global summary; keep it ≤ R tokens.”
* That means each update call has context size ~2R, not N×R.

### 5.3 Hierarchical reduction for many branches

For large jobs (e.g. 30–50 branches):

1. **Per-branch compression**:

   * Deep Research returns a long report.
   * Compress it into:

     * ~1–2k token summary.
     * Structured bullets (pros/cons, metrics, etc.).
   * Store in vector store.

2. **Grouping**:

   * Group related branches:

     * e.g. “Frontend frameworks”, “Backend frameworks”.
     * e.g. “Core product features”, “Engagement features”.
   * Summarise each group into a **group meta-summary**.

3. **Top-level synthesis**:

   * Final answer uses:

     * Global summary `G_final`.
     * Group meta-summaries.
     * Optionally, retrieved detail chunks pulled from vector store.

### 5.4 Retrieval instead of giant context

* Whenever you need more detail for synthesis or judging:

  * Query vector store for relevant chunks by:

    * original question,
    * node/section label,
    * or explicit query text.

* Each synthesis step sees:

  * `G_final` (or a subset),
  * Top-k retrieved chunks,
  * Possibly a short checklist of the user’s requirements.

### 5.5 Avoiding repeated web/deep research

* Maintain a **cache** for:

  * Web search queries,
  * Deep Research tasks.

Simple approach:

* Normalise queries (lowercase, strip).
* Cache key: `(tool_name, normalized_query)`.
* If a new task matches an existing key (or is highly similar), reuse previous result instead of re-running the tool.

---

## 6. Deep Research Integration with LangChain

### 6.1 Philosophy

* Treat OpenAI **Deep Research** as a **heavy tool** that itself:

  * plans,
  * performs multi-round web search & reasoning,
  * and outputs a long, citation-rich report for one focused question.

* LangChain orchestrator:

  * Decomposes the user’s main question into **multiple Deep Research jobs** + **internal RAG jobs**.
  * Runs them in parallel or sequence according to dependencies.
  * Compresses and aggregates results.

### 6.2 Deep Research tool wrapper

Create a LangChain tool that calls the OpenAI Deep Research API via `openai` (or `openai`’s `responses.create`):

```python
# tools.py
from langchain.tools import tool
from openai import OpenAI

client = OpenAI()

@tool("deep_research_report")
def deep_research_report(query: str) -> str:
    """
    Run a Deep Research job for a single focused query.
    Returns the final report text (headings, bullets, citations).
    Use for large, web-heavy questions.
    """
    system_message = (
        "You are a professional researcher. "
        "Produce a structured, citation-rich report with headings, "
        "bullet points, and clear recommendations."
    )

    resp = client.responses.create(
        model="o4-mini-deep-research-2025-06-26",  # or current DR model
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            },
        ],
        reasoning={"summary": "auto"},
        tools=[{"type": "web_search_preview"}],  # add MCP / internal KB later
    )

    # Extract just the final natural-language report text.
    # Structure of resp.output may vary; adapt as needed.
    final_message = resp.output[-1].content[0].text
    return final_message
```

> **Phase 2:** Add MCP tool config to let Deep Research call your internal KB directly. For MVP, just let Deep Research hit the web; use separate LangChain RAG for internal markdown.

### 6.3 Internal KB RAG tool

Assuming Chroma (or any vector store) built from your markdown:

```python
# tools.py (continued)
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

emb = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="internal_md",
    embedding_function=emb,
    persist_directory="chroma_data",
)

@tool("internal_rag")
def internal_rag(query: str, k: int = 5) -> str:
    """
    Query the internal markdown knowledge base.
    Returns a small synthesized answer with inline references.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    # MVP: simple concatenation with sources
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] {src}\n{d.page_content[:1000]}")

    return "\n\n".join(parts)
```

### 6.4 When to use Deep Research vs RAG

Heuristics in your **planner** or **orchestrator**:

* Use **Deep Research** when:

  * The sub-question is open-ended, web-heavy, or broad:

    * “What is the current landscape of X?”
    * “What are the tradeoffs between frameworks A/B/C?”
    * “Summarise recent trends in Y across the last 12–24 months.”

* Use **internal RAG** when:

  * The question clearly depends on internal docs:

    * “What have we decided about our own architecture?”
    * “How is this feature spec’d in existing designs?”
  * Or when the user explicitly says “internal only”.

Implementation:

* Planner can tag each sub-task with `source_hint: "web" | "internal" | "mixed"`.
* Orchestrator chooses which tool to route to based on that hint.

---

## 7. Chains (LangChain Runnables)

### 7.1 Planning chain

```python
# chains.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .models import ResearchNode
from .models import PlanningOutput

def build_planning_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research planner. You maintain a tree of research questions.\n"
         "Given the original question and current nodes, propose new sub-tasks.\n"
         "For each sub-task, decide if it can run in parallel and what it depends on.\n"
         "Return ONLY JSON matching the provided schema."),
        ("human",
         "Original question:\n{original_question}\n\n"
         "Completed nodes:\n{completed_nodes_json}\n\n"
         "Pending nodes:\n{pending_nodes_json}\n\n"
         "Judge feedback (if any):\n{judge_feedback}\n")
    ])

    chain = prompt | llm.with_structured_output(PlanningOutput)
    return chain
```

### 7.2 Branch execution chain (non-DR branches)

For branches that stay within LangChain tools (internal RAG, simple search):

```python
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .tools import internal_rag  # and optionally web_search, python_eval

def build_branch_execution_chain(llm: ChatOpenAI) -> Runnable:
    tools = [internal_rag]  # add other tools if needed
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant working on a sub-question within a larger project.\n"
         "You can use the provided tools when helpful.\n"
         "Gather evidence, then produce:\n"
         "1) a concise summary,\n"
         "2) a confidence rating 0-1,\n"
         "3) a list of citations (source_id + brief note)."),
        ("human",
         "Sub-question:\n{question}\n\n"
         "Branch-local notes:\n{branch_notes}\n\n"
         "Relevant global context:\n{global_context}\n")
    ])

    return prompt | llm_with_tools
```

Deep Research branches can bypass this and call `deep_research_report` directly from the orchestrator.

### 7.3 Synthesis chain

```python
def build_synthesis_chain(llm: ChatOpenAI) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are synthesizing results of a multi-branch research process.\n"
         "Use the research tree and global summary as your fact base.\n"
         "Highlight uncertainties and contradictions. Always keep citations."),
        ("human",
         "Original question:\n{original_question}\n\n"
         "Global summary:\n{global_summary}\n\n"
         "Research tree JSON:\n{tree_json}\n\n"
         "Produce:\n"
         "1. Executive summary\n"
         "2. Key findings (with references)\n"
         "3. Contradictions or open questions\n"
         "4. Limitations & caveats\n"
         "5. Overall confidence (0-1)\n")
    ])
    return prompt | llm
```

### 7.4 Judge chain

```python
from pydantic import BaseModel
from typing import List

class JudgeOutput(BaseModel):
    is_complete: bool
    recommend_more_iterations: bool
    missing_aspects: List[str]
    weak_evidence_areas: List[str]
    overall_confidence: float

def build_judge_chain(llm: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict research evaluator.\n"
         "Given the original question, a candidate synthesis, and a high-level tree, "
         "judge completeness, quality, and coverage. Be conservative: if in doubt, "
         "recommend more work."),
        ("human",
         "Original question:\n{original_question}\n\n"
         "Candidate synthesis:\n{synthesis}\n\n"
         "High-level node list:\n{nodes_overview}\n\n"
         "Return JSON with fields: is_complete, recommend_more_iterations, "
         "missing_aspects, weak_evidence_areas, overall_confidence.")
    ])

    return prompt | llm.with_structured_output(JudgeOutput)
```

---

## 8. Orchestrator Loop (High-Level)

```python
# orchestrator.py
import time
import asyncio
from typing import Dict
from langchain_openai import ChatOpenAI
from .config import ResearchSettings
from .models import ResearchTree, ResearchNode, NodeStatus
from .chains import (
    build_planning_chain,
    build_branch_execution_chain,
    build_synthesis_chain,
    build_judge_chain,
)
from .tools import deep_research_report

class ResearchOrchestrator:
    def __init__(self, settings: ResearchSettings):
        self.settings = settings
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
        )
        self.planning_chain = build_planning_chain(self.llm)
        self.branch_chain = build_branch_execution_chain(self.llm)
        self.synthesis_chain = build_synthesis_chain(self.llm)
        self.judge_chain = build_judge_chain(self.llm)

        # memory: vector store, global summary, caches...
        self.global_summary = ""
        self.search_cache: Dict[str, str] = {}
        self.total_tokens = 0

    async def run(self, question: str) -> dict:
        start = time.time()
        tree = self._init_tree(question)
        iteration = 0

        # 1) initial plan
        await self._initial_plan(tree, question)

        while True:
            iteration += 1
            if iteration > self.settings.max_iterations:
                break
            if self._depth_exceeded(tree):
                break
            if self._limits_exceeded(start):
                break

            # 2) select executable nodes
            executable = tree.get_executable_nodes(
                self.settings.max_parallel_branches
            )
            if not executable:
                break

            # 3) execute branches (some with Deep Research)
            for node in executable:
                node.status = NodeStatus.RUNNING

            await asyncio.gather(
                *[self.execute_branch(node, tree) for node in executable]
            )

            # 4) temporary synthesis + judge
            tmp_synthesis = await self._temporary_synthesis(tree, question)
            judge_out = await self._judge(tmp_synthesis, tree, question)

            if judge_out.is_complete and not judge_out.recommend_more_iterations:
                final = await self._final_synthesis(tree, question)
                return final

            # 5) re-plan using judge feedback
            await self._replan(tree, question, judge_out)

        # fallback final synthesis
        final = await self._final_synthesis(tree, question, incomplete=True)
        return final

    async def execute_branch(self, node: ResearchNode, tree: ResearchTree):
        """Execute a single branch; choose Deep Research or RAG depending on node type/hints."""
        # for MVP you can use simple heuristics on node.question
        if self._should_use_deep_research(node):
            report = deep_research_report(node.question)
            summary = await self._compress_report(report)
            node.notes = summary
            node.confidence = 0.8  # or estimate via a small scoring chain
            node.citations.extend(self._extract_citations(report))
            await self._add_to_global_memory(node, report)
        else:
            branch_notes = node.notes
            global_ctx = await self._retrieve_global_context(node)
            resp = await self.branch_chain.ainvoke({
                "question": node.question,
                "branch_notes": branch_notes,
                "global_context": global_ctx,
            })
            summary, conf, cits = self._parse_branch_response(resp)
            node.notes = summary
            node.confidence = conf
            node.citations.extend(cits)
            await self._add_to_global_memory(node, summary)

        node.status = NodeStatus.DONE
        await self._update_global_summary(node)

    # ... helper methods: _init_tree, _initial_plan, _replan, _temporary_synthesis,
    # _final_synthesis, _depth_exceeded, _limits_exceeded, _add_to_global_memory,
    # _update_global_summary, _should_use_deep_research, _compress_report,
    # _extract_citations, _retrieve_global_context, _parse_branch_response, etc.
```

---

## 9. Config & Guardrails

```python
# config.py
from pydantic import BaseSettings

class ResearchSettings(BaseSettings):
    openai_api_key: str

    # LLM choices
    model_name: str = "gpt-4.1"
    temperature: float = 0.2

    # Research limits
    max_depth: int = 3
    max_iterations: int = 6
    max_parallel_branches: int = 3

    max_tokens_per_call: int = 4000
    max_total_tokens: int = 150_000
    max_runtime_seconds: int = 900  # 15 minutes

    estimated_cost_limit_usd: float = 10.0

    class Config:
        env_file = ".env"
```

At runtime:

* Track `total_tokens` (via callbacks or your own estimator).
* Check limits each iteration; if exceeded, break and synthesise with what you have.
* Maintain a “reason for stopping” flag (`"complete" | "budget_exceeded" | "time_exceeded" | "no_more_tasks"`).

---

## 10. MVP vs Phase 2

### MVP (what to implement first)

* **Single CLI**:

  * `python -m deep_research.cli "your question here"`
* **Tree but no re-planning yet**:

  * Initial planning → flat or shallow tree of sub-questions.
  * Run all sub-questions once (some via Deep Research, some via RAG).
  * Rolling global summary using Rx2 pattern.
  * Final synthesis (no judge loop).
* **Simplified heuristics**:

  * Hard-code `parallelizable=True` for entity-comparison style questions.
  * Hard-code “Deep Research vs RAG” choices based on keyword hints.
* **Basic vector store**:

  * Chroma / PGVector for branch summaries.
  * Use retrieval only in final synthesis.

### Phase 2 (when MVP is stable)

* **Full judge → re-plan loop**:

  * Add `JudgeOutput` and iterative planning to spawn new branches over multiple iterations.
* **Better tool dispatch**:

  * Fine-tune heuristics or use a small classifier LLM to tag sub-tasks as `web`, `internal`, or `mixed`.
* **MCP for internal KB**:

  * Wrap your markdown KB as an MCP server and let Deep Research use it directly.
* **Rich UI / logging**:

  * Add LangSmith tracing, per-branch logs, and detailed run reports.
* **Persistence & resume**:

  * Persist `ResearchTree` + global summary + vector store to disk.
  * Support resuming a previous research session by ID.

---

This file should give your coding assistant enough structured context to:

* Generate the actual modules (`config.py`, `models.py`, `tools.py`, `chains.py`, `orchestrator.py`, `cli.py`).
* Implement the Deep Research integration as a LangChain tool.
* Respect the parallelisation rules and context management strategy you want.
