Project Brief
-------------
High-level workspace for a research assistant toolkit. This repo will evolve to include scripts, libraries, and automation to support research workflows.

Goals (initial)
---------------
- Establish a clean structure for code, notes, and rules.
- Keep decision-making documented in `notes/design.md`.
- Maintain a focused backlog in `notes/todo.md`.

Non-goals (initial)
-------------------
- Full productization or deployment concerns until the core workflows are defined.

Dependencies
------------
- Runtime: see `requirements.txt`
- Python: 3.10+ recommended

Architecture (initial)
----------------------
- Package: `src/research_assistant/`
- Tests: `tests/`
- Notes: `notes/`
- Cursor rules: `agents.md` (symlinked under `.cursor/rules/agents.md`)

Process Rules
-------------
- Always update this design document when we make decisions that expand, add detail to, or change the direction of the design.
- Stay focused and on task. If new tasks or opportunities for improvement arise, document them in `notes/todo.md` and defer completion until the current task has been completed.

Decision Log
------------
- 2025-12-03: Initialized repository structure and baseline conventions.


