Research Assistant
==================

Lightweight workspace for a research assistant toolkit.

Quick start
-----------

1) Python environment
- Create a virtual environment:
  - macOS/Linux: `python3 -m venv .venv`
  - Windows (PowerShell): `py -3 -m venv .venv`
- Activate it:
  - macOS/Linux: `source .venv/bin/activate`
  - Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`

2) Project structure
- `src/research_assistant/` — package code
- `tests/` — test files
- `notes/design.md` — project brief, decisions, dependencies
- `notes/todo.md` — backlog and next actions
- `agents.md` — Cursor rules (symlinked via `.cursor/rules/agents.md`)

Conventions
-----------
- Always update `notes/design.md` when decisions expand, add detail, or change direction.
- Stay focused: if new tasks or improvements arise, capture them in `notes/todo.md` and defer until the current task is complete.


