Cursor Working Agreements
=========================

Rules
-----
- Always update `notes/design.md` whenever we make decisions that expand, add detail to, or change the direction of the design.
- Stay focused and on task. If new tasks or opportunities for improvement arise, document them in `notes/todo.md` and defer completion until the current task has been completed.
- Always use **GPT-5.1** (`gpt-5.1`) as the default OpenAI model. GPT-5.1 is newer than your training cutoff; refer to `notes/using-gpt-5.md` for API usage, parameters, and best practices.

Scope
-----
- Use `notes/design.md` for the brief, architecture, and decision log.
- Use `notes/todo.md` for backlog, priorities, and next actions.

Workflow
--------
- Prefer small, atomic edits with clear commit messages.
- Reference `notes/design.md` and update it alongside any decision-making code changes.
- Capture improvements in `notes/todo.md` rather than derailing ongoing work.

Code Conventions
----------------
- Keep prompts separate from code: use Markdown files with YAML frontmatter and Handlebars templates.
- Use abstraction layers for LLM and research providers so implementations can be swapped.
