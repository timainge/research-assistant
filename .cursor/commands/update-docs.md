# Update Project Documentation

## Overview

Review the current implementation status and update all project documentation to ensure it accurately reflects the current state of the codebase, decisions made, and lessons learned.

## Steps

1. **Review implementation status**
   - Check `notes/todo.md` for completed and in-progress tasks
   - Identify any tasks that have been completed but not marked done
   - Note any new tasks that have emerged during development

2. **Update `notes/todo.md`**
   - Mark completed tasks as done
   - Add any new tasks discovered during development
   - Re-prioritize if needed based on current understanding
   - Move deferred items to the appropriate section

3. **Update `README.md`**
   - Ensure installation instructions are current
   - Update usage examples if CLI or API has changed
   - Add any new features or configuration options
   - Verify all code examples still work

4. **Update `notes/design.md`** (only if needed)
   - Add new architectural decisions to the Decision Log
   - Update component descriptions if they've changed
   - Document any new abstractions or patterns introduced
   - Note any deviations from the original design

5. **Capture shared learnings**
   - Identify patterns, conventions, or best practices discovered
   - Ask user if they should be added as Cursor rules in `.cursor/rules/`

## Documentation Checklist

- [ ] `notes/todo.md` reflects current task status
- [ ] Completed tasks are marked done
- [ ] New tasks are captured
- [ ] `README.md` installation steps are accurate
- [ ] `README.md` usage examples work
- [ ] `notes/design.md` reflect the current state of design decisions (this is not a log, but a current/future state)
- [ ] New patterns/conventions offered as potential rules

## Output

Provide a summary of:
- What documentation was updated
- Key changes made
- Any new Cursor rules suggested for user approval
