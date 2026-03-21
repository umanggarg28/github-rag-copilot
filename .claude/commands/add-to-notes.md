# Add a Note Entry

Create a new entry in the `notes/` directory documenting what was just built.

## Usage
```
/add-to-notes
```

## Steps

1. Look at the most recent note file in `notes/` to determine the next number (NNN)
2. Look at recent git changes (`git diff HEAD~1` or `git status`) to understand what was built
3. Create `notes/NNN-<short-title>.md` with this structure:

```markdown
# Note NNN — <Title>

**Date:** <today>
**PR:** <PR title or "in progress">

---

## What was built
<what was added/changed>

## Key decisions
<why these choices were made>

## Concepts learned
<RAG/code concepts this feature demonstrates>

## What's next
<next steps>
```
