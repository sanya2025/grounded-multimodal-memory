# 4. Separate shared `docs/` from personal `notes/`

Status: Accepted

## Context

`docs/` had accumulated a mix of two different kinds of content: material
meant for anyone reading the repo (architecture rationale, references) and
material that was really personal working notes for one contributor's own
process — a blog-drafting file, a changelog, a project runbook, hardware
notes written for a specific personal machine, and HPC/Slurm walkthroughs
tied to one university's cluster. None of the personal-notes content was
sensitive, but it also wasn't meant to be part of what the repo presents to
readers on GitHub, and mixing the two made `docs/` unclear as a signal of
"this is the project's documentation."

## Decision

Personal/process content (`BLOG_SERIES.md`, `CHANGELOG.md`,
`PROJECT_RUNBOOK.md`, `HARDWARE_NOTES.md`, `REFERENCES.md`, and the
`slurm/` HPC walkthrough) moved to a `notes/` directory that is fully
git-ignored (`notes/` in `.gitignore`) — tracked in no git history, never
pushed. `docs/` is reserved for documentation meant to be part of the
tracked, shared repository, starting with these ADRs.

## Consequences

- `notes/` content has no git history and isn't backed up by pushing to the
  remote — losing the local clone loses it. Treat anything that needs
  durability or that should survive a "just reclone the repo" as `docs/`
  content instead.
- Cross-links between the moved files (e.g. `HARDWARE_NOTES.md` →
  `slurm/HPC_SJSU.md`) still resolve because the whole set moved together,
  preserving relative paths.
- Anyone else who clones this repo will not see `notes/` at all — by design,
  but worth remembering if content is later found to actually belong in
  `docs/` for other readers.
