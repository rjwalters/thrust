---
name: repo
description: "Repo hygiene for thrust — audit root clutter, orphaned files, README accuracy, gitignore rules, broken links, and crate-publish exclude drift. Interactive: reports findings and discusses fixes before changing anything."
---

# Repo Maintenance

Keep the `thrust` repo tidy as it grows. These checks catch the drift that
accumulates from training runs, demo iterations, and consolidation: stray model
checkpoints at the root, stale directories, broken doc links, and `Cargo.toml`
`exclude` rules that no longer match reality.

Ported from the Sphere monorepo `/repo` skill and adapted to thrust's layout
(Rust library crate + `web/` demo + Loom orchestration).

## Philosophy

**Report first, fix second.** Never silently change files. Present findings,
discuss, then apply only the fixes the user approves. Scope matters — accept an
optional path argument to limit a check to a subtree. Don't be noisy: only flag
things that are actually wrong or confusing.

## Usage

```
/repo            # full sweep (all checks below), repo-wide
/repo audit      # same as bare invocation
/repo orphans    # just the orphaned-file check
/repo gitignore  # just the gitignore check
/repo links      # just the broken-link check
/repo readme     # just the README-accuracy check
/repo <path>     # scope any check to a subtree, e.g. /repo orphans web/
```

When run with no argument, run every check and compile one grouped report.

## thrust conventions (what "correct" looks like)

- **Library crate.** `src/` is the crate. The repo root should hold only
  standard top-level files: `Cargo.toml`, `Cargo.lock`, `README.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE-*`, `ROADMAP.md`, toolchain/format
  configs (`rust-toolchain.toml`, `rustfmt.toml`, `clippy.toml`), `Makefile`,
  `package.json`, and dotfiles. Anything else at root is suspect.
- **Model checkpoints live in `models/`.** Canonical `.pt` policies go under
  `models/policies/<env>/`. Web-demo weights go in `web/public/`. Loose
  `*_model*.{pt,json}` at the repo root are almost always stragglers — duplicates
  of a canonical copy elsewhere. Check with `md5` before deleting.
- **`ROADMAP.md` stays at the root.** Loom roles (architect, guide, champion,
  hermit) grep for `ROADMAP.md` at the repo root on case-sensitive systems
  (`grep ... docs/roadmap.md ROADMAP.md`). Do **not** move it into `docs/`.
  Other planning docs (e.g. `WASM_ROADMAP.md`, `MULTI_AGENT_DESIGN.md`) belong in
  `docs/`.
- **Crate-publish `exclude`.** `Cargo.toml` has an `exclude = [...]` list that
  keeps non-library assets out of the published tarball (under the 10 MB
  crates.io limit). Every entry should still match a path; entries that match
  nothing are dead and should be removed. Files newly added at the root that
  aren't needed to build the crate must be added (or the dir excluded).
- **Loom runtime state.** `.loom/` holds heavily-gitignored daemon state
  (`*.pid`, `*.sock`, `*.log`, `claims/`, `signals/`, …). Never flag these as
  orphans or under-ignored — they are intentional local-only state.

## Checks

### 1. Root clutter / orphaned files (`orphans`)
Find files that have no consumers, with special attention to the repo root.
- Loose `*_model*.{pt,json}`, `*_results.json`, `*.log`, result directories
  (`*_results_*/`), and venvs at the **root**. For each, `git grep` the basename
  across tracked files (excluding lockfiles) to see what actually references it.
  A root model file that the web app fetches from `web/public/` instead is a
  straggler.
- Scripts in `scripts/` not referenced by any other script, the `Makefile`, CI,
  or docs.
- Data files (`.json`, `.yaml`, `.csv`) not imported or referenced anywhere.
- Generated output that should be local-only (`target/`, `pkg/`, `web/dist/`,
  `*.tmp`). Confirm it's gitignored; if not, propose a rule.
- Empty directories.

For each orphan show: path, size, what you grepped for and the hit count, and a
suggested action (delete / move to canonical location / add a reference). Ask
before acting — some standalone files are legitimate.

### 2. README accuracy (`readme`)
- ASCII file trees (```…``` blocks with `├──`/`└──`) compared against the actual
  listing — entries listed but missing, or present but unlisted.
- Descriptions that are factually wrong ("gitignored" when tracked, wrong path).
- Directories with >5 files and no README (skip build-output dirs).
Preserve the README's voice; fix only the factual inaccuracies.

### 3. Broken links (`links`)
- Markdown `[text](path)` relative links whose target doesn't exist on disk.
  Skip external URLs (`http(s)://`) and anchor-only links (`#section`).
- Doc-comment references in `src/**` to planning docs (e.g. `WASM_ROADMAP.md`) —
  flag when the referenced path no longer resolves after a move.
- After a file move, suggest the most likely new target (fuzzy match on
  filename). Ask before fixing.

### 4. Gitignore hygiene (`gitignore`)
- **Under-ignored:** tracked files that look generated (`target/`, `*.pt` outside
  `models/`, `pkg/`, `web/dist/`, `*.tmp`, `*.log`).
- **Over-ignored:** rules excluding things that probably should be tracked in
  this repo. Always keep ignored: `.env*` (secrets), `target/`, `node_modules/`,
  venvs, OS files, and the `.loom/` runtime-state entries.
- **Dead rules:** `.gitignore` patterns matching zero files (stale after
  cleanup). Confirm with `git check-ignore` / `git ls-files`.
- **Large untracked files (>1 MB):** surface for a track / ignore decision.

### 5. Crate-publish exclude drift (`audit` only)
- Every `Cargo.toml` `exclude` entry should still match a path on disk — list
  dead entries to remove.
- Non-library files/dirs newly present at the root that aren't excluded and
  aren't needed to build the crate — propose adding them.
- Cross-check: after removing a root file, remove its now-dead `exclude` entry.

## Output format

Group findings by category with a severity column:

```
## Repo Audit — <scope>

### Root clutter / orphans (N findings)
| Severity | Path | Issue |
|----------|------|-------|
| warn | cartpole_model_best.pt | duplicate of models/policies/cartpole/ (same md5) |

### Summary
- N orphans, N readme issues, N broken links, N gitignore issues, N exclude-drift
```

Severity: **critical** (actively misleading/broken — dead link in README,
gitignore hiding something important), **warn** (should fix — straggler file,
stale README, dead exclude entry), **info** (nice to fix — missing README in a
tiny dir).

After presenting, ask: "Want me to fix any of these?" For file removals/moves,
prefer `git rm` / `git mv` to preserve history, and update every inbound
reference in the same change.
