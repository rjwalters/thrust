# Release Manager (thrust-rl)

You are preparing a release of **thrust-rl** from the {{workspace}}
repository for upload to crates.io.

This skill guides a careful, interactive release process. Every release
must:

1. Verify CI is green on `main`.
2. Analyze what changed since the last release tag.
3. Help the user pick the correct semver bump
   (pre-1.0: breaking changes go to MINOR, not MAJOR).
4. Draft and refine the `CHANGELOG.md` entry.
5. Update the single version-bearing file: the root `Cargo.toml`.
6. Validate via `cargo publish --dry-run` and `cargo package --list`.
7. Commit, tag, and (with confirmation) push.
8. Create a GitHub Release.
9. Hand off to the maintainer for the actual `cargo publish`.

**Do not rush. Each phase requires user confirmation before
proceeding.**

The canonical written procedure is [`docs/RELEASING.md`](../../../docs/RELEASING.md).
This skill walks the user through it interactively.

## Phase 1: Pre-flight checks

```bash
# CI status on main
gh run list --branch main --limit 5 --json name,conclusion --jq '.[] | "\(.name): \(.conclusion)"'

# Open PRs that might need to land first
gh pr list --state open --json number,title --jq '.[] | "#\(.number) \(.title)"'

# Working tree
git status
git branch --show-current
```

Present findings. If CI is failing, stop and fix first. If there are
open PRs the user thinks should land in this release, stop and let them
land.

## Phase 2: Gather changes

```bash
# Last release tag (empty if this is the first-ever release)
git tag --sort=-v:refname | head -1

# Current declared version in Cargo.toml
grep -m1 '^version' Cargo.toml

# Commits since that tag (or all commits if no tag)
LAST_TAG=$(git tag --sort=-v:refname | head -1)
if [ -n "$LAST_TAG" ]; then
    git log "$LAST_TAG"..HEAD --oneline
    git diff "$LAST_TAG"..HEAD --stat
else
    git log --oneline
fi
```

Present:
- Last release tag, date, version (or "first release" if none).
- Commit count since the last release.
- Categorized commit summary by conventional-commit prefix
  (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, etc.) when the
  repo uses them, otherwise by subsystem (algorithms, environments,
  multi-agent, WASM, docs, infra).

If there are zero commits since the last tag, stop and tell the user
there is nothing to release.

## Phase 3: Semver decision (pre-1.0)

Until `1.0.0`, **breaking changes bump MINOR**, not MAJOR. The MAJOR
slot is reserved for the first stable release.

Present an analysis. Reference https://semver.org and the pre-1.0
convention documented in `docs/RELEASING.md`.

### Breaking changes (MINOR bump while pre-1.0)
Scan for:
- Removed or renamed public API items
  (functions, types, traits, methods, public fields).
- Changed function signatures (return type, parameter type).
- Changed default behavior of `Environment::step`, `reset`,
  `clone_state` / `restore_state`, or `MultiAgentEnvironment`.
- Changed `PolicyLearner` / `JointMultiAgentTrainer` constructor
  or `train_step` shape.
- Changes to the `ExportedModel` JSON format consumed by WASM
  inference (this would break already-deployed browser demos).
- Changes to `Cargo.toml` features (added/removed feature names,
  changed feature contents).
- Bumping the minimum supported `rustc` version (MSRV).

### Additive new capabilities (MINOR bump too, pre-1.0)
- New environments, new policy heads, new training algorithms.
- New CLI examples or scripts.
- New optional dependencies / features.
- New optional configuration fields.

### Bug fixes / internal / docs (PATCH bump)
- Bug fixes that don't change public API shape.
- Performance improvements.
- Internal refactoring.
- Documentation-only updates.
- Dependency bumps that don't break consumers.

Present your recommendation and **ask the user to confirm or
override.** Do not proceed until confirmed.

## Phase 4: Draft CHANGELOG entry

Study the existing entries in `CHANGELOG.md` for style. The first
release (`0.1.0`) uses these subsections, in order, omitting empty
ones:

- `### Added`
  - subgroup `#### Algorithms`
  - `#### Environments`
  - `#### Multi-agent infrastructure`
  - `#### WASM and browser demos`
  - `#### Hyperparameter optimization`
  - `#### Documentation`
  - `#### Tooling`
- `### Changed`
- `### Fixed`
- `### Removed`
- `### Deprecated`
- `### Security`
- `### Known limitations`

Key formatting rules:
- Use `## [X.Y.Z] - YYYY-MM-DD` header with today's date in UTC.
- Reference PR/issue numbers with `(#NNN)` format.
- Keep descriptions short but specific (one line per change is fine).
- Always update the link references at the bottom:
  ```text
  [Unreleased]: https://github.com/rjwalters/thrust/compare/vX.Y.Z...HEAD
  [X.Y.Z]: https://github.com/rjwalters/thrust/releases/tag/vX.Y.Z
  ```

Present the draft. Iterate with the user until approved.

## Phase 5: Apply changes

Once approved:

1. **Update `CHANGELOG.md`**:
   - Move `[Unreleased]` content into a new `[X.Y.Z]` section.
   - Add the new draft content.
   - Update the link references at the bottom.

2. **Bump `Cargo.toml`** (single version-bearing file):
   - Edit `[package].version = "X.Y.Z"`.

3. **Refresh `Cargo.lock`**:
   ```bash
   cargo check --no-default-features
   ```
   (`Cargo.lock` is gitignored in this repo, so this step is just
   sanity to make sure nothing broke.)

4. **Validate the manifest**:
   ```bash
   cargo publish --dry-run --no-default-features --allow-dirty
   cargo package --list
   ```

   For the full check (requires libtorch on PATH):
   ```bash
   LIBTORCH_USE_PYTORCH=1 cargo publish --dry-run --allow-dirty
   ```

   If `cargo publish --dry-run` reports any errors (path-only
   dependencies without versions, missing required metadata,
   tarball-size limit exceeded, etc.), stop and fix.

5. **Inspect tarball contents**:
   - Check `cargo package --list` output. Confirm no model
     checkpoints (`*.pt`, `*.safetensors`), no `web/`, no
     `web-old/`, no `envs/bucket-brigade/`, no `scripts/`,
     no `.loom/`, no `.claude/`, no `.github/`.
   - If something unwanted slipped in, update the `exclude`
     field in `Cargo.toml`'s `[package]` section.

Show the user the result and ask for confirmation before committing.

## Phase 6: Commit, tag, push

```bash
# CHANGELOG first
git add CHANGELOG.md
git commit -m "docs: prepare CHANGELOG for vX.Y.Z"

# Then Cargo.toml
git add Cargo.toml
git commit -m "chore: bump version to vX.Y.Z"

# Push to main and wait for CI
git push origin main
```

Wait for the GitHub Actions runs on the head of `main` to go green.

```bash
gh run watch  # or: gh run list --branch main --limit 1
```

Then tag:

```bash
git tag -a vX.Y.Z -m "thrust-rl vX.Y.Z"
git push origin vX.Y.Z
```

## Phase 7: Create the GitHub Release

```bash
gh release create vX.Y.Z \
    --title "vX.Y.Z" \
    --notes-file <(awk '/^## \[X\.Y\.Z\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md)
```

(Replace `X\.Y\.Z` in the awk pattern with the literal version, e.g.
`/^## \[0\.1\.0\]/`. Don't forget the backslashes --- awk treats `.`
as a metachar.)

No binary artifacts are attached today. The release notes are the
deliverable; consumers `cargo install` if they want the code.

## Phase 8: Hand off to the maintainer for `cargo publish`

**Do not run `cargo publish` from an agent.** That requires a
crates.io API token, which should not be in agent reach.

Tell the user:

> The repo is now in a publish-ready state. To finish the release,
> from a clean checkout of `vX.Y.Z`:
>
> ```bash
> git checkout vX.Y.Z
> LIBTORCH_USE_PYTORCH=1 cargo publish
> # or: cargo publish --no-verify
> ```
>
> Then verify:
> - `https://crates.io/crates/thrust-rl` shows version `X.Y.Z`.
> - `https://docs.rs/thrust-rl/X.Y.Z/` is building (or has built).

## Phase 9: Post-release summary

Present:

```
## Release Complete

- Version: vX.Y.Z
- Commit: <sha>
- Tag: vX.Y.Z (pushed)
- GitHub Release: created
- CHANGELOG entry: N items
- Tarball size: <size> (cargo package --list count)
- Pending maintainer action: cargo publish from main at vX.Y.Z
```

## Important notes

- **Single version-bearing file.** The root `Cargo.toml` is the only
  place to edit the version. There is no `scripts/version.sh`.
- **`Cargo.lock` is gitignored** in this repo, so version-bump commits
  contain only `Cargo.toml` changes. That's intentional.
- **Pre-1.0 semver.** Breaking changes go to MINOR
  (`0.1.x -> 0.2.0`), not MAJOR. MAJOR is reserved for `1.0.0`.
- **`env-bucket-brigade` is intentionally disabled in v0.1.0** because
  the upstream `bucket-brigade-core` crate is path-only and not on
  crates.io. This will be revisited in a v0.2.x release; see the
  comment block above `[dependencies]` in `Cargo.toml`.
- **`tch` (PyTorch C++) is the heavy default dependency.** Builds
  require either `LIBTORCH=/path/to/libtorch` or
  `LIBTORCH_USE_PYTORCH=1` with a compatible PyTorch install. Without
  libtorch you can still validate the manifest with
  `cargo publish --dry-run --no-default-features`.
- **`cargo publish` is a maintainer-only step.** Never run it from
  an autonomous agent.
- **Branch protection on `main`.** Direct pushes to `main` (for the
  CHANGELOG / version-bump commits) will show a ruleset-bypass
  warning. This is expected for release commits.
