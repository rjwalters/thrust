# Releasing thrust-rl

This document is the **maintainer-facing** guide to cutting a new
release of `thrust-rl` to crates.io. The PR workflow (Builder + Judge +
merge) is documented elsewhere; this file picks up after a green `main`
that you want to publish as a new version.

## Versioning policy

`thrust-rl` follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with the **pre-1.0 convention** that breaking API changes land in MINOR
bumps until we cut `1.0.0`:

| Change                                | Bump  | Example          |
| ------------------------------------- | ----- | ---------------- |
| Bug fix, doc-only, additive non-API   | PATCH | `0.1.0 -> 0.1.1` |
| New public API, removed/renamed API   | MINOR | `0.1.x -> 0.2.0` |
| Stable-API guarantee (post-1.0 only)  | MAJOR | `0.x   -> 1.0.0` |

Until `1.0.0`, MAJOR is reserved. Once we ship `1.0.0`, MAJOR is for
breaking API changes and MINOR is for additive ones, per standard
semver.

The **single source of truth** for the version is the root
`Cargo.toml`'s `[package].version` field. There is no
`scripts/version.sh` in this repository, and no other file needs to be
edited in lock-step for a release.

## Pre-flight checks

Before tagging:

```bash
# 1. CI is green on main
gh run list --branch main --limit 5

# 2. No open PRs that should land first
gh pr list --state open

# 3. Working tree is clean and on main
git checkout main && git pull && git status
```

If CI is failing or there are uncommitted changes, stop and fix first.

## Step 1: Update `CHANGELOG.md`

1. Move items from the `## [Unreleased]` section into a new
   `## [X.Y.Z] - YYYY-MM-DD` block (today's date, UTC).
2. Group changes under `### Added`, `### Changed`, `### Fixed`,
   `### Removed`, `### Deprecated`, `### Security` as appropriate.
   Omit empty sections.
3. Update the link references at the bottom of the file:
   ```text
   [Unreleased]: https://github.com/rjwalters/thrust/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/rjwalters/thrust/releases/tag/vX.Y.Z
   ```
4. Commit:
   ```bash
   git add CHANGELOG.md
   git commit -m "docs: prepare CHANGELOG for vX.Y.Z"
   ```

## Step 2: Bump `Cargo.toml`

Edit `Cargo.toml`'s `[package].version`:

```toml
[package]
version = "X.Y.Z"
```

Then rebuild so `Cargo.lock` picks up the new version:

```bash
cargo check --no-default-features
```

Commit:

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: bump version to vX.Y.Z"
```

(`Cargo.lock` is `.gitignore`d in this repo, so only `Cargo.toml`
will appear in the commit. That's expected.)

## Step 3: Validate the publish

Run `cargo publish --dry-run` to make sure the manifest is publishable
and the tarball builds:

```bash
# Without libtorch on the host machine, run with --no-default-features:
cargo publish --dry-run --no-default-features

# With libtorch available (either via $LIBTORCH or LIBTORCH_USE_PYTORCH=1),
# run the full default-features dry-run too:
LIBTORCH_USE_PYTORCH=1 cargo publish --dry-run
```

Inspect the tarball contents:

```bash
cargo package --list
```

Confirm the file list does **not** include:
- Model checkpoints (`*.pt`, `*.safetensors`, `models/**`).
- The `web/` React app, `web-old/`, or `envs/bucket-brigade/` submodule.
- Training/automation scripts under `scripts/`.
- `.loom/`, `.claude/`, `.github/`.

If something unexpected slipped in, update the `exclude = [...]` field
in `[package]` (`Cargo.toml`) and re-run `cargo package --list`. Aim
for a tarball well under crates.io's 10 MB limit; v0.1.0 ships at
~300 KB.

## Step 4: Push the version commits

```bash
git push origin main
```

Wait for CI to go green on the head of `main` before tagging.

## Step 5: Tag and create the GitHub Release

Tag the version-bump commit (or whichever commit you intend as the
release point):

```bash
git tag -a vX.Y.Z -m "thrust-rl vX.Y.Z"
git push origin vX.Y.Z
```

Create the GitHub Release with notes pulled from `CHANGELOG.md`:

```bash
# Extract the section for vX.Y.Z from CHANGELOG.md and pass as notes:
gh release create vX.Y.Z \
    --title "vX.Y.Z" \
    --notes-file <(awk '/^## \[X\.Y\.Z\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md)
```

(Substitute the literal version into the `awk` pattern, e.g.
`/^## \[0\.1\.0\]/`.)

No `release.yml` GitHub Actions workflow is configured for v0.1.0, so
no binary artifacts are attached. This is intentional --- consumers
build from source via `cargo install thrust-rl`.

## Step 6: Publish to crates.io

**Only the maintainer with a valid crates.io API token runs this step.**
Do not delegate it to autonomous agents.

```bash
# Make sure you're on the exact tagged commit:
git checkout vX.Y.Z

# Authenticate (one-time per machine; token stored in
# ~/.cargo/credentials.toml):
cargo login <your-crates.io-token>

# Publish for real (no --dry-run this time):
LIBTORCH_USE_PYTORCH=1 cargo publish

# Or, if you trust the dry-run and want to skip the verification rebuild:
cargo publish --no-verify
```

`cargo publish` will:
1. Re-run the manifest verification.
2. Rebuild the tarball.
3. Upload to crates.io.

Within a few minutes:
- `https://crates.io/crates/thrust-rl/X.Y.Z` is live.
- `https://docs.rs/thrust-rl/X.Y.Z/` starts a doc build (takes 5-30
  minutes for the first version).

## Step 7: Verify and announce

```bash
# crates.io
curl -s https://crates.io/api/v1/crates/thrust-rl | jq '.crate.max_stable_version'

# docs.rs build status
open https://docs.rs/thrust-rl/X.Y.Z/
```

If docs.rs failed (commonly because of `tch` / libtorch), open a
follow-up issue with the failing log and consider whether to publish
without the `training` feature on docs.rs by adding to `Cargo.toml`:

```toml
[package.metadata.docs.rs]
no-default-features = true
features = []
```

## Yanking a bad release

If you discover a serious bug post-publish:

```bash
cargo yank --vers X.Y.Z
```

Yanking removes the version from new dependency resolution but does not
delete it. Existing `Cargo.lock` files pinning that version continue to
work. Follow up with a `X.Y.(Z+1)` patch release containing the fix.

## Out of scope (today)

These are explicit non-goals for the current release procedure:

- **Automated `cargo publish` from CI.** The first few releases are
  hand-published. Automate later, once the procedure is stable.
- **Binary release artifacts.** No `release.yml` builds a binary
  tarball; consumers `cargo install`.
- **Workspace publishing.** `thrust-rl` is a single crate today.
- **Bumping `web/package.json`'s version.** The web subproject is not
  an npm package and is excluded from the published tarball.
