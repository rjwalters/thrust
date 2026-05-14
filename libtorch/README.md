# `libtorch/` — local, gitignored

This directory is **gitignored** (see `.gitignore: /libtorch`). Only this
README is tracked. The actual `libtorch/lib`, `libtorch/include`, etc. are
populated locally by one of:

1. **`scripts/setup-libtorch.sh`** (recommended) — installs PyTorch into
   `./venv/` and does NOT populate `./libtorch/`. With this path you can
   leave this directory empty or `rm -rf` it; the wrappers use the venv.
2. **`scripts/download-libtorch.sh`** — fetches a self-contained libtorch
   zip from pytorch.org into `./libtorch/`. Bundles its own deps so it
   is immune to Homebrew bottle drift on macOS.

**Do NOT** copy or symlink Homebrew's pytorch lib into here:

```text
# This is the broken pattern that caused issue #8:
# cp -R /opt/homebrew/opt/pytorch/libexec/lib/python3.14/site-packages/torch/lib ./libtorch/lib
```

The dylibs in the Homebrew bottle hardcode the SONAME of the specific
`protobuf` / `abseil` they were built against (e.g. `libprotobuf.33.0.0.dylib`),
which immediately desynchronizes from the current Homebrew installation and
produces `dyld: Library not loaded: ...` errors. See
[issue #8](https://github.com/rjwalters/thrust/issues/8) and
`docs/LIBTORCH_SETUP.md` for the full story.

## Cleanup

If you previously populated this directory the broken way (Homebrew copy /
symlink), wipe it and start over:

```bash
rm -rf libtorch
./scripts/setup-libtorch.sh
source .envrc.libtorch
```
