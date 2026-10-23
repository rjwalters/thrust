# Training Scripts

After phase 5 of the Burn migration (#82), the libtorch-flavoured training
scripts have been removed. Training now runs as a direct `cargo run`
against the Burn backend; no external system libraries are required.

## Quick start (CPU)

```bash
cargo run --release --example train_simple_bandit
```

## GPU backends

Burn ships several GPU backends behind feature flags. They compose with
the default `training` feature:

```bash
# wgpu (cross-platform: Vulkan / Metal / DX12 / WebGPU)
cargo run --release --features "training,wgpu" --example train_simple_bandit

# CUDA (Linux + NVIDIA)
cargo run --release --features "training,cuda" --example train_simple_bandit
```

## Security audit

Install the pinned `cargo-audit` and `cargo-deny` versions in
`.github/workflows/ci.yml`, then run `bash scripts/security-audit.sh`.
The required audit covers all exposed Thrust features, all targets, and normal,
build, and development dependencies. `deny.toml` has no package exclusions or
advisory ignores. Vulnerability and unsoundness findings fail; unmaintained
package notices remain visible warnings. Configuration, resolution, and database
fetch errors also fail.

The script retains the generated lockfile, the full-lock `cargo-audit` report and
exit code, and the supported-build report under `target/security-audit/`. CI
uploads these artifacts even when the audit fails. The full lock includes unused
upstream Candle/Tch backends with ZIP findings tracked in
[#356](https://github.com/rjwalters/thrust/issues/356); those dependencies are not
claimed patched. This audit covers configurations exposed by this project, not
additional features that a downstream application might enable through another
Burn dependency.

`python3 scripts/test_security_audit.py` runs the same entrypoint and configuration
against vulnerable optional, WebAssembly-only, build, and development dependency
fixtures, and verifies that configuration and fetch failures remain fatal. These
checks fetch advisory data and require the same installed audit tools.
