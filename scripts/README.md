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
