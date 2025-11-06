# tch-rs Integration Status

## Current Status: Pending

The MlpPolicy implementation is complete but requires tch-rs to be enabled.

## Version Compatibility Issue

We're currently blocked by a version incompatibility:

| Component | Version | Status |
|-----------|---------|--------|
| Homebrew PyTorch | 2.9.0 | ✅ Installed |
| tch-rs 0.22.0 | Supports PyTorch 2.9 | ❌ Requires Rust edition 2024 (unstable) |
| tch-rs 0.16/0.17 | Compatible with current Rust | ❌ Only supports PyTorch 2.1.x |
| LibTorch 2.1.x ARM64 | Would work with tch-rs 0.16 | ❌ No official pre-built binaries |

## Resolution Paths

### Option 1: Wait for Rust 1.86+ (Recommended)

**Timeline**: Rust 1.86 (early 2026) will stabilize edition 2024

**Steps**:
1. Wait for Rust 1.86 release
2. Update `Cargo.toml` edition to 2024
3. Use tch-rs 0.22.0 with Homebrew PyTorch 2.9
4. Uncomment full MLP implementation

**Pros**: Clean, official solution
**Cons**: Need to wait ~2-3 months

### Option 2: Build LibTorch 2.1.2 from Source

**Steps**:
```bash
# Clone PyTorch 2.1.2
git clone -b v2.1.2 --recurse-submodule https://github.com/pytorch/pytorch.git
cd pytorch

# Build libtorch
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS:BOOL=ON \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DPYTHON_EXECUTABLE:PATH=`which python3` \
      -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install \
      ..
cmake --build . --target install

# Set environment variable
export LIBTORCH=/path/to/pytorch-install
```

**Pros**: Works immediately
**Cons**: Long build time (1-2 hours), complex setup

### Option 3: Use Nightly Rust

**Steps**:
```bash
# Install nightly
rustup install nightly

# Build with nightly
cargo +nightly build
```

**Pros**: Can use tch-rs 0.22.0 now
**Cons**: Nightly is unstable, not recommended for production

### Option 4: Use Python Interop (Alternative Approach)

Instead of tch-rs, use PyO3 to call PyTorch from Rust:

**Pros**: Works today, well-supported
**Cons**: Different API, requires Python runtime

## Current Workaround

For now, we have:

1. **Placeholder MlpPolicy** in `src/policy/mlp.rs`
   - Struct defined with proper API
   - Full implementation in comments
   - Ready to uncomment when tch-rs is available

2. **Complete documentation** of the neural network architecture
   - Policy head (action logits)
   - Value head (state values)
   - Shared feature extractor

3. **Test structure** ready for when tch-rs is enabled

## What's Implemented

Even without tch-rs, we have working implementations of:

- ✅ CartPole environment (pure Rust physics)
- ✅ EnvPool (Rayon-based parallelism)
- ✅ RolloutBuffer (GAE computation)
- ✅ Policy structure (ready for tch-rs)

The missing piece is just the neural network forward/backward passes.

## Recommendation

**For development**: Continue building the PPO trainer with placeholder policy

**For production**: Wait for Rust 1.86 (early 2026) and use tch-rs 0.22.0

**For immediate use**: Build LibTorch 2.1.2 from source (Option 2)

## Timeline

- **Now**: All infrastructure except neural networks works
- **Q1 2026**: Rust 1.86 releases with edition 2024
- **Q1 2026**: Full tch-rs integration with Homebrew PyTorch

## Testing Without Neural Networks

You can still test the RL system using:

1. **Random policy**: Sample actions uniformly
2. **Rule-based policy**: Hand-coded heuristics for CartPole
3. **Tabular methods**: Q-learning with discretized state space

Example random policy:
```rust
use rand::Rng;

pub struct RandomPolicy;

impl RandomPolicy {
    pub fn get_action(&self, _obs: &[f32], num_actions: usize) -> i64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(0..num_actions as i64)
    }
}
```

This lets you test the entire training loop mechanics without neural networks.

---

**Last Updated**: 2025-11-05
**Status**: Waiting for Rust edition 2024 stabilization
