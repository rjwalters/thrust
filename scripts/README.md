# Training Scripts

## GPU Training Workflow

### Setup (one-time on GPU machine)
```bash
# On rwalters-sandbox-2
cd ~/thrust
git pull
# Ensure venv with PyTorch and Rust are set up
```

### Running Training on GPU

```bash
# SSH to GPU machine
ssh rwalters-sandbox-2

# Pull latest code
cd ~/thrust && git pull

# Run training (this will use GPU automatically)
./scripts/gpu-train.sh train_cartpole_long

# Check status
./scripts/gpu-status.sh

# Build without running
./scripts/gpu-build.sh --release
```

### Local Training (CPU)

```bash
# Run on local Mac with CPU
./scripts/train.sh train_cartpole

# Run tests
./scripts/test.sh

# Run full CI checks
./scripts/check.sh
```

## Available Scripts

### GPU Scripts (run ON the remote machine)
- `gpu-train.sh <example>` - Run training example on GPU
- `gpu-build.sh [--release]` - Build project on GPU
- `gpu-status.sh` - Check GPU usage and training progress

### Local Scripts (run on your local machine)
- `train.sh <example>` - Run training locally
- `test.sh` - Run tests locally
- `check.sh` - Run full CI (fmt, clippy, tests)

### Remote Helper Scripts (deprecated - use git workflow instead)
- `remote-*.sh` - Old scripts that rsync code (prefer git workflow)

## Training Examples

- `train_cartpole` - Short CartPole training (50K steps, 4 envs)
- `train_cartpole_long` - Long GPU training (500K steps, 8 envs) [GPU only]
