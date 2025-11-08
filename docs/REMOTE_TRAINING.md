# Remote Training Guide

## Overview

This project uses remote GPU machines for training. **DO NOT sync code from your local machine.** Instead, work directly on the remote machines by SSH'ing in and running scripts there.

## Available Remote Machines

- `rwalters-sandbox-2` - GPU machine (NVIDIA L4) for training
- `rwalters-sandbox-1` - CPU machine for testing

## Setup on Remote Machine

1. SSH into the remote machine:
   ```bash
   ssh rwalters-sandbox-2
   ```

2. Clone or pull the latest code:
   ```bash
   cd ~/thrust
   git pull
   # or if first time: git clone https://github.com/rjwalters/thrust.git ~/thrust
   ```

3. The remote machine should already have:
   - PyTorch installed (`pip3 list | grep torch`)
   - Rust/Cargo installed (`cargo --version`)
   - CUDA available (`nvidia-smi`)

## Training Workflow

### On Remote Machine (rwalters-sandbox-2)

```bash
# 1. SSH into remote
ssh rwalters-sandbox-2

# 2. Navigate to project
cd ~/thrust

# 3. Pull latest changes
git pull

# 4. Run training using gpu-train.sh script
./scripts/gpu-train.sh train_snake_multi_v2

# Training will log to: training_train_snake_multi_v2_TIMESTAMP.log
```

### Monitoring Training

While still SSH'd into the remote:

```bash
# Check GPU usage
nvidia-smi

# Watch training logs
tail -f training_*.log

# Check if training is running
ps aux | grep cargo
```

### After Training

Export the model:

```bash
# Still on remote machine
LIBTORCH_USE_PYTORCH=1 cargo run --example export_snake_model --release

# Download model to local machine (from your local machine)
scp rwalters-sandbox-2:~/thrust/snake_model.json ./web/public/
```

## Important Notes

- **Always work on the remote machine** - Don't rsync from local to remote
- Remote machines pull from git, just like your local machine
- Training runs directly on the remote using `./scripts/gpu-train.sh`
- The remote machine has all dependencies pre-installed
- Use `LIBTORCH_USE_PYTORCH=1` to use Python PyTorch installation

## Common Issues

### CUDA Not Detected
Make sure you're using the Python PyTorch:
```bash
export LIBTORCH_USE_PYTORCH=1
```

### Wrong Directory
The project should be at `~/thrust` on the remote, not `~/GitHub/thrust`

### Missing Dependencies
PyTorch is installed globally with:
```bash
pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
