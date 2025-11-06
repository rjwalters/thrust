# 🚀 Getting Started with Thrust

Welcome to Thrust! This guide will help you get up and running.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Rust 1.75+** - Install via [rustup](https://rustup.rs/)
- **Git** - For cloning the repository
- **libtorch** (coming in Phase 1, Week 3) - PyTorch C++ library
- **CUDA Toolkit** (optional) - For GPU acceleration

## 🏃 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/thrust.git
cd thrust
```

### 2. Build the Project

```bash
cargo build
```

This will download and compile all dependencies. First build may take a few minutes.

### 3. Run Tests

```bash
cargo test
```

### 4. Build Documentation

```bash
cargo doc --open
```

## 📚 Project Structure

```
thrust/
├── src/               # Library source code
│   ├── env/          # Environment implementations
│   ├── policy/       # Neural network policies
│   ├── buffer/       # Experience buffers
│   ├── train/        # Training algorithms
│   └── utils/        # Helper functions
├── examples/         # Example programs (coming soon)
├── benches/          # Performance benchmarks (coming soon)
├── tests/            # Integration tests (coming soon)
└── docs/             # Additional documentation
```

## 🎯 Current Status

**Phase 1: Foundation** - Building core infrastructure

We're currently implementing:
- ✅ Project structure and module layout
- 🔄 CartPole environment
- 🔄 Experience buffers
- 🔄 PPO training loop

See [WORKPLAN.md](../WORKPLAN.md) for the complete roadmap.

## 👥 How to Contribute

1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Check [open issues](https://github.com/yourusername/thrust/issues)
3. Look for issues labeled `good-first-issue`
4. Join discussions in [GitHub Discussions](https://github.com/yourusername/thrust/discussions)

## 📖 Learning Resources

### Rust
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Sutton & Barto](http://incompleteideas.net/book/the-book.html)

### Related Projects
- [PufferLib](https://github.com/PufferAI/PufferLib) - Our inspiration
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - PyTorch bindings

## ❓ FAQ

### When will v1.0 be released?
Target: Q2 2025 (see [WORKPLAN.md](../WORKPLAN.md))

### Can I use Thrust in production?
Not yet - we're in pre-alpha. Follow development and we'll announce when production-ready.

### How can I help?
See [CONTRIBUTING.md](../CONTRIBUTING.md) and check issues labeled `help-wanted`.

### Will there be Python bindings?
Not in v1.0, but it's on the roadmap for the future.

## 🐛 Found a Bug?

Please [open an issue](https://github.com/yourusername/thrust/issues/new/choose) with:
- Description of the problem
- Steps to reproduce
- Your environment (OS, Rust version, GPU)
- Error messages or logs

## 💬 Get Help

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Questions and general discussion
- **Discord:** Coming soon!

---

**Ready to dive in?** Check out [WORKPLAN.md](../WORKPLAN.md) for areas where we need help!

*Built with 🦀 Rust and ❤️ for reinforcement learning*
