# 🚀 Getting Started with Thrust

Welcome to Thrust! This guide will help you get up and running.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Rust nightly** - Install via [rustup](https://rustup.rs/); the
  workspace pins a nightly toolchain in `rust-toolchain.toml`.
- **Git** - For cloning the repository.

That's it. The default `training` feature uses Burn's pure-Rust NdArray
backend, so `cargo build` works out of the box without libtorch, CUDA,
or any other system library. Opt-in GPU backends (wgpu, cuda) are
documented in [BURN_BACKENDS.md](BURN_BACKENDS.md).

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

### 4. Run an Example

```bash
cargo run --release --features training --example train_simple_bandit
```

This is one of twelve runnable trainers. Browse the full
[Example Gallery](EXAMPLES.md) for a trainer per algorithm (PPO, A2C, DQN, SAC,
BC, PSRO, NFSP) across every environment, with copy-paste run commands.

### 5. Follow the Tutorial Series

Examples are reference material; the **[Tutorial Series](tutorials/README.md)**
is a guided, dependency-ordered path from your first agent to a
trained-and-deployed policy. Every code block is CI-doc-tested. Start with
[Your first agent](tutorials/01-your-first-agent.md) (SimpleBandit + actor-critic,
the rollout → loss → update loop by hand), then
[Solving CartPole with PPO](tutorials/02-cartpole-ppo.md).

### 6. Build Documentation

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
├── examples/         # Runnable trainers (see docs/EXAMPLES.md)
├── benches/          # Performance benchmarks
├── tests/            # Integration tests
└── docs/             # Additional documentation
```

## 🎯 Current Status

**v0.3.0 — published on [crates.io](https://crates.io/crates/thrust-rl).**
The training loop is fully shipped: five single-agent algorithms (PPO, A2C,
DQN, SAC, BC) plus recurrent (LSTM) PPO and an async actor-learner, two
multi-agent meta-solvers (PSRO, NFSP), WASM inference, and a
[live browser demo](https://rjwalters.github.io/thrust/). See
[CHANGELOG.md](../CHANGELOG.md) for release history and
[ROADMAP.md](../ROADMAP.md) for what's next.
- 🔄 PPO training loop

See [ROADMAP.md](../ROADMAP.md) for the complete roadmap.

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
- [Burn](https://burn.dev) - The tensor framework Thrust uses

## ❓ FAQ

### When will v1.0 be released?
No date committed. Per the pre-1.0 policy ([CHANGELOG.md](../CHANGELOG.md)),
breaking API changes land in MINOR bumps until the public API stabilizes;
1.0 comes when it stops moving.

### Can I use Thrust in production?
The crate is published and the API is documented and tested, but pre-1.0
semantics apply: MINOR releases may break the API. Pin a version and read the
CHANGELOG when upgrading.

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

**Ready to dive in?** Check out [ROADMAP.md](../ROADMAP.md) and the open issues for areas where we need help!

*Built with 🦀 Rust and ❤️ for reinforcement learning*
