# 🤝 Contributing to Thrust

Thank you for your interest in contributing to Thrust! This document provides guidelines and information for contributors.

---

## 🌟 Ways to Contribute

### 🐛 Report Bugs
Found a bug? Please [open an issue](https://github.com/yourusername/thrust/issues/new) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Rust version, GPU)
- Relevant logs or error messages

### 💡 Suggest Features
Have an idea? We'd love to hear it! Open an issue with:
- Description of the feature
- Use case / motivation
- Proposed API (if applicable)
- Implementation notes (optional)

### 📝 Improve Documentation
Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Create diagrams or visualizations

### 🔧 Write Code
Ready to contribute code? Great! See below for the development workflow.

---

## 🚀 Getting Started

### Prerequisites
- **Rust:** Install via [rustup](https://rustup.rs/) (stable channel)
- **Git:** For version control
- **libtorch:** PyTorch C++ library (will be needed for Phase 1, Week 3)
  ```bash
  # On macOS with Homebrew
  brew install pytorch

  # Or download from PyTorch website
  # https://pytorch.org/get-started/locally/
  ```
- **CUDA Toolkit:** (Optional, for GPU support)
  ```bash
  # On Ubuntu
  sudo apt install nvidia-cuda-toolkit
  ```

### Setup
1. **Fork the repository**
   ```bash
   # Via GitHub UI, then:
   git clone https://github.com/YOUR_USERNAME/thrust.git
   cd thrust
   ```

2. **Install dependencies**
   ```bash
   cargo build
   ```

3. **Run tests**
   ```bash
   cargo test
   ```

4. **Check code style**
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   ```

---

## 🔄 Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `perf/` - Performance improvements
- `refactor/` - Code refactoring

### 2. Make Changes
- Write clear, self-documenting code
- Follow Rust conventions and idioms
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes
```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture

# Run benchmarks (when available)
cargo bench
```

### 4. Format and Lint
```bash
# Format code
cargo fmt

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Check documentation
cargo doc --no-deps --open
```

### 5. Commit Changes
Write clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(env): add CartPole environment"
git commit -m "fix(buffer): resolve memory leak in rollout storage"
git commit -m "docs(readme): add installation instructions"
git commit -m "perf(train): optimize minibatch sampling"
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `perf` - Performance improvement
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance tasks

### 6. Push and Create PR
```bash
git push origin your-branch-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Link to related issues
- Screenshots/videos (for UI changes)
- Checklist of changes
- Notes for reviewers

---

## 📋 Pull Request Checklist

Before submitting, ensure:
- [ ] Code compiles without errors
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] Changes are focused and minimal

---

## 🎨 Code Style Guidelines

### General Rust Style
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for consistent formatting
- Prefer explicit types in public APIs
- Use descriptive variable names
- Add comments for complex logic

### Documentation
```rust
/// Brief description of function
///
/// More detailed explanation if needed. Can include examples:
///
/// # Examples
///
/// ```
/// use thrust_rl::prelude::*;
/// let env = CartPoleEnv::new();
/// ```
///
/// # Errors
///
/// Returns `Err` if...
///
/// # Panics
///
/// Panics if...
pub fn example_function() -> Result<()> {
    // Implementation
}
```

### Error Handling
- Use `Result<T, E>` for recoverable errors
- Use `anyhow::Result` for application code
- Use `thiserror` for library errors
- Document error conditions

### Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_something() {
        let result = function_under_test();
        assert_eq!(result, expected_value);
    }

    #[test]
    #[should_panic(expected = "specific error message")]
    fn test_panic_case() {
        // Test code that should panic
    }
}
```

---

## 🏗️ Project Structure

```
thrust/
├── src/
│   ├── lib.rs           # Main library entry
│   ├── env/             # Environment implementations
│   │   ├── mod.rs       # Environment trait
│   │   ├── cartpole.rs  # CartPole environment
│   │   └── snake.rs     # Snake environment
│   ├── policy/          # Neural network policies
│   │   ├── mod.rs       # Policy trait
│   │   └── mlp.rs       # MLP policy
│   ├── buffer/          # Experience buffers
│   │   ├── mod.rs       # Buffer trait
│   │   └── rollout.rs   # Rollout buffer
│   ├── train/           # Training algorithms
│   │   ├── mod.rs       # Trainer trait
│   │   └── ppo.rs       # PPO implementation
│   └── utils/           # Utilities
│       ├── mod.rs
│       └── logging.rs
├── examples/            # Example programs
├── benches/            # Benchmarks
├── tests/              # Integration tests
└── docs/               # Additional documentation
```

---

## 🎯 Current Priorities

See [WORKPLAN.md](WORKPLAN.md) for the complete roadmap. We're currently in **Phase 1: Foundation**.

### High Priority (Phase 1)
- [ ] CartPole environment implementation
- [ ] Experience buffer with CPU backend
- [ ] PPO training loop
- [ ] Basic testing framework

### Help Wanted
Check issues labeled `good-first-issue` or `help-wanted` for specific tasks.

---

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Cover edge cases and error conditions
- Keep tests fast (<1s each)

### Integration Tests
- Test complete workflows
- Use real environments and policies
- Verify end-to-end functionality
- Can be slower but should complete in <30s

### Benchmarks
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("function_name", |b| {
        b.iter(|| {
            // Code to benchmark
            black_box(expensive_function())
        })
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

---

## 🐛 Debugging Tips

### Enable Logging
```bash
RUST_LOG=debug cargo run
RUST_LOG=thrust_rl=trace cargo run
```

### Use rust-gdb or rust-lldb
```bash
rust-gdb target/debug/thrust-rl
rust-lldb target/debug/thrust-rl
```

### Profiling
```bash
# CPU profiling
cargo flamegraph --bin your-binary

# Memory profiling
cargo valgrind --bin your-binary
```

---

## 📚 Resources

### Rust Learning
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Rustlings](https://github.com/rust-lang/rustlings) - Interactive exercises

### RL Background
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Sutton & Barto Book](http://incompleteideas.net/book/the-book.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

### Related Projects
- [PufferLib](https://github.com/PufferAI/PufferLib) - Python RL library
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - PyTorch bindings
- [Border](https://github.com/laboroai/border) - Rust RL library

---

## 💬 Communication

### GitHub Discussions
Use for:
- General questions
- Feature discussions
- Architecture decisions
- Showcase your projects

### Issues
Use for:
- Bug reports
- Feature requests
- Specific technical problems

### Discord (Coming Soon)
- Real-time chat
- Community support
- Development updates

---

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. Please be:
- **Respectful:** Treat others with respect and consideration
- **Constructive:** Provide helpful, actionable feedback
- **Collaborative:** Work together toward common goals
- **Patient:** Remember that everyone was a beginner once

Unacceptable behavior includes:
- Harassment, discrimination, or personal attacks
- Trolling or deliberate disruption
- Publishing others' private information
- Other conduct inappropriate in a professional setting

Violations may result in temporary or permanent ban from the project.

---

## 🎉 Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Credited in release notes
- Mentioned in blog posts (for major contributions)
- Given credit in relevant documentation

---

## ❓ Questions?

Not sure where to start? Have questions?
- Check existing [issues](https://github.com/yourusername/thrust/issues) and [discussions](https://github.com/yourusername/thrust/discussions)
- Open a new discussion
- Join our Discord (coming soon!)

---

**Thank you for contributing to Thrust! 🚀**

*Built with 🦀 Rust and ❤️ for reinforcement learning*
