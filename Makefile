.PHONY: help check fmt clippy test test-all doc clean build build-release install

# Default target
help:
	@echo "🚀 Thrust Development Commands"
	@echo ""
	@echo "make check        - Quick compile check"
	@echo "make fmt          - Format code with rustfmt"
	@echo "make clippy       - Run clippy lints"
	@echo "make test         - Run all tests"
	@echo "make test-all     - Run tests with all features"
	@echo "make doc          - Build and open documentation"
	@echo "make ci           - Run all CI checks locally"
	@echo "make build        - Build debug version"
	@echo "make build-release - Build optimized release version"
	@echo "make clean        - Clean build artifacts"
	@echo ""

# Quick compile check
check:
	@echo "🔍 Checking code..."
	cargo check --all-targets --all-features

# Format code
fmt:
	@echo "🎨 Formatting code..."
	cargo fmt --all

# Check formatting
fmt-check:
	@echo "🎨 Checking code formatting..."
	cargo fmt --all -- --check

# Run clippy
clippy:
	@echo "📎 Running clippy..."
	cargo clippy --all-targets --all-features -- -D warnings

# Run tests
test:
	@echo "🧪 Running tests..."
	cargo test --all-features

# Run all tests including doc tests
test-all:
	@echo "🧪 Running all tests..."
	cargo test --all-features
	cargo test --doc --all-features

# Build and open documentation
doc:
	@echo "📚 Building documentation..."
	cargo doc --no-deps --all-features --open

# Build documentation without opening (for CI)
doc-ci:
	@echo "📚 Building documentation..."
	cargo doc --no-deps --all-features

# Build debug version
build:
	@echo "🔨 Building debug version..."
	cargo build

# Build release version
build-release:
	@echo "🚀 Building release version..."
	cargo build --release

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean

# Run all CI checks locally
ci: fmt-check clippy test-all doc-ci
	@echo "✅ All CI checks passed!"

# Install from source
install:
	@echo "📦 Installing thrust-rl..."
	cargo install --path .
