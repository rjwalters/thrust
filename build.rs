//! Build script for thrust-rl.
//!
//! Primary purpose: defeat the GNU linker's `--as-needed` default behavior so
//! that `libtorch_cuda.so` (and `libc10_cuda.so`) stay in the binary's
//! `DT_NEEDED` list even though the Rust source never references a symbol
//! that resolves to those libraries directly.
//!
//! ## Background
//!
//! Modern Linux distros (Ubuntu 22.04+, Debian 12+, etc.) ship `ld` configured
//! with `--as-needed` enabled by default. Under that flag, any `DT_NEEDED`
//! entry for a shared library from which no symbol is actually referenced is
//! dropped at link time. `torch-sys`'s build script emits
//! `cargo:rustc-link-lib=torch_cuda` correctly when CUDA is detected, but
//! since `tch` itself only directly references symbols in `libtorch_cpu` and
//! `libc10`, the linker silently strips `-ltorch_cuda` from the final binary.
//!
//! At runtime, `libtorch` performs lazy CUDA backend registration that
//! depends on `libtorch_cuda.so` already being loaded into the process. With
//! the `DT_NEEDED` entry stripped, the CUDA backend never initializes and
//! `tch::Cuda::is_available()` returns `false` --- *silent CPU fallback*.
//!
//! ## Why downstream `RUSTFLAGS="-C link-arg=-Wl,--no-as-needed"` is not enough
//!
//! `-Wl,--no-as-needed` is *positional*: it only affects libraries that
//! appear *after* it on the linker command line. When set via
//! `RUSTFLAGS`, the flag is injected at a position that does not cover
//! the `-ltorch_cuda` argument emitted by the `torch-sys` build script.
//! Re-emitting the bracket from a leaf-crate build script puts the flag
//! at the *end* of the link line where it does cover the late-emitted
//! `-ltorch_cuda`.
//!
//! ## What this script does
//!
//! On Linux only, and only when the `training` feature is enabled, it emits:
//!
//! ```text
//! cargo:rustc-link-arg=-Wl,--no-as-needed
//! cargo:rustc-link-lib=torch_cuda
//! cargo:rustc-link-lib=c10_cuda
//! cargo:rustc-link-arg=-Wl,--as-needed
//! ```
//!
//! when CUDA is detected in the libtorch installation. On macOS, Windows, or
//! CPU-only builds, the script is a no-op.
//!
//! ## CUDA detection
//!
//! Detection mirrors `torch-sys`'s logic at a coarse level: if
//! `LIBTORCH_USE_PYTORCH=1` is set, we shell out to Python to ask whether
//! CUDA is compiled in and discover the lib path. Otherwise we look at
//! `LIBTORCH` env var (the standalone libtorch path) and check for the
//! presence of `libtorch_cuda.so`. If neither path yields a CUDA-enabled
//! libtorch, the script emits nothing extra and the build proceeds as
//! CPU-only.
//!
//! ## Opting out
//!
//! Set `THRUST_DISABLE_CUDA_LINK_FIX=1` to skip the link-arg emission. Useful
//! if a downstream user has a custom linker setup that the fix interferes
//! with. The `THRUST_EXPECT_CUDA` runtime check (see `src/utils/cuda.rs`)
//! is independent of this opt-out.

use std::{env, path::PathBuf, process::Command};

fn main() {
    // Re-run conditions: only re-run when the relevant environment variables
    // change. We deliberately do NOT use `cargo:rerun-if-changed=build.rs`
    // alone because Cargo defaults to that. Instead, we add env-var triggers
    // for the inputs we actually consult.
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_BYPASS_VERSION_CHECK");
    println!("cargo:rerun-if-env-changed=THRUST_DISABLE_CUDA_LINK_FIX");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TRAINING");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let training_enabled = env::var("CARGO_FEATURE_TRAINING").is_ok();
    let opted_out = env::var("THRUST_DISABLE_CUDA_LINK_FIX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Only apply the fix on Linux with the training feature on. macOS, iOS,
    // and Windows do not exhibit the `--as-needed` behavior we are countering
    // (Windows uses a totally different link model, macOS's ld doesn't strip
    // unreferenced dylibs the same way).
    if target_os != "linux" {
        return;
    }
    if !training_enabled {
        return;
    }
    if opted_out {
        eprintln!(
            "thrust-rl build.rs: THRUST_DISABLE_CUDA_LINK_FIX is set; skipping \
             CUDA link-arg emission."
        );
        return;
    }

    let cuda_lib_dir = match detect_cuda_lib_dir() {
        Some(dir) => dir,
        None => {
            // CPU-only build, or libtorch without CUDA. Nothing to do.
            eprintln!(
                "thrust-rl build.rs: no CUDA-enabled libtorch detected; \
                 skipping CUDA link-arg emission. \
                 (training feature is on, but libtorch_cuda.so was not found.)"
            );
            return;
        }
    };

    eprintln!(
        "thrust-rl build.rs: emitting positional link args to prevent \
         --as-needed from stripping libtorch_cuda (lib dir: {}).",
        cuda_lib_dir.display()
    );

    // Add a `rustc-link-search` directive so the linker can find the CUDA
    // libs at link time. `torch-sys` typically already adds this; we re-emit
    // it as a belt-and-suspenders guard.
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    // The bracket: enable --no-as-needed, list the libraries we want to keep
    // referenced, then restore --as-needed so we don't accidentally retain
    // unrelated unused libs later on the command line.
    //
    // Position matters: Cargo emits build-script `rustc-link-arg` directives
    // *after* most dependency `rustc-link-lib` directives, so our re-emitted
    // `-Wl,--no-as-needed` covers the `-ltorch_cuda` we re-list here AND any
    // late `-ltorch_cuda` already emitted by `torch-sys`'s own build script.
    //
    // We use the `-examples` / `-tests` / `-benches` variants of
    // `rustc-link-arg` so the flags apply to binary artifacts that link this
    // crate (not the rlib itself, which is a static archive and doesn't go
    // through the linker). `rustc-link-lib` applies to every artifact and
    // doesn't need a variant. We deliberately omit `-bins`: `thrust-rl` is
    // library-only (no `[[bin]]` target), and emitting `rustc-link-arg-bins`
    // when there are no bins triggers `error: invalid instruction` on stable
    // Cargo.
    for variant in ["examples", "tests", "benches"] {
        println!("cargo:rustc-link-arg-{variant}=-Wl,--no-as-needed");
    }
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=c10_cuda");
    for variant in ["examples", "tests", "benches"] {
        println!("cargo:rustc-link-arg-{variant}=-Wl,--as-needed");
    }

    // Belt-and-suspenders: also embed an RPATH entry so the binary can find
    // the CUDA libs at runtime without relying on `LD_LIBRARY_PATH`. We
    // use `-Wl,-rpath,...` rather than passing `-rpath` directly so the
    // flag survives gcc/clang's argument forwarding (rustc invokes the
    // system linker via the C compiler driver).
    for variant in ["examples", "tests", "benches"] {
        println!("cargo:rustc-link-arg-{variant}=-Wl,-rpath,{}", cuda_lib_dir.display());
    }
}

/// Locate the directory containing `libtorch_cuda.so` and `libc10_cuda.so`,
/// or `None` if we can't find them (CPU-only or non-Linux libtorch).
fn detect_cuda_lib_dir() -> Option<PathBuf> {
    // Strategy 1: pip-installed PyTorch via LIBTORCH_USE_PYTORCH=1.
    if env::var("LIBTORCH_USE_PYTORCH").as_deref() == Ok("1") {
        if let Some(dir) = pytorch_lib_dir_from_python() {
            if has_cuda_libs(&dir) {
                return Some(dir);
            }
        }
    }

    // Strategy 2: explicit LIBTORCH path.
    if let Ok(libtorch) = env::var("LIBTORCH") {
        let lib_dir = PathBuf::from(libtorch).join("lib");
        if has_cuda_libs(&lib_dir) {
            return Some(lib_dir);
        }
    }

    // Strategy 3: fallback to invoking python3 unconditionally. This catches
    // setups where the user has pip-installed torch but forgot to set
    // LIBTORCH_USE_PYTORCH=1 explicitly --- if we can find a CUDA-enabled
    // torch we may as well emit the link args (they're a no-op without the
    // matching torch-sys link directives anyway).
    if let Some(dir) = pytorch_lib_dir_from_python() {
        if has_cuda_libs(&dir) {
            return Some(dir);
        }
    }

    None
}

fn pytorch_lib_dir_from_python() -> Option<PathBuf> {
    // Try python3 first, then python. Quiet errors --- we'll just fail
    // detection and let the caller handle the None case.
    for py in ["python3", "python"] {
        let out = Command::new(py)
            .arg("-c")
            .arg(
                "import os, torch; \
                 print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
            )
            .output();
        if let Ok(out) = out {
            if out.status.success() {
                let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }
    None
}

fn has_cuda_libs(dir: &PathBuf) -> bool {
    dir.join("libtorch_cuda.so").exists() && dir.join("libc10_cuda.so").exists()
}
