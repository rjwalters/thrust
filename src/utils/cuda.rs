//! CUDA runtime guard utilities.
//!
//! The `build.rs` at the crate root takes care of *link-time* correctness so
//! that `libtorch_cuda.so` ends up in `DT_NEEDED`. This module provides a
//! *runtime* guard that hard-fails if the user expected CUDA but the
//! `tch::Cuda::is_available()` probe still returns false. The intent is to
//! turn silent CPU fallback (which is hard to notice in benchmarks) into a
//! loud, fast failure.
//!
//! ## Usage
//!
//! In training examples:
//!
//! ```rust,no_run
//! # #[cfg(feature = "training")]
//! # fn main() {
//! use thrust_rl::utils::cuda::ensure_cuda_if_expected;
//! use tch::Device;
//!
//! let device = Device::cuda_if_available();
//! ensure_cuda_if_expected(device); // exits(2) if THRUST_EXPECT_CUDA=1 and device == Cpu
//! # }
//! # #[cfg(not(feature = "training"))] fn main() {}
//! ```
//!
//! Set `THRUST_EXPECT_CUDA=1` in the environment to make the guard active.
//! With the variable unset, the guard is a no-op and behavior is unchanged
//! (silent CPU fallback for any user who hasn't opted in).
//!
//! ## Why an opt-in env var rather than always-on?
//!
//! Examples like `train_cartpole` happily run on CPU and we don't want to
//! force every developer to install CUDA. Only training runs that *expect*
//! a GPU should hard-fail, and those runs are always launched via one of
//! the `scripts/*-gpu*.sh` wrappers, where the env var can be set centrally.

#[cfg(feature = "training")]
use tch::Device;

/// Hard-fail if `THRUST_EXPECT_CUDA` is set and `device` is `Cpu`.
///
/// On failure, prints a diagnostic explaining the most common root cause
/// (linker stripped `libtorch_cuda`) and exits with status 2 so that
/// shell scripts can detect the failure (`$? == 2` means "expected CUDA,
/// got CPU"; status 1 stays reserved for ordinary `Result::Err`).
///
/// No-op if:
/// - the `THRUST_EXPECT_CUDA` environment variable is not set, or
/// - `device` is anything other than `Device::Cpu`.
#[cfg(feature = "training")]
pub fn ensure_cuda_if_expected(device: Device) {
    let Ok(val) = std::env::var("THRUST_EXPECT_CUDA") else {
        return;
    };
    if val.is_empty() || val == "0" || val.eq_ignore_ascii_case("false") {
        return;
    }
    if device != Device::Cpu {
        return;
    }

    eprintln!();
    eprintln!("FATAL: THRUST_EXPECT_CUDA={val} but tch fell back to Device::Cpu.");
    eprintln!();
    eprintln!("Most likely cause: the linker stripped libtorch_cuda.so from");
    eprintln!("the binary's NEEDED list because no symbol from it was directly");
    eprintln!("referenced. The crate-level build.rs is supposed to prevent");
    eprintln!("this --- verify with:");
    eprintln!();
    eprintln!("    ldd <this-binary> | grep torch_cuda");
    eprintln!();
    eprintln!("If the line is empty, the build.rs did not run or did not detect");
    eprintln!("CUDA. Make sure LIBTORCH_USE_PYTORCH=1 is set (or LIBTORCH points");
    eprintln!("at a CUDA-enabled libtorch) and re-run `cargo clean && cargo build`.");
    eprintln!();
    eprintln!("As a last resort, set LD_PRELOAD to libtorch_cuda.so + libc10_cuda.so.");
    eprintln!();
    std::process::exit(2);
}

/// Stub for non-training builds so callers can unconditionally invoke the
/// guard without `cfg` gating at every call site.
///
/// This intentionally accepts `_device: &()` (a unit reference) because
/// non-training builds don't have access to `tch::Device`. Most call sites
/// are inside `#[cfg(feature = "training")]` code anyway, so this stub
/// rarely gets used in practice.
#[cfg(not(feature = "training"))]
#[allow(dead_code)]
pub fn ensure_cuda_if_expected(_device: &()) {}
