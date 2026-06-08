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
//! use tch::Device;
//! use thrust_rl::utils::cuda::ensure_cuda_if_expected;
//!
//! let device = Device::cuda_if_available();
//! ensure_cuda_if_expected(device); // exits(2) if THRUST_EXPECT_CUDA=1 and device == Cpu
//! //
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

/// Backend-generic default device helper for the Burn path.
///
/// Burn's device type is associated with the backend (`B::Device`); each
/// backend implements `Default` for it, so for CPU (`NdArray`) this returns
/// the singleton CPU device, for `Wgpu` the default GPU adapter, and so on.
/// Trainer code that wants to stay backend-agnostic should call this
/// instead of constructing a device manually.
///
/// Parallels the tch-path convention of using
/// [`tch::Device::cuda_if_available`] (or `Device::Cpu`); a unified helper
/// across both backends is deferred to phase 6 of the Burn migration
/// (#65).
///
/// # Example
/// ```rust,no_run
/// # #[cfg(feature = "training-burn")]
/// # {
/// use burn::backend::NdArray;
/// use thrust_rl::utils::cuda::default_burn_device;
///
/// let device = default_burn_device::<NdArray<f32>>();
/// # let _ = device;
/// # }
/// ```
#[cfg(feature = "training-burn")]
pub fn default_burn_device<B: burn::tensor::backend::Backend>()
-> <B as burn::tensor::backend::BackendTypes>::Device {
    <<B as burn::tensor::backend::BackendTypes>::Device as Default>::default()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "training-burn")]
    #[test]
    fn test_default_burn_device_ndarray_smoke() {
        // Sanity-check: the helper compiles, returns a device, and the
        // returned device can actually allocate a tensor (i.e. it isn't
        // some uninitialized placeholder).
        use burn::{
            backend::NdArray,
            tensor::{Tensor, TensorData},
        };

        type B = NdArray<f32>;
        let device = super::default_burn_device::<B>();
        let t: Tensor<B, 1> = Tensor::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), &device);
        assert_eq!(t.dims(), [3]);
        let data: Vec<f32> = t.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
