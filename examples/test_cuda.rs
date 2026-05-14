//! Test CUDA availability in tch-rs.
//!
//! Verifies that the `build.rs` link-arg fix is working: on a CUDA-equipped
//! Linux box, this should print `Cuda(0)` and `tch::Cuda::is_available() =
//! true` *without* any `LD_PRELOAD` workaround. On macOS or CPU-only builds it
//! gracefully reports `Cpu`.
//!
//! Acceptance check (from issue #9):
//!
//! ```bash
//! cargo build --release --example test_cuda
//! ldd target/release/examples/test_cuda | grep -E '(torch_cuda|c10_cuda)'
//! # Both lines must appear.
//! ./target/release/examples/test_cuda
//! # Must report Cuda(0).
//! ```
//!
//! Setting `THRUST_EXPECT_CUDA=1` makes this example hard-fail (exit 2)
//! instead of reporting CPU, which is what scripts like
//! `scripts/train-snake-remote.sh` want.

use tch::Device;
use thrust_rl::utils::cuda::ensure_cuda_if_expected;

fn main() {
    println!("Testing CUDA availability in tch-rs");
    println!();

    // Test cuda_if_available
    let device = Device::cuda_if_available();
    println!("Device::cuda_if_available() = {device:?}");

    // Check if CUDA is available
    let cuda_available = tch::Cuda::is_available();
    println!("tch::Cuda::is_available() = {cuda_available}");

    let cuda_count = tch::Cuda::device_count();
    println!("tch::Cuda::device_count() = {cuda_count}");

    // Try creating a tensor on CUDA
    if cuda_available {
        let tensor = tch::Tensor::randn([2, 2], (tch::Kind::Float, Device::Cuda(0)));
        println!("Successfully created tensor on CUDA");
        println!("   Tensor device: {:?}", tensor.device());
    } else {
        println!("CUDA not available, cannot create CUDA tensor");
        println!();
        println!("Hints if you expected CUDA to be available:");
        println!("  - Are you on a Linux box with an NVIDIA GPU?");
        println!("  - Is `nvidia-smi` working?");
        println!("  - Did Python report CUDA? Run:");
        println!("      python3 -c 'import torch; print(torch.cuda.is_available())'");
        println!("  - Did the build.rs detect a CUDA-enabled libtorch?");
        println!("    Check `ldd target/release/examples/test_cuda | grep torch_cuda`.");
        println!("  - If `ldd` shows libtorch_cuda but is_available is still false,");
        println!("    that is a deeper tch/libtorch issue --- see issue #9 history.");
    }

    // The Option D hard-fail guard. Only triggers if THRUST_EXPECT_CUDA is set.
    ensure_cuda_if_expected(device);
}
