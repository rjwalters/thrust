//! Test CUDA availability in tch-rs

use tch::Device;

fn main() {
    println!("🔍 Testing CUDA availability in tch-rs");
    println!();

    // Test cuda_if_available
    let device = Device::cuda_if_available();
    println!("Device::cuda_if_available() = {:?}", device);

    // Try to explicitly create CUDA device
    match Device::Cuda(0).try_set_default() {
        Ok(_) => println!("✅ Successfully set CUDA:0 as default device"),
        Err(e) => println!("❌ Failed to set CUDA:0 as default: {}", e),
    }

    // Check if CUDA is available
    let cuda_available = tch::Cuda::is_available();
    println!("tch::Cuda::is_available() = {}", cuda_available);

    let cuda_count = tch::Cuda::device_count();
    println!("tch::Cuda::device_count() = {}", cuda_count);

    // Try creating a tensor on CUDA
    if cuda_available {
        match tch::Tensor::randn([2, 2], (tch::Kind::Float, Device::Cuda(0))) {
            tensor => {
                println!("✅ Successfully created tensor on CUDA");
                println!("   Tensor device: {:?}", tensor.device());
            }
        }
    }
}
