//! Backend device helpers for the Burn training stack.
//!
//! After phase 5 of the Burn migration (#82), libtorch's CUDA runtime
//! guard is gone — Burn picks its device from the backend type
//! parameter (`B::Device`) instead of through a runtime probe. This
//! module just exposes the default-device helper trainers use.

/// Backend-generic default device helper for the Burn path.
///
/// Burn's device type is associated with the backend (`B::Device`); each
/// backend implements `Default` for it, so for CPU (`NdArray`) this returns
/// the singleton CPU device, for `Wgpu` the default GPU adapter, and so on.
/// Trainer code that wants to stay backend-agnostic should call this
/// instead of constructing a device manually.
///
/// # Example
/// ```rust,no_run
/// # #[cfg(feature = "training")]
/// # {
/// use burn::backend::NdArray;
/// use thrust_rl::utils::cuda::default_burn_device;
///
/// let device = default_burn_device::<NdArray<f32>>();
/// # let _ = device;
/// # }
/// ```
#[cfg(feature = "training")]
pub fn default_burn_device<B: burn::tensor::backend::Backend>()
-> <B as burn::tensor::backend::BackendTypes>::Device {
    <<B as burn::tensor::backend::BackendTypes>::Device as Default>::default()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "training")]
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
        let t: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(vec![1.0_f32, 2.0, 3.0], [3]), &device);
        assert_eq!(t.dims(), [3]);
        let data: Vec<f32> = t.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
