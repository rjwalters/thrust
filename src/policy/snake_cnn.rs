//! Burn-backend CNN policy for the Snake environment (phase 4 of the
//! Burn migration, #65).
//!
//! Sibling to [`crate::policy::snake_cnn::SnakeCnnBurnPolicy`] (tch path). The
//! two modules implement the same compact 3-conv + 2-fc topology used by
//! multi-agent Snake gameplay:
//!
//! ```text
//! grid [B, C_in, H, W]
//!   → conv1 (3x3, pad=1, 8 ch)  → ReLU
//!   → conv2 (3x3, pad=1, 16 ch) → ReLU
//!   → conv3 (3x3, pad=1, 16 ch) → ReLU
//!   → flatten → fc_common (-> 64) → ReLU
//!   → policy_head (64 -> 4)   [action logits]
//!   → value_head  (64 -> 1)   [state value]
//! ```
//!
//! Channel layout (matches the tch path):
//!  - 0: Own snake body
//!  - 1: Own snake head
//!  - 2: Other snakes
//!  - 3: Food
//!  - 4: Walls / boundaries
//!
//! The intentionally-small width (8/16/16/64) is the same WASM-friendly
//! recipe the tch network uses. Output layout — `(logits [B, 4],
//! values [B, 1])` — also matches the tch path so the multi-agent
//! trainer can drop in either backend.

use burn::{
    module::Module,
    nn::{
        Linear, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, activation, backend::Backend},
};

use super::mlp::linear_with_init;

/// Compact convolutional Snake policy on Burn.
#[derive(Module, Debug)]
pub struct SnakeCnnBurnPolicy<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    fc_common: Linear<B>,
    fc_policy: Linear<B>,
    fc_value: Linear<B>,
    /// Cached `grid_size * grid_size * 16` so [`Self::forward`] can
    /// reshape the post-conv activations without re-deriving the
    /// flattened length from the runtime tensor dims. Stored as a
    /// plain field so it lands in the `Record` and survives
    /// `Module::load_record`.
    flat_size: usize,
}

impl<B: Backend> SnakeCnnBurnPolicy<B> {
    /// Construct a fresh Snake CNN policy on `device`.
    ///
    /// # Arguments
    /// * `grid_size`      - Spatial extent of the (square) grid.
    /// * `input_channels` - Number of input channels (typically 5).
    /// * `device`         - Burn backend device.
    pub fn new(grid_size: usize, input_channels: usize, device: &B::Device) -> Self {
        // The tch path uses `nn::ConvConfig { padding: 1, ..Default }`
        // (padding=1 with kernel=3 = "same" output spatial dims).
        // Burn's PaddingConfig2d::Explicit([1, 1]) is the direct
        // equivalent.
        let conv1 = Conv2dConfig::new([input_channels, 8], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([8, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv3 = Conv2dConfig::new([16, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let flat_size = 16 * grid_size * grid_size;

        // FC layers — same Kaiming-uniform default as the tch path
        // (`Default::default()` for `nn::LinearConfig`). We re-use
        // `linear_with_init` with Kaiming so the bias is explicitly
        // zeroed and the call sites are uniform across all four
        // policies; the weight init still matches what Burn's stock
        // `LinearConfig::default()` would do.
        let init = burn::nn::Initializer::KaimingUniform {
            gain: 1.0_f64 / 3.0_f64.sqrt(),
            fan_out_only: false,
        };
        let fc_common = linear_with_init::<B>(flat_size, 64, init.clone(), device);
        let fc_policy = linear_with_init::<B>(64, 4, init.clone(), device);
        let fc_value = linear_with_init::<B>(64, 1, init, device);

        Self { conv1, conv2, conv3, fc_common, fc_policy, fc_value, flat_size }
    }

    /// Forward pass through the network.
    ///
    /// * `grid` shape `[batch, channels, height, width]`.
    /// * Returns `(action_logits [batch, 4], values [batch, 1])` — value
    ///   retains the rank-2 layout the tch path uses (so the multi-agent
    ///   trainer code that calls `.squeeze_dim(-1)` itself stays portable).
    pub fn forward(&self, grid: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = activation::relu(self.conv1.forward(grid));
        let x = activation::relu(self.conv2.forward(x));
        let x = activation::relu(self.conv3.forward(x));

        // Flatten: [B, C, H, W] -> [B, C*H*W]
        let batch = x.dims()[0];
        let x_flat: Tensor<B, 2> = x.reshape([batch, self.flat_size]);

        let features = activation::relu(self.fc_common.forward(x_flat));
        let logits = self.fc_policy.forward(features.clone());
        let values = self.fc_value.forward(features);
        (logits, values)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_snake_cnn_creation() {
        let device = Default::default();
        let _policy = SnakeCnnBurnPolicy::<B>::new(20, 5, &device);
    }

    #[test]
    fn test_snake_cnn_forward_shape() {
        let device = Default::default();
        let policy = SnakeCnnBurnPolicy::<B>::new(20, 5, &device);
        let grid = Tensor::<B, 4>::zeros([1, 5, 20, 20], &device);
        let (logits, values) = policy.forward(grid);
        assert_eq!(logits.dims(), [1, 4]);
        assert_eq!(values.dims(), [1, 1]);
    }

    #[test]
    fn test_snake_cnn_batch_forward_shape() {
        let device = Default::default();
        let policy = SnakeCnnBurnPolicy::<B>::new(10, 5, &device);
        let grid = Tensor::<B, 4>::zeros([4, 5, 10, 10], &device);
        let (logits, values) = policy.forward(grid);
        assert_eq!(logits.dims(), [4, 4]);
        assert_eq!(values.dims(), [4, 1]);
    }
}
