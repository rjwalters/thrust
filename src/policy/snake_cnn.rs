//! CNN policy network for Snake environment
//!
//! A compact convolutional network designed for multi-agent Snake gameplay.
//! Uses spatial convolutions to process the grid and make decisions.

use tch::{Device, Tensor, nn, nn::Module};

/// CNN policy network for Snake
#[derive(Debug)]
pub struct SnakeCNN {
    /// Feature extraction convolutions
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,

    /// Policy head
    fc_policy: nn::Linear,

    /// Value head
    fc_value: nn::Linear,

    /// Common feature layer
    fc_common: nn::Linear,
}

impl SnakeCNN {
    /// Create a new Snake CNN policy
    ///
    /// # Arguments
    /// * `vs` - Variable store for parameters
    /// * `grid_size` - Size of the grid (assumes square)
    /// * `input_channels` - Number of input channels
    ///   - Channel 0: Own snake body
    ///   - Channel 1: Own snake head
    ///   - Channel 2: Other snakes
    ///   - Channel 3: Food
    ///   - Channel 4: Walls/boundaries
    pub fn new(vs: &nn::Path, grid_size: i64, input_channels: i64) -> Self {
        // Convolutional layers - ULTRA-COMPACT for fast WASM inference!
        let conv1 = nn::conv2d(
            vs / "conv1",
            input_channels,
            8,  // Reduced from 16 (4x smaller for WASM)
            3,
            nn::ConvConfig { padding: 1, ..Default::default() },
        );

        let conv2 = nn::conv2d(
            vs / "conv2",
            8,  // Reduced from 16
            16,  // Reduced from 32 (4x smaller for WASM)
            3,
            nn::ConvConfig { padding: 1, ..Default::default() },
        );

        let conv3 = nn::conv2d(
            vs / "conv3",
            16,  // Reduced from 32
            16,  // Reduced from 32 (4x smaller for WASM)
            3,
            nn::ConvConfig { padding: 1, ..Default::default() },
        );

        // Calculate flattened size after convolutions
        // With padding=1, size stays the same after each conv
        let flat_size = 16 * grid_size * grid_size;  // Updated for new conv3 size

        // Common feature layer - ULTRA-COMPACT for fast WASM inference!
        let fc_common = nn::linear(vs / "fc_common", flat_size, 64, Default::default());  // Reduced from 128 (4x smaller for WASM)

        // Policy head (outputs 4 actions)
        let fc_policy = nn::linear(vs / "policy", 64, 4, Default::default());  // Updated for new fc_common size

        // Value head (outputs single value)
        let fc_value = nn::linear(vs / "value", 64, 1, Default::default());  // Updated for new fc_common size

        Self { conv1, conv2, conv3, fc_common, fc_policy, fc_value }
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `grid` - Input grid tensor [batch, channels, height, width]
    ///
    /// # Returns
    /// * `(action_logits, values)` - Policy logits and value estimates
    pub fn forward(&self, grid: &Tensor) -> (Tensor, Tensor) {
        // Convolutional feature extraction
        let x = grid
            .apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            .apply(&self.conv3)
            .relu();

        // Flatten
        let batch_size = x.size()[0];
        let x = x.view([batch_size, -1]);

        // Common features
        let features = x.apply(&self.fc_common).relu();

        // Policy and value heads
        let action_logits = features.apply(&self.fc_policy);
        let values = features.apply(&self.fc_value);

        (action_logits, values)
    }

    /// Get action probabilities and value from observations
    pub fn forward_policy(&self, grid: &Tensor) -> (Tensor, Tensor) {
        let (logits, values) = self.forward(grid);
        let probs = logits.softmax(-1, tch::Kind::Float);
        (probs, values)
    }

    /// Sample action from policy
    pub fn sample_action(&self, grid: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (logits, values) = self.forward(grid);
        let probs = logits.softmax(-1, tch::Kind::Float);
        let action = probs.multinomial(1, true);
        let log_prob = logits.log_softmax(-1, tch::Kind::Float).gather(1, &action, false);
        (action, log_prob, values)
    }

    /// Get deterministic action (for evaluation)
    pub fn get_action(&self, grid: &Tensor) -> Tensor {
        let (logits, _) = self.forward(grid);
        logits.argmax(-1, false)
    }

    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> i64 {
        // This is an approximation
        // Conv1: 32 * input_channels * 3 * 3 + 32
        // Conv2: 64 * 32 * 3 * 3 + 64
        // Conv3: 64 * 64 * 3 * 3 + 64
        // FC common: 256 * flat_size + 256
        // FC policy: 4 * 256 + 4
        // FC value: 1 * 256 + 1
        // Total depends on grid_size, but for 20x20 with 5 channels:
        // ~6-8k parameters
        0 // TODO: implement proper parameter counting
    }

    /// Export model weights for WASM inference
    ///
    /// Extracts all weights and biases from the PyTorch model and converts them
    /// to a pure Rust format that can be used in WebAssembly.
    pub fn export_for_inference(
        &self,
        grid_width: usize,
        grid_height: usize,
    ) -> crate::policy::inference::SnakeCNNInference {
        use tch::{Device, Kind, Tensor};

        // Helper function to convert a 4D tensor to Vec<Vec<Vec<Vec<f32>>>>
        fn tensor_to_4d(tensor: &Tensor) -> Vec<Vec<Vec<Vec<f32>>>> {
            let size = tensor.size();
            assert_eq!(size.len(), 4, "Expected 4D tensor");
            let d0 = size[0] as usize;
            let d1 = size[1] as usize;
            let d2 = size[2] as usize;
            let d3 = size[3] as usize;

            let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            let flat: Vec<f32> = Vec::try_from(&cpu_tensor).unwrap();

            let mut result = Vec::with_capacity(d0);
            for i0 in 0..d0 {
                let mut layer1 = Vec::with_capacity(d1);
                for i1 in 0..d1 {
                    let mut layer2 = Vec::with_capacity(d2);
                    for i2 in 0..d2 {
                        let start = ((i0 * d1 + i1) * d2 + i2) * d3;
                        layer2.push(flat[start..start + d3].to_vec());
                    }
                    layer1.push(layer2);
                }
                result.push(layer1);
            }
            result
        }

        // Helper function to convert a 2D tensor to Vec<Vec<f32>>
        fn tensor_to_2d(tensor: &Tensor) -> Vec<Vec<f32>> {
            let size = tensor.size();
            assert_eq!(size.len(), 2, "Expected 2D tensor");
            let rows = size[0] as usize;
            let cols = size[1] as usize;

            let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            let flat: Vec<f32> = Vec::try_from(&cpu_tensor).unwrap();

            let mut result = Vec::with_capacity(rows);
            for i in 0..rows {
                result.push(flat[i * cols..(i + 1) * cols].to_vec());
            }
            result
        }

        // Helper function to convert a 1D tensor to Vec<f32>
        fn tensor_to_1d(tensor: &Tensor) -> Vec<f32> {
            let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            Vec::try_from(&cpu_tensor).unwrap()
        }

        // Get layer references (we need to extract weights from the modules)
        // For now, create a dummy structure - we'll need to properly extract from
        // VarStore
        let input_channels = 5; // Fixed for Snake
        let num_actions = 4; // Fixed for Snake (4 directions)

        // Note: This is a placeholder. In a real implementation, you'd need to:
        // 1. Store a VarStore reference in SnakeCNN
        // 2. Extract weights from it similar to MlpPolicy::export_for_inference

        crate::policy::inference::SnakeCNNInference {
            grid_width,
            grid_height,
            input_channels,
            num_actions,
            conv1_weight: vec![vec![vec![vec![0.0; 3]; 3]; input_channels]; 32],
            conv1_bias: vec![0.0; 32],
            conv2_weight: vec![vec![vec![vec![0.0; 3]; 3]; 32]; 64],
            conv2_bias: vec![0.0; 64],
            conv3_weight: vec![vec![vec![vec![0.0; 3]; 3]; 64]; 64],
            conv3_bias: vec![0.0; 64],
            fc_common_weight: vec![vec![0.0; 64 * grid_width * grid_height]; 256],
            fc_common_bias: vec![0.0; 256],
            fc_policy_weight: vec![vec![0.0; 256]; num_actions],
            fc_policy_bias: vec![0.0; num_actions],
            fc_value_weight: vec![vec![0.0; 256]; 1],
            fc_value_bias: vec![0.0; 1],
            metadata: None, // Placeholder export, no metadata
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_cnn_creation() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = SnakeCNN::new(&vs.root(), 20, 5);

        // Test forward pass
        let grid = Tensor::randn(&[1, 5, 20, 20], tch::kind::FLOAT_CPU);
        let (logits, values) = policy.forward(&grid);

        assert_eq!(logits.size(), vec![1, 4]);
        assert_eq!(values.size(), vec![1, 1]);
    }

    #[test]
    fn test_action_sampling() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = SnakeCNN::new(&vs.root(), 20, 5);

        let grid = Tensor::randn(&[1, 5, 20, 20], tch::kind::FLOAT_CPU);
        let (action, log_prob, value) = policy.sample_action(&grid);

        assert_eq!(action.size(), vec![1, 1]);
        assert_eq!(log_prob.size(), vec![1, 1]);
        assert_eq!(value.size(), vec![1, 1]);

        // Action should be in range [0, 3]
        let action_val = i64::try_from(&action).unwrap();
        assert!(action_val >= 0 && action_val < 4);
    }
}
