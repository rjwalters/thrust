//! Snake CNN inference for WASM
//!
//! Pure Rust implementation of Snake CNN forward pass without PyTorch
//! dependencies

use serde::{Deserialize, Serialize};

/// A serializable CNN model for Snake inference
///
/// This struct contains all the weights and biases needed to run
/// CNN inference without any PyTorch dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnakeCNNInference {
    /// Grid width
    pub grid_width: usize,
    /// Grid height
    pub grid_height: usize,
    /// Number of input channels (should be 5 for Snake)
    pub input_channels: usize,
    /// Number of actions (should be 4 for Snake)
    pub num_actions: usize,

    // Conv layer 1: input_channels -> 32 channels, 3x3 kernel
    pub conv1_weight: Vec<Vec<Vec<Vec<f32>>>>, // [32, input_channels, 3, 3]
    pub conv1_bias: Vec<f32>,                  // [32]

    // Conv layer 2: 32 -> 64 channels, 3x3 kernel
    pub conv2_weight: Vec<Vec<Vec<Vec<f32>>>>, // [64, 32, 3, 3]
    pub conv2_bias: Vec<f32>,                  // [64]

    // Conv layer 3: 64 -> 64 channels, 3x3 kernel
    pub conv3_weight: Vec<Vec<Vec<Vec<f32>>>>, // [64, 64, 3, 3]
    pub conv3_bias: Vec<f32>,                  // [64]

    // FC common: 64*grid_width*grid_height -> 256
    pub fc_common_weight: Vec<Vec<f32>>, // [256, flat_size]
    pub fc_common_bias: Vec<f32>,        // [256]

    // Policy head: 256 -> num_actions
    pub fc_policy_weight: Vec<Vec<f32>>, // [num_actions, 256]
    pub fc_policy_bias: Vec<f32>,        // [num_actions]

    // Value head: 256 -> 1
    pub fc_value_weight: Vec<Vec<f32>>, // [1, 256]
    pub fc_value_bias: Vec<f32>,        // [1]

    /// Optional training metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::policy::inference::TrainingMetadata>,
}

impl SnakeCNNInference {
    /// Apply 2D convolution with padding=1
    fn conv2d(
        &self,
        input: &[Vec<Vec<f32>>],       // [in_channels, height, width]
        weight: &[Vec<Vec<Vec<f32>>>], // [out_channels, in_channels, 3, 3]
        bias: &[f32],
    ) -> Vec<Vec<Vec<f32>>> {
        let out_channels = weight.len();
        let in_channels = input.len();
        let height = input[0].len();
        let width = input[0][0].len();

        let mut output = vec![vec![vec![0.0; width]; height]; out_channels];

        for out_c in 0..out_channels {
            for h in 0..height {
                for w in 0..width {
                    let mut sum = bias[out_c];

                    // 3x3 convolution with padding=1
                    for in_c in 0..in_channels {
                        for kh in 0..3 {
                            for kw in 0..3 {
                                let ih = h as i32 + kh as i32 - 1;
                                let iw = w as i32 + kw as i32 - 1;

                                if ih >= 0 && ih < height as i32 && iw >= 0 && iw < width as i32 {
                                    sum += input[in_c][ih as usize][iw as usize]
                                        * weight[out_c][in_c][kh][kw];
                                }
                            }
                        }
                    }

                    output[out_c][h][w] = sum;
                }
            }
        }

        output
    }

    /// Apply ReLU activation
    fn relu(&self, input: &mut [Vec<Vec<f32>>]) {
        for channel in input.iter_mut() {
            for row in channel.iter_mut() {
                for val in row.iter_mut() {
                    if *val < 0.0 {
                        *val = 0.0;
                    }
                }
            }
        }
    }

    /// Forward pass: compute action logits and value
    ///
    /// # Arguments
    /// * `grid` - Input grid [channels, height, width] flattened as
    ///   [c0_pixels..., c1_pixels..., ...]
    ///
    /// # Returns
    /// * `(logits, value)` - Action logits [num_actions] and state value
    ///   (scalar)
    pub fn forward(&self, grid: &[f32]) -> (Vec<f32>, f32) {
        let grid_size = self.grid_width * self.grid_height;
        assert_eq!(grid.len(), self.input_channels * grid_size);

        // Reshape input to [channels, height, width]
        let mut input =
            vec![vec![vec![0.0; self.grid_width]; self.grid_height]; self.input_channels];
        for c in 0..self.input_channels {
            for h in 0..self.grid_height {
                for w in 0..self.grid_width {
                    let idx = c * grid_size + h * self.grid_width + w;
                    input[c][h][w] = grid[idx];
                }
            }
        }

        // Conv1 + ReLU
        let mut x = self.conv2d(&input, &self.conv1_weight, &self.conv1_bias);
        self.relu(&mut x);

        // Conv2 + ReLU
        x = self.conv2d(&x, &self.conv2_weight, &self.conv2_bias);
        self.relu(&mut x);

        // Conv3 + ReLU
        x = self.conv2d(&x, &self.conv3_weight, &self.conv3_bias);
        self.relu(&mut x);

        // Flatten
        let flat_size = 64 * grid_size;
        let mut flattened = Vec::with_capacity(flat_size);
        for channel in &x {
            for row in channel {
                for &val in row {
                    flattened.push(val);
                }
            }
        }

        // FC common + ReLU
        let mut features = vec![0.0; 256];
        for (i, row) in self.fc_common_weight.iter().enumerate() {
            for (j, &val) in flattened.iter().enumerate() {
                features[i] += row[j] * val;
            }
            features[i] += self.fc_common_bias[i];
            if features[i] < 0.0 {
                features[i] = 0.0;
            }
        }

        // Policy head
        let mut logits = vec![0.0; self.num_actions];
        for (i, row) in self.fc_policy_weight.iter().enumerate() {
            for (j, &val) in features.iter().enumerate() {
                logits[i] += row[j] * val;
            }
            logits[i] += self.fc_policy_bias[i];
        }

        // Value head
        let mut value = 0.0;
        for (j, &val) in features.iter().enumerate() {
            value += self.fc_value_weight[0][j] * val;
        }
        value += self.fc_value_bias[0];

        (logits, value)
    }

    /// Get action from grid using argmax (deterministic)
    pub fn get_action(&self, grid: &[f32]) -> usize {
        let (logits, _value) = self.forward(grid);

        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Save model to JSON file
    pub fn save_json(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load_json(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }
}
