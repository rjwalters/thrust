//! Snake CNN inference for WASM
//!
//! Pure Rust implementation of Snake CNN forward pass. Stays off the Burn
//! tensor stack used on the training side so the WASM bundle stays small;
//! see `crate::inference` module docs and `WASM_ROADMAP.md`.

use serde::{Deserialize, Serialize};

/// A serializable CNN model for Snake inference
///
/// This struct contains all the weights and biases needed to run
/// CNN inference in pure Rust (no Burn/tensor-stack dependency).
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

    /// Conv1 kernel weights, shape `[out=32, in=input_channels, kh=3, kw=3]`.
    /// Applied first in [`SnakeCNNInference::forward`] with `padding=1`.
    pub conv1_weight: Vec<Vec<Vec<Vec<f32>>>>,
    /// Conv1 per-output-channel bias, shape `[32]`.
    pub conv1_bias: Vec<f32>,

    /// Conv2 kernel weights, shape `[out=64, in=32, kh=3, kw=3]`. Applied
    /// after Conv1 + ReLU with `padding=1`.
    pub conv2_weight: Vec<Vec<Vec<Vec<f32>>>>,
    /// Conv2 per-output-channel bias, shape `[64]`.
    pub conv2_bias: Vec<f32>,

    /// Conv3 kernel weights, shape `[out=64, in=64, kh=3, kw=3]`. Applied
    /// after Conv2 + ReLU with `padding=1`.
    pub conv3_weight: Vec<Vec<Vec<Vec<f32>>>>,
    /// Conv3 per-output-channel bias, shape `[64]`.
    pub conv3_bias: Vec<f32>,

    /// Shared fully-connected weights, shape
    /// `[256, 64 * grid_width * grid_height]`, applied to the flattened
    /// Conv3 output.
    pub fc_common_weight: Vec<Vec<f32>>,
    /// Shared fully-connected bias, shape `[256]`.
    pub fc_common_bias: Vec<f32>,

    /// Policy head weights, shape `[num_actions, 256]`. Output is raw logits
    /// (no softmax applied here — callers do that in
    /// [`SnakeCNNInference::get_action`] or externally).
    pub fc_policy_weight: Vec<Vec<f32>>,
    /// Policy head bias, shape `[num_actions]`.
    pub fc_policy_bias: Vec<f32>,

    /// Value head weights, shape `[1, 256]`. Produces a scalar state-value
    /// estimate (no activation).
    pub fc_value_weight: Vec<Vec<f32>>,
    /// Value head bias, shape `[1]`.
    pub fc_value_bias: Vec<f32>,

    /// Optional training metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<crate::policy::inference::TrainingMetadata>,
}

impl SnakeCNNInference {
    /// Apply 2D convolution with padding=1
    // Indexed loops mirror the [out_c][h][w][in_c][kh][kw] tensor layout and the
    // padded neighbor arithmetic; rewriting as enumerate() iterators would obscure
    // the spatial indexing rather than clarify it.
    #[allow(clippy::needless_range_loop)]
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
    /// * `(logits, value)` - Action logits `[num_actions]` and state value
    ///   (scalar)
    // The [c][h][w] reshape loop computes a flat index from three counters;
    // enumerate() cannot express that arithmetic cleanly.
    #[allow(clippy::needless_range_loop)]
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

        // Global average pooling: collapse each channel to its spatial mean.
        // Mirrors the training model's adaptive_avg_pool2d([1,1]).
        let num_conv3_out = self.conv3_weight.len();
        let spatial_size = (self.grid_height * self.grid_width) as f32;
        let mut flattened = Vec::with_capacity(num_conv3_out);
        for channel in &x {
            let sum: f32 = channel.iter().flat_map(|row| row.iter().copied()).sum();
            flattened.push(sum / spatial_size);
        }

        // FC common + ReLU — derive hidden size from actual weights
        let fc_hidden = self.fc_common_weight.len();
        let mut features = vec![0.0; fc_hidden];
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
