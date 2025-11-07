//! Inference-only model format for WASM deployment
//!
//! This module provides a pure Rust implementation of neural network inference
//! that doesn't depend on PyTorch/LibTorch, making it suitable for WebAssembly.

use anyhow::Result;
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
}

impl SnakeCNNInference {
    /// Save model to JSON file
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }

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
}

/// Activation function type for inference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InferenceActivation {
    ReLU,
    Tanh,
}

/// Training metadata for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Total training steps
    pub total_steps: usize,
    /// Total episodes
    pub total_episodes: usize,
    /// Final average performance (steps per episode)
    pub final_performance: f64,
    /// Training wall time in seconds
    pub training_time_secs: f64,
    /// Device used for training (CPU, CUDA, MPS)
    pub device: String,
    /// Environment name
    pub environment: String,
    /// Training algorithm
    pub algorithm: String,
    /// Timestamp when trained
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Hyperparameters used for training
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// Additional notes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// A serializable MLP model for inference
///
/// This struct contains all the weights and biases needed to run
/// inference without any PyTorch dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceModel {
    /// Input dimension
    pub obs_dim: usize,
    /// Output dimension (number of actions)
    pub action_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Activation function
    #[serde(default = "default_activation")]
    pub activation: InferenceActivation,

    /// Training metadata (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TrainingMetadata>,

    /// Shared layer 1: weights [obs_dim, hidden_dim]
    pub shared_fc1_weight: Vec<Vec<f32>>,
    /// Shared layer 1: bias [hidden_dim]
    pub shared_fc1_bias: Vec<f32>,

    /// Shared layer 2: weights [hidden_dim, hidden_dim]
    pub shared_fc2_weight: Vec<Vec<f32>>,
    /// Shared layer 2: bias [hidden_dim]
    pub shared_fc2_bias: Vec<f32>,

    /// Policy head: weights [hidden_dim, action_dim]
    pub policy_weight: Vec<Vec<f32>>,
    /// Policy head: bias [action_dim]
    pub policy_bias: Vec<f32>,

    /// Value head: weights [hidden_dim, 1]
    pub value_weight: Vec<Vec<f32>>,
    /// Value head: bias [1]
    pub value_bias: Vec<f32>,
}

fn default_activation() -> InferenceActivation {
    InferenceActivation::Tanh
}

impl InferenceModel {
    /// Save model to JSON file
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from JSON file
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        Ok(model)
    }

    /// Apply activation function
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        match self.activation {
            InferenceActivation::ReLU => {
                if x < 0.0 {
                    0.0
                } else {
                    x
                }
            }
            InferenceActivation::Tanh => x.tanh(),
        }
    }

    /// Forward pass: compute action logits and value
    ///
    /// # Arguments
    /// * `obs` - Observation vector [obs_dim]
    ///
    /// # Returns
    /// * `(logits, value)` - Action logits [action_dim] and state value
    ///   (scalar)
    pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(obs.len(), self.obs_dim, "Observation dimension mismatch");

        // Layer 1: obs -> hidden_dim
        let mut hidden1 = vec![0.0; self.hidden_dim];
        for (i, row) in self.shared_fc1_weight.iter().enumerate() {
            for (j, &val) in obs.iter().enumerate() {
                hidden1[i] += row[j] * val;
            }
            hidden1[i] = self.activate(hidden1[i] + self.shared_fc1_bias[i]);
        }

        // Layer 2: hidden_dim -> hidden_dim
        let mut hidden2 = vec![0.0; self.hidden_dim];
        for (i, row) in self.shared_fc2_weight.iter().enumerate() {
            for (j, &val) in hidden1.iter().enumerate() {
                hidden2[i] += row[j] * val;
            }
            hidden2[i] = self.activate(hidden2[i] + self.shared_fc2_bias[i]);
        }

        // Policy head: hidden_dim -> action_dim
        let mut logits = vec![0.0; self.action_dim];
        for (i, row) in self.policy_weight.iter().enumerate() {
            for (j, &val) in hidden2.iter().enumerate() {
                logits[i] += row[j] * val;
            }
            logits[i] += self.policy_bias[i];
        }

        // Value head: hidden_dim -> 1
        let mut value = 0.0;
        for (i, row) in self.value_weight.iter().enumerate() {
            for (j, &val) in hidden2.iter().enumerate() {
                value += row[j] * val;
            }
        }
        value += self.value_bias[0];

        (logits, value)
    }

    /// Get action from observation using softmax sampling
    ///
    /// # Arguments
    /// * `obs` - Observation vector [obs_dim]
    ///
    /// # Returns
    /// * Action index
    pub fn get_action(&self, obs: &[f32]) -> usize {
        let (logits, _value) = self.forward(obs);

        // Convert logits to probabilities using softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // For deterministic behavior (useful for demo), just take argmax
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass() {
        // Create a simple model with known weights
        let model = InferenceModel {
            obs_dim: 2,
            action_dim: 2,
            hidden_dim: 4,
            activation: InferenceActivation::Tanh,
            metadata: None,
            shared_fc1_weight: vec![vec![1.0, 0.0]; 4],
            shared_fc1_bias: vec![0.0; 4],
            shared_fc2_weight: vec![vec![1.0, 0.0, 0.0, 0.0]; 4],
            shared_fc2_bias: vec![0.0; 4],
            policy_weight: vec![vec![1.0, 0.0, 0.0, 0.0]; 2],
            policy_bias: vec![0.0; 2],
            value_weight: vec![vec![1.0, 0.0, 0.0, 0.0]],
            value_bias: vec![0.0],
        };

        let obs = vec![1.0, 2.0];
        let (logits, value) = model.forward(&obs);

        assert_eq!(logits.len(), 2);
        assert!(value.is_finite());
    }

    #[test]
    fn test_get_action() {
        let model = InferenceModel {
            obs_dim: 2,
            action_dim: 2,
            hidden_dim: 4,
            activation: InferenceActivation::ReLU,
            metadata: None,
            shared_fc1_weight: vec![vec![1.0, 0.0]; 4],
            shared_fc1_bias: vec![0.0; 4],
            shared_fc2_weight: vec![vec![1.0, 0.0, 0.0, 0.0]; 4],
            shared_fc2_bias: vec![0.0; 4],
            policy_weight: vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 0.0]],
            policy_bias: vec![0.0, 0.0],
            value_weight: vec![vec![1.0, 0.0, 0.0, 0.0]],
            value_bias: vec![0.0],
        };

        let obs = vec![1.0, 2.0];
        let action = model.get_action(&obs);

        assert!(action < 2);
    }

    #[test]
    fn test_save_load_json() {
        let model = InferenceModel {
            obs_dim: 4,
            action_dim: 2,
            hidden_dim: 64,
            activation: InferenceActivation::Tanh,
            metadata: None,
            shared_fc1_weight: vec![vec![0.0; 4]; 64],
            shared_fc1_bias: vec![0.0; 64],
            shared_fc2_weight: vec![vec![0.0; 64]; 64],
            shared_fc2_bias: vec![0.0; 64],
            policy_weight: vec![vec![0.0; 64]; 2],
            policy_bias: vec![0.0; 2],
            value_weight: vec![vec![0.0; 64]],
            value_bias: vec![0.0],
        };

        let temp_path = "/tmp/test_inference_model.json";
        model.save_json(temp_path).unwrap();

        let loaded = InferenceModel::load_json(temp_path).unwrap();
        assert_eq!(loaded.obs_dim, model.obs_dim);
        assert_eq!(loaded.action_dim, model.action_dim);
        assert_eq!(loaded.hidden_dim, model.hidden_dim);

        std::fs::remove_file(temp_path).ok();
    }
}
