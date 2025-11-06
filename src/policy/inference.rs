//! Inference-only model format for WASM deployment
//!
//! This module provides a pure Rust implementation of neural network inference
//! that doesn't depend on PyTorch/LibTorch, making it suitable for WebAssembly.

use anyhow::Result;
use serde::{Deserialize, Serialize};

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

    /// Forward pass: compute action logits and value
    ///
    /// # Arguments
    /// * `obs` - Observation vector [obs_dim]
    ///
    /// # Returns
    /// * `(logits, value)` - Action logits [action_dim] and state value (scalar)
    pub fn forward(&self, obs: &[f32]) -> (Vec<f32>, f32) {
        assert_eq!(obs.len(), self.obs_dim, "Observation dimension mismatch");

        // Layer 1: obs -> hidden_dim
        let mut hidden1 = vec![0.0; self.hidden_dim];
        for (i, row) in self.shared_fc1_weight.iter().enumerate() {
            for (j, &val) in obs.iter().enumerate() {
                hidden1[i] += row[j] * val;
            }
            hidden1[i] += self.shared_fc1_bias[i];
            // ReLU activation
            if hidden1[i] < 0.0 {
                hidden1[i] = 0.0;
            }
        }

        // Layer 2: hidden_dim -> hidden_dim
        let mut hidden2 = vec![0.0; self.hidden_dim];
        for (i, row) in self.shared_fc2_weight.iter().enumerate() {
            for (j, &val) in hidden1.iter().enumerate() {
                hidden2[i] += row[j] * val;
            }
            hidden2[i] += self.shared_fc2_bias[i];
            // ReLU activation
            if hidden2[i] < 0.0 {
                hidden2[i] = 0.0;
            }
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
        probs.iter()
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
