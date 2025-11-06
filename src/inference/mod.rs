//! Pure Rust inference module for WASM compatibility
//!
//! This module provides a lightweight neural network inference engine
//! that can be compiled to WebAssembly. It avoids dependencies on
//! libtorch/tch-rs, implementing only forward pass for trained models.

pub mod nn;
pub mod weights;

use serde::{Deserialize, Serialize};

/// Exported model format for WASM inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedModel {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension (number of actions)
    pub output_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Feature extractor weights and biases
    pub feature_extractor: LayerWeights,
    /// Policy head weights and biases
    pub policy_head: LayerWeights,
    /// Value head weights and biases (optional, not needed for inference)
    pub value_head: Option<LayerWeights>,
}

/// Weights and biases for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Weight matrix (flattened, row-major)
    pub weights: Vec<f32>,
    /// Bias vector
    pub biases: Vec<f32>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
}

impl ExportedModel {
    /// Create a new exported model
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        feature_extractor: LayerWeights,
        policy_head: LayerWeights,
        value_head: Option<LayerWeights>,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            hidden_dim,
            feature_extractor,
            policy_head,
            value_head,
        }
    }

    /// Run inference on the model
    pub fn predict(&self, observation: &[f32]) -> Vec<f32> {
        assert_eq!(
            observation.len(),
            self.input_dim,
            "Input dimension mismatch"
        );

        // Feature extraction with ReLU
        let features = self.feature_extractor.forward(observation);
        let features_relu: Vec<f32> = features.iter().map(|&x| x.max(0.0)).collect();

        // Policy head
        let logits = self.policy_head.forward(&features_relu);

        // Softmax for action probabilities
        softmax(&logits)
    }

    /// Get the best action (greedy policy)
    pub fn get_action(&self, observation: &[f32]) -> usize {
        let probs = self.predict(observation);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Get action probabilities
    pub fn get_action_probs(&self, observation: &[f32]) -> Vec<f32> {
        self.predict(observation)
    }
}

impl LayerWeights {
    /// Create new layer weights
    pub fn new(weights: Vec<f32>, biases: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(
            weights.len(),
            in_features * out_features,
            "Weight matrix size mismatch"
        );
        assert_eq!(biases.len(), out_features, "Bias vector size mismatch");

        Self {
            weights,
            biases,
            in_features,
            out_features,
        }
    }

    /// Forward pass through a linear layer
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features, "Input size mismatch");

        let mut output = vec![0.0; self.out_features];

        for i in 0..self.out_features {
            let mut sum = self.biases[i];
            for j in 0..self.in_features {
                // Row-major indexing: weights[i * in_features + j]
                sum += self.weights[i * self.in_features + j] * input[j];
            }
            output[i] = sum;
        }

        output
    }
}

/// Softmax activation function
fn softmax(logits: &[f32]) -> Vec<f32> {
    // Subtract max for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        // 2D input -> 3D output
        let weights = vec![
            1.0, 2.0, // First output neuron
            3.0, 4.0, // Second output neuron
            5.0, 6.0, // Third output neuron
        ];
        let biases = vec![0.1, 0.2, 0.3];
        let layer = LayerWeights::new(weights, biases, 2, 3);

        let input = vec![1.0, 2.0];
        let output = layer.forward(&input);

        // Expected: [1*1 + 2*2 + 0.1, 1*3 + 2*4 + 0.2, 1*5 + 2*6 + 0.3]
        //         = [5.1, 11.2, 17.3]
        assert!((output[0] - 5.1).abs() < 1e-5);
        assert!((output[1] - 11.2).abs() < 1e-5);
        assert!((output[2] - 17.3).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Should be in ascending order (higher logit -> higher prob)
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_exported_model() {
        // Simple 2-2-2 network
        let feature_weights = LayerWeights::new(
            vec![1.0, 0.0, 0.0, 1.0], // Identity-like
            vec![0.0, 0.0],
            2,
            2,
        );
        let policy_weights = LayerWeights::new(
            vec![1.0, -1.0, -1.0, 1.0], // Swap with negation
            vec![0.0, 0.0],
            2,
            2,
        );

        let model = ExportedModel::new(2, 2, 2, feature_weights, policy_weights, None);

        let obs = vec![1.0, 0.5];
        let action = model.get_action(&obs);

        // Should return valid action (0 or 1)
        assert!(action < 2);
    }
}
