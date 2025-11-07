//! Universal inference system for diverse neural network architectures
//!
//! This module provides a flexible, extensible inference engine that can handle
//! various model architectures defined in JSON format, without hardcoding
//! specific layer types or architectures.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Activation function types supported by the inference engine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Identity,
    #[serde(rename = "GELU")]
    Gelu,
    #[serde(rename = "Swish")]
    Swish,
}

impl Activation {
    /// Apply activation function to a single value
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Identity => x,
            Activation::Gelu => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                const SQRT_2_OVER_PI: f32 = 0.7978845608;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh())
            }
            Activation::Swish => x / (1.0 + (-x).exp()), // x * sigmoid(x)
        }
    }

    /// Apply activation function to a vector in-place
    #[inline]
    pub fn apply_vec(&self, vec: &mut [f32]) {
        for val in vec.iter_mut() {
            *val = self.apply(*val);
        }
    }
}

/// Layer type definition for the universal inference system
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Layer {
    /// Fully connected (linear) layer
    #[serde(rename = "linear")]
    Linear {
        name: String,
        in_features: usize,
        out_features: usize,
        weight: Vec<Vec<f32>>, // [out_features, in_features]
        bias: Vec<f32>,        // [out_features]
    },

    /// 2D Convolutional layer
    #[serde(rename = "conv2d")]
    Conv2d {
        name: String,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize, // Assumes square kernel
        padding: usize,
        stride: usize,
        weight: Vec<Vec<Vec<Vec<f32>>>>, // [out_channels, in_channels, kernel_h, kernel_w]
        bias: Vec<f32>,                  // [out_channels]
    },

    /// Activation layer
    #[serde(rename = "activation")]
    Activation { name: String, activation: Activation },

    /// Reshape/Flatten layer
    #[serde(rename = "flatten")]
    Flatten { name: String },

    /// Reshape layer with specific dimensions
    #[serde(rename = "reshape")]
    Reshape { name: String, shape: Vec<usize> },

    /// Residual/Skip connection (adds input to output)
    #[serde(rename = "residual")]
    Residual { name: String, layers: Vec<Layer> },
}

impl Layer {
    /// Get the layer name
    pub fn name(&self) -> &str {
        match self {
            Layer::Linear { name, .. } => name,
            Layer::Conv2d { name, .. } => name,
            Layer::Activation { name, .. } => name,
            Layer::Flatten { name } => name,
            Layer::Reshape { name, .. } => name,
            Layer::Residual { name, .. } => name,
        }
    }

    /// Apply this layer to input data
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Layer::Linear { weight, bias, out_features, .. } => {
                // Input should be 1D [in_features]
                let in_vec = input.as_1d()?;
                let mut output = vec![0.0; *out_features];

                for (i, row) in weight.iter().enumerate() {
                    for (j, &val) in in_vec.iter().enumerate() {
                        output[i] += row[j] * val;
                    }
                    output[i] += bias[i];
                }

                Ok(Tensor::D1(output))
            }

            Layer::Conv2d { weight, bias, padding, stride, .. } => {
                // Input should be 3D [channels, height, width]
                let input_3d = input.as_3d()?;
                let output = Self::conv2d_forward(input_3d, weight, bias, *padding, *stride)?;
                Ok(Tensor::D3(output))
            }

            Layer::Activation { activation, .. } => {
                let mut output = input.clone();
                match &mut output {
                    Tensor::D1(v) => activation.apply_vec(v),
                    Tensor::D3(v) => {
                        for channel in v.iter_mut() {
                            for row in channel.iter_mut() {
                                activation.apply_vec(row);
                            }
                        }
                    }
                }
                Ok(output)
            }

            Layer::Flatten { .. } => {
                // Flatten any tensor to 1D
                let flat = match input {
                    Tensor::D1(v) => v.clone(),
                    Tensor::D3(v) => {
                        let mut flat = Vec::new();
                        for channel in v {
                            for row in channel {
                                flat.extend_from_slice(row);
                            }
                        }
                        flat
                    }
                };
                Ok(Tensor::D1(flat))
            }

            Layer::Reshape { shape, .. } => {
                // Flatten input first, then reshape
                let flat = match input {
                    Tensor::D1(v) => v.clone(),
                    Tensor::D3(v) => {
                        let mut flat = Vec::new();
                        for channel in v {
                            for row in channel {
                                flat.extend_from_slice(row);
                            }
                        }
                        flat
                    }
                };

                // Reshape based on target shape
                if shape.len() == 1 {
                    if flat.len() != shape[0] {
                        return Err(anyhow!(
                            "Reshape size mismatch: {} != {}",
                            flat.len(),
                            shape[0]
                        ));
                    }
                    Ok(Tensor::D1(flat))
                } else if shape.len() == 3 {
                    let (c, h, w) = (shape[0], shape[1], shape[2]);
                    if flat.len() != c * h * w {
                        return Err(anyhow!(
                            "Reshape size mismatch: {} != {}",
                            flat.len(),
                            c * h * w
                        ));
                    }

                    let mut reshaped = vec![vec![vec![0.0; w]; h]; c];
                    for ch in 0..c {
                        for row in 0..h {
                            for col in 0..w {
                                let idx = ch * h * w + row * w + col;
                                reshaped[ch][row][col] = flat[idx];
                            }
                        }
                    }
                    Ok(Tensor::D3(reshaped))
                } else {
                    Err(anyhow!("Unsupported reshape dimensions: {:?}", shape))
                }
            }

            Layer::Residual { layers, .. } => {
                // Apply sub-layers and add result to input
                let mut x = input.clone();
                for layer in layers {
                    x = layer.forward(&x)?;
                }

                // Add input to output (residual connection)
                let mut result = input.clone();
                match (&mut result, &x) {
                    (Tensor::D1(r), Tensor::D1(x)) => {
                        for (r_val, &x_val) in r.iter_mut().zip(x.iter()) {
                            *r_val += x_val;
                        }
                    }
                    (Tensor::D3(r), Tensor::D3(x)) => {
                        for (r_ch, x_ch) in r.iter_mut().zip(x.iter()) {
                            for (r_row, x_row) in r_ch.iter_mut().zip(x_ch.iter()) {
                                for (r_val, &x_val) in r_row.iter_mut().zip(x_row.iter()) {
                                    *r_val += x_val;
                                }
                            }
                        }
                    }
                    _ => return Err(anyhow!("Residual connection dimension mismatch")),
                }
                Ok(result)
            }
        }
    }

    /// Perform 2D convolution
    fn conv2d_forward(
        input: &[Vec<Vec<f32>>],
        weight: &[Vec<Vec<Vec<f32>>>],
        bias: &[f32],
        padding: usize,
        stride: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let out_channels = weight.len();
        let in_channels = input.len();
        let in_height = input[0].len();
        let in_width = input[0][0].len();
        let kernel_size = weight[0][0].len();

        let out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

        let mut output = vec![vec![vec![0.0; out_width]; out_height]; out_channels];

        for out_c in 0..out_channels {
            for h in 0..out_height {
                for w in 0..out_width {
                    let mut sum = bias[out_c];

                    for in_c in 0..in_channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = h * stride + kh;
                                let iw = w * stride + kw;

                                // Apply padding
                                let ih = ih as i32 - padding as i32;
                                let iw = iw as i32 - padding as i32;

                                if ih >= 0
                                    && ih < in_height as i32
                                    && iw >= 0
                                    && iw < in_width as i32
                                {
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

        Ok(output)
    }
}

/// Multi-dimensional tensor representation for intermediate activations
#[derive(Debug, Clone)]
pub enum Tensor {
    /// 1D tensor [size]
    D1(Vec<f32>),
    /// 3D tensor [channels, height, width]
    D3(Vec<Vec<Vec<f32>>>),
}

impl Tensor {
    /// Extract as 1D tensor
    pub fn as_1d(&self) -> Result<&Vec<f32>> {
        match self {
            Tensor::D1(v) => Ok(v),
            _ => Err(anyhow!("Expected 1D tensor")),
        }
    }

    /// Extract as 3D tensor
    pub fn as_3d(&self) -> Result<&Vec<Vec<Vec<f32>>>> {
        match self {
            Tensor::D3(v) => Ok(v),
            _ => Err(anyhow!("Expected 3D tensor")),
        }
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        match self {
            Tensor::D1(v) => v.len(),
            Tensor::D3(v) => v.len() * v[0].len() * v[0][0].len(),
        }
    }
}

/// Input specification for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputSpec {
    /// 1D vector input (e.g., for CartPole, SimpleBandit)
    #[serde(rename = "vector")]
    Vector { size: usize },

    /// 3D grid input (e.g., for Snake)
    #[serde(rename = "grid")]
    Grid { channels: usize, height: usize, width: usize },
}

/// Output specification for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// Number of actions (for policy head)
    pub num_actions: usize,
    /// Whether to include value head
    pub has_value: bool,
}

/// Training metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_episodes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_performance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_time_secs: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<HashMap<String, serde_json::Value>>,
}

/// Universal model definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalModel {
    /// Model version (for future compatibility)
    pub version: String,

    /// Input specification
    pub input: InputSpec,

    /// Output specification
    pub output: OutputSpec,

    /// Optional training metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,

    /// Shared feature extraction layers
    pub shared_layers: Vec<Layer>,

    /// Policy head layers (outputs action logits)
    pub policy_head: Vec<Layer>,

    /// Value head layers (outputs state value estimate)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_head: Option<Vec<Layer>>,
}

impl UniversalModel {
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

    /// Load model from JSON string
    pub fn from_json_str(json: &str) -> Result<Self> {
        let model = serde_json::from_str(json)?;
        Ok(model)
    }

    /// Convert raw input to tensor based on input spec
    fn input_to_tensor(&self, input: &[f32]) -> Result<Tensor> {
        match &self.input {
            InputSpec::Vector { size } => {
                if input.len() != *size {
                    return Err(anyhow!(
                        "Input size mismatch: expected {}, got {}",
                        size,
                        input.len()
                    ));
                }
                Ok(Tensor::D1(input.to_vec()))
            }
            InputSpec::Grid { channels, height, width } => {
                let grid_size = channels * height * width;
                if input.len() != grid_size {
                    return Err(anyhow!(
                        "Input size mismatch: expected {}, got {}",
                        grid_size,
                        input.len()
                    ));
                }

                // Reshape to [channels, height, width]
                let mut grid = vec![vec![vec![0.0; *width]; *height]; *channels];
                for c in 0..*channels {
                    for h in 0..*height {
                        for w in 0..*width {
                            let idx = c * height * width + h * width + w;
                            grid[c][h][w] = input[idx];
                        }
                    }
                }
                Ok(Tensor::D3(grid))
            }
        }
    }

    /// Forward pass through the model
    ///
    /// # Returns
    /// * `(logits, value)` - Action logits and optional state value
    pub fn forward(&self, input: &[f32]) -> Result<(Vec<f32>, Option<f32>)> {
        // Convert input to tensor
        let mut x = self.input_to_tensor(input)?;

        // Apply shared layers
        for layer in &self.shared_layers {
            x = layer.forward(&x)?;
        }

        // Apply policy head
        let mut policy_features = x.clone();
        for layer in &self.policy_head {
            policy_features = layer.forward(&policy_features)?;
        }
        let logits = policy_features.as_1d()?.clone();

        // Apply value head if present
        let value = if let Some(value_head) = &self.value_head {
            let mut value_features = x;
            for layer in value_head {
                value_features = layer.forward(&value_features)?;
            }
            let value_vec = value_features.as_1d()?;
            if value_vec.len() != 1 {
                return Err(anyhow!("Value head output should be size 1, got {}", value_vec.len()));
            }
            Some(value_vec[0])
        } else {
            None
        };

        Ok((logits, value))
    }

    /// Get action from input using argmax (deterministic)
    pub fn get_action(&self, input: &[f32]) -> Result<usize> {
        let (logits, _value) = self.forward(input)?;

        let action = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("No actions available"))?;

        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(1.0), 1.0);

        let tanh_result = Activation::Tanh.apply(0.0);
        assert!((tanh_result - 0.0).abs() < 1e-6);

        let sigmoid_result = Activation::Sigmoid.apply(0.0);
        assert!((sigmoid_result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_simple_mlp() -> Result<()> {
        // Create a simple 2-layer MLP
        let model = UniversalModel {
            version: "1.0".to_string(),
            input: InputSpec::Vector { size: 4 },
            output: OutputSpec { num_actions: 2, has_value: true },
            metadata: None,
            shared_layers: vec![
                Layer::Linear {
                    name: "fc1".to_string(),
                    in_features: 4,
                    out_features: 8,
                    weight: vec![vec![0.1; 4]; 8],
                    bias: vec![0.0; 8],
                },
                Layer::Activation { name: "relu1".to_string(), activation: Activation::ReLU },
                Layer::Linear {
                    name: "fc2".to_string(),
                    in_features: 8,
                    out_features: 8,
                    weight: vec![vec![0.1; 8]; 8],
                    bias: vec![0.0; 8],
                },
                Layer::Activation { name: "relu2".to_string(), activation: Activation::ReLU },
            ],
            policy_head: vec![Layer::Linear {
                name: "policy".to_string(),
                in_features: 8,
                out_features: 2,
                weight: vec![vec![0.1; 8]; 2],
                bias: vec![0.0; 2],
            }],
            value_head: Some(vec![Layer::Linear {
                name: "value".to_string(),
                in_features: 8,
                out_features: 1,
                weight: vec![vec![0.1; 8]],
                bias: vec![0.0],
            }]),
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (logits, value) = model.forward(&input)?;

        assert_eq!(logits.len(), 2);
        assert!(value.is_some());

        Ok(())
    }

    #[test]
    fn test_get_action() -> Result<()> {
        let model = UniversalModel {
            version: "1.0".to_string(),
            input: InputSpec::Vector { size: 2 },
            output: OutputSpec { num_actions: 2, has_value: false },
            metadata: None,
            shared_layers: vec![],
            policy_head: vec![Layer::Linear {
                name: "policy".to_string(),
                in_features: 2,
                out_features: 2,
                weight: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                bias: vec![0.0, 0.0],
            }],
            value_head: None,
        };

        let input = vec![2.0, 1.0];
        let action = model.get_action(&input)?;

        assert_eq!(action, 0); // First action should have higher logit (2.0 > 1.0)

        Ok(())
    }
}
