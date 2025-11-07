//! Multi-Layer Perceptron (MLP) policy for discrete actions
//!
//! This module provides a simple feedforward neural network policy using
//! tch-rs. The policy outputs both action probabilities and value estimates,
//! which is standard for actor-critic algorithms like PPO.
//!
//! # Architecture
//!
//! ```text
//! Input (observations)
//!         |
//!     [Dense(64)]
//!         |
//!      Tanh
//!         |
//!     [Dense(64)]
//!         |
//!      Tanh
//!         |
//!    [Dense(64)] (optional 3rd layer)
//!         |
//!      Tanh
//!      /     \
//!  Policy   Value
//!  Network  Network
//!     |        |
//! [Dense(n)]  [Dense(1)]
//!     |        |
//!  Actions   Value
//! ```

use anyhow::Result;
use tch::{
    Device, Kind, Tensor,
    nn::{self, Init, Module, OptimizerConfig},
};

/// Configuration for MLP policy architecture
#[derive(Debug, Clone)]
pub struct MlpConfig {
    pub num_layers: usize,
    pub hidden_dim: i64,
    pub use_orthogonal_init: bool,
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            use_orthogonal_init: true,
            activation: Activation::Tanh,
        }
    }
}

/// Multi-layer perceptron policy for discrete actions
///
/// Implements an actor-critic architecture with:
/// - Shared feature extraction layers (2-3 layers)
/// - Orthogonal weight initialization (better for RL)
/// - Separate policy head (outputs action logits)
/// - Separate value head (outputs state value estimate)
pub struct MlpPolicy {
    vs: nn::VarStore,
    shared: nn::Sequential,
    policy_head: nn::Linear,
    value_head: nn::Linear,
    device: Device,
    config: MlpConfig,
}

impl MlpPolicy {
    /// Create a new MLP policy with default 2-layer architecture
    ///
    /// # Arguments
    ///
    /// * `obs_dim` - Observation space dimensionality
    /// * `action_dim` - Number of discrete actions
    /// * `hidden_dim` - Size of hidden layers
    pub fn new(obs_dim: i64, action_dim: i64, hidden_dim: i64) -> Self {
        let config = MlpConfig { hidden_dim, ..Default::default() };
        Self::with_config(obs_dim, action_dim, config)
    }

    /// Create a new MLP policy with custom configuration
    ///
    /// # Arguments
    ///
    /// * `obs_dim` - Observation space dimensionality
    /// * `action_dim` - Number of discrete actions
    /// * `config` - Architecture configuration
    pub fn with_config(obs_dim: i64, action_dim: i64, config: MlpConfig) -> Self {
        let device = Device::cuda_if_available();
        tracing::info!("MlpPolicy using device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Weight initialization - use orthogonal for hidden layers
        let hidden_init = if config.use_orthogonal_init {
            Init::Orthogonal { gain: 2.0_f64.sqrt() }
        } else {
            Init::Randn { mean: 0.0, stdev: 0.01 }
        };

        let mut linear_config = nn::LinearConfig::default();
        linear_config.ws_init = hidden_init;

        // Build shared layers
        let mut shared = nn::seq();

        // First layer
        shared = shared
            .add(nn::linear(&root / "shared" / "fc1", obs_dim, config.hidden_dim, linear_config))
            .add_fn(move |x| match config.activation {
                Activation::ReLU => x.relu(),
                Activation::Tanh => x.tanh(),
            });

        // Second layer
        shared = shared
            .add(nn::linear(
                &root / "shared" / "fc2",
                config.hidden_dim,
                config.hidden_dim,
                linear_config,
            ))
            .add_fn(move |x| match config.activation {
                Activation::ReLU => x.relu(),
                Activation::Tanh => x.tanh(),
            });

        // Optional third layer
        if config.num_layers >= 3 {
            shared = shared
                .add(nn::linear(
                    &root / "shared" / "fc3",
                    config.hidden_dim,
                    config.hidden_dim,
                    linear_config,
                ))
                .add_fn(move |x| match config.activation {
                    Activation::ReLU => x.relu(),
                    Activation::Tanh => x.tanh(),
                });
        }

        // Policy and value heads with smaller gain for output layers
        let output_init = if config.use_orthogonal_init {
            Init::Orthogonal { gain: 0.01 }
        } else {
            Init::Randn { mean: 0.0, stdev: 0.01 }
        };

        let mut output_config = nn::LinearConfig::default();
        output_config.ws_init = output_init;

        let policy_head =
            nn::linear(&root / "policy", config.hidden_dim, action_dim, output_config);
        let value_head = nn::linear(&root / "value", config.hidden_dim, 1, output_config);
        let device = vs.device();

        Self { vs, shared, policy_head, value_head, device, config }
    }

    /// Forward pass: compute action logits and values
    pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let features = self.shared.forward(obs);
        let logits = self.policy_head.forward(&features);
        let values = self.value_head.forward(&features).squeeze_dim(-1);
        (logits, values)
    }

    /// Get action, log probability, and value for given observations
    pub fn get_action(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (logits, values) = self.forward(obs);

        // Use log_softmax for numerical stability
        let log_probs_all = logits.log_softmax(-1, Kind::Float);

        // Sample from probabilities (softmax already produces valid probabilities)
        let probs = logits.softmax(-1, Kind::Float);
        let actions = probs.multinomial(1, true).squeeze_dim(-1);

        // Get log probabilities for the sampled actions
        let log_probs = log_probs_all.gather(-1, &actions.unsqueeze(-1), false).squeeze_dim(-1);
        (actions, log_probs, values)
    }

    /// Evaluate actions: compute log probabilities and entropy
    pub fn evaluate_actions(&self, obs: &Tensor, actions: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (logits, values) = self.forward(obs);

        // Use log_softmax for numerical stability
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let probs = log_probs.exp();

        let action_log_probs = log_probs.gather(-1, &actions.unsqueeze(-1), false).squeeze_dim(-1);

        // Compute entropy: H = -Σ p(x) * log(p(x))
        // Use the same probs and log_probs (no clamping - softmax already handles it)
        let entropy =
            -(probs * log_probs).sum_dim_intlist(-1, false, Kind::Float).mean(Kind::Float);
        (action_log_probs, entropy, values)
    }

    /// Get the device this policy is on (CPU or CUDA)
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get mutable reference to variable store (for optimizer creation)
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Get reference to variable store
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Create an Adam optimizer for this policy
    pub fn optimizer(&mut self, learning_rate: f64) -> nn::Optimizer {
        nn::Adam::default().build(&self.vs, learning_rate).unwrap()
    }

    /// Save model parameters to a file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model parameters from a file
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    /// Freeze gradients (for evaluation or inference)
    pub fn freeze(&mut self) {
        self.vs.freeze();
    }

    /// Unfreeze gradients (for training)
    pub fn unfreeze(&mut self) {
        self.vs.unfreeze();
    }

    /// Export model weights for WASM inference
    ///
    /// Extracts all weights and biases from the PyTorch model and converts them
    /// to a pure Rust format that can be used in WebAssembly.
    ///
    /// Note: Only supports 2-layer models for now (3-layer export not yet
    /// implemented)
    pub fn export_for_inference(&self) -> crate::policy::inference::InferenceModel {
        use tch::Tensor;

        if self.config.num_layers > 2 {
            panic!("Export for 3+ layer models not yet implemented");
        }

        // Helper function to convert a 2D tensor to Vec<Vec<f32>>
        fn tensor_to_2d(tensor: &Tensor) -> Vec<Vec<f32>> {
            let size = tensor.size();
            assert_eq!(size.len(), 2, "Expected 2D tensor");
            let rows = size[0] as usize;
            let cols = size[1] as usize;

            // Move tensor to CPU and convert to f32, make contiguous, then flatten to 1D
            let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            let flat_tensor = cpu_tensor.view([-1]); // Flatten to 1D

            // Extract as Vec<f32>
            let flat: Vec<f32> = match Vec::try_from(flat_tensor) {
                Ok(v) => v,
                Err(e) => panic!(
                    "Failed to convert tensor to Vec: {:?}. Tensor shape: {:?}, device: {:?}",
                    e,
                    cpu_tensor.size(),
                    cpu_tensor.device()
                ),
            };

            // Reshape into 2D Vec<Vec<f32>>
            let mut result = Vec::with_capacity(rows);
            for i in 0..rows {
                result.push(flat[i * cols..(i + 1) * cols].to_vec());
            }
            result
        }

        // Helper function to convert a 1D tensor to Vec<f32>
        fn tensor_to_1d(tensor: &Tensor) -> Vec<f32> {
            let cpu_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
            // Ensure it's 1D by flattening
            let flat_tensor = cpu_tensor.view([-1]);
            match Vec::try_from(flat_tensor) {
                Ok(v) => v,
                Err(e) => panic!(
                    "Failed to convert tensor to Vec: {:?}. Tensor shape: {:?}, device: {:?}",
                    e,
                    cpu_tensor.size(),
                    cpu_tensor.device()
                ),
            }
        }

        // Get the variable store's named variables
        let variables = self.vs.variables();

        // Extract dimensions
        let obs_dim = variables.get("shared.fc1.weight").expect("Missing shared.fc1.weight").size()
            [1] as usize;
        let hidden_dim =
            variables.get("shared.fc1.weight").expect("Missing shared.fc1.weight").size()[0]
                as usize;
        let action_dim =
            variables.get("policy.weight").expect("Missing policy.weight").size()[0] as usize;

        // Extract weights - note: PyTorch stores linear weights transposed
        let shared_fc1_weight =
            tensor_to_2d(variables.get("shared.fc1.weight").expect("Missing shared.fc1.weight"));
        let shared_fc1_bias =
            tensor_to_1d(variables.get("shared.fc1.bias").expect("Missing shared.fc1.bias"));

        let shared_fc2_weight =
            tensor_to_2d(variables.get("shared.fc2.weight").expect("Missing shared.fc2.weight"));
        let shared_fc2_bias =
            tensor_to_1d(variables.get("shared.fc2.bias").expect("Missing shared.fc2.bias"));

        let policy_weight =
            tensor_to_2d(variables.get("policy.weight").expect("Missing policy.weight"));
        let policy_bias = tensor_to_1d(variables.get("policy.bias").expect("Missing policy.bias"));

        let value_weight =
            tensor_to_2d(variables.get("value.weight").expect("Missing value.weight"));
        let value_bias = tensor_to_1d(variables.get("value.bias").expect("Missing value.bias"));

        let activation = match self.config.activation {
            Activation::ReLU => crate::policy::inference::InferenceActivation::ReLU,
            Activation::Tanh => crate::policy::inference::InferenceActivation::Tanh,
        };

        crate::policy::inference::InferenceModel {
            obs_dim,
            action_dim,
            hidden_dim,
            activation,
            metadata: None, // Will be set by training script
            shared_fc1_weight,
            shared_fc1_bias,
            shared_fc2_weight,
            shared_fc2_bias,
            policy_weight,
            policy_bias,
            value_weight,
            value_bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_creation() {
        let policy = MlpPolicy::new(4, 2, 64);
        assert!(policy.device == Device::Cpu || policy.device == Device::Cuda(0));
    }

    #[test]
    fn test_forward_pass() {
        let policy = MlpPolicy::new(4, 2, 64);
        let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));

        let (logits, values) = policy.forward(&obs);

        assert_eq!(logits.size(), vec![8, 2]);
        assert_eq!(values.size(), vec![8]);
    }

    #[test]
    fn test_get_action() {
        let policy = MlpPolicy::new(4, 2, 64);
        let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));

        let (actions, log_probs, values) = policy.get_action(&obs);

        assert_eq!(actions.size(), vec![8]);
        assert_eq!(log_probs.size(), vec![8]);
        assert_eq!(values.size(), vec![8]);

        // Check actions are in valid range [0, 1]
        let actions_vec: Vec<i64> = Vec::try_from(actions).unwrap();
        for &action in &actions_vec {
            assert!(action == 0 || action == 1);
        }
    }

    #[test]
    fn test_evaluate_actions() {
        let policy = MlpPolicy::new(4, 2, 64);
        let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));
        let actions = Tensor::randint(2, [8], (Kind::Int64, policy.device()));

        let (log_probs, entropy, values) = policy.evaluate_actions(&obs, &actions);

        assert_eq!(log_probs.size(), vec![8]);
        assert_eq!(entropy.size(), Vec::<i64>::new()); // Scalar
        assert_eq!(values.size(), vec![8]);

        // Entropy should be positive for non-degenerate distribution
        let entropy_val: f64 = entropy.try_into().unwrap();
        assert!(entropy_val >= 0.0);
    }

    #[test]
    fn test_optimizer_creation() {
        let mut policy = MlpPolicy::new(4, 2, 64);
        let _optimizer = policy.optimizer(3e-4);
    }

    #[test]
    fn test_freeze_unfreeze() {
        let mut policy = MlpPolicy::new(4, 2, 64);

        // Freeze and unfreeze (no direct way to check state in tch-rs 0.22)
        policy.freeze();
        policy.unfreeze();

        // Just verify methods don't panic
    }

    #[test]
    fn test_save_load() {
        let policy = MlpPolicy::new(4, 2, 64);
        let temp_path = "/tmp/thrust_test_policy.safetensors";

        // Get initial output
        let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));
        let (logits_before, _) = policy.forward(&obs);

        // Save
        policy.save(temp_path).unwrap();

        // Create new policy and load
        let mut policy2 = MlpPolicy::new(4, 2, 64);
        policy2.load(temp_path).unwrap();
        let (logits_after, _) = policy2.forward(&obs);

        // Should be the same
        let diff = (&logits_before - &logits_after).abs().mean(Kind::Float);
        let diff_val: f64 = diff.try_into().unwrap();
        assert!(diff_val < 1e-5);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_batch_consistency() {
        let policy = MlpPolicy::new(4, 2, 64);

        // Single observation
        let obs_single = Tensor::randn([1, 4], (Kind::Float, policy.device()));
        let (logits_single, _) = policy.forward(&obs_single);

        // Batch of same observation
        let obs_batch = obs_single.repeat([8, 1]);
        let (logits_batch, _) = policy.forward(&obs_batch);

        // All batch outputs should be similar to single output
        for i in 0..8 {
            let logits_i = logits_batch.get(i);
            let diff = (&logits_single.squeeze() - &logits_i).abs().mean(Kind::Float);
            let diff_val: f64 = diff.try_into().unwrap();
            assert!(diff_val < 1e-5);
        }
    }
}
