//! Q-Network for DQN training
//!
//! This module provides a multi-layer perceptron Q-network used by the
//! DQN training algorithm. The network outputs one Q-value per discrete
//! action and shares the same backbone topology as
//! [`crate::policy::mlp::MlpPolicy`] (2-layer Tanh MLP with orthogonal
//! initialization).
//!
//! Unlike `MlpPolicy`, `QNetwork` has a single output head — no separate
//! value head — and the head's outputs are interpreted directly as
//! `Q(s, a)` values (no softmax).
//!
//! # Architecture
//!
//! ```text
//! Input (observations) [batch, obs_dim]
//!         |
//!     [Dense(hidden_dim)]
//!         |
//!      Tanh
//!         |
//!     [Dense(hidden_dim)]
//!         |
//!      Tanh
//!         |
//!     [Dense(n_actions)]
//!         |
//!  Q-values [batch, n_actions]
//! ```

use anyhow::Result;
use tch::{
    Device, Tensor,
    nn::{self, Init, Module, OptimizerConfig},
};

/// Multi-layer perceptron Q-network for discrete actions
///
/// Implements a feedforward Q-network with:
/// - 2 shared hidden layers with Tanh activations
/// - Orthogonal weight initialization (gain = sqrt(2) for hidden, 0.01 for the
///   output head — same recipe as `MlpPolicy`)
/// - Single linear output head mapping to `n_actions` Q-values
///
/// The network is intentionally a sibling of `MlpPolicy` rather than a
/// modification of it: PPO depends on the exact structure of `MlpPolicy`,
/// and decoupling the Q-network keeps DQN changes self-contained.
pub struct QNetwork {
    vs: nn::VarStore,
    backbone: nn::Sequential,
    q_head: nn::Linear,
    device: Device,
    obs_dim: i64,
    n_actions: i64,
    hidden_dim: i64,
}

impl QNetwork {
    /// Create a new Q-network with the standard 2-layer Tanh backbone.
    ///
    /// # Arguments
    ///
    /// * `obs_dim` - Observation space dimensionality
    /// * `n_actions` - Number of discrete actions
    /// * `hidden_dim` - Size of hidden layers (typically 64 for low-dim
    ///   control)
    pub fn new(obs_dim: i64, n_actions: i64, hidden_dim: i64) -> Self {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let hidden_init = Init::Orthogonal { gain: 2.0_f64.sqrt() };
        let mut hidden_config = nn::LinearConfig::default();
        hidden_config.ws_init = hidden_init;

        // Shared backbone: two Linear -> Tanh blocks. The path names are
        // chosen so the VarStore layout mirrors `MlpPolicy::shared`: this
        // makes the parameter dictionaries comparable and keeps the WASM
        // export pathway uniform if QNetwork export is added later.
        let backbone = nn::seq()
            .add(nn::linear(&root / "backbone" / "fc1", obs_dim, hidden_dim, hidden_config))
            .add_fn(|x| x.tanh())
            .add(nn::linear(&root / "backbone" / "fc2", hidden_dim, hidden_dim, hidden_config))
            .add_fn(|x| x.tanh());

        let output_init = Init::Orthogonal { gain: 0.01 };
        let mut output_config = nn::LinearConfig::default();
        output_config.ws_init = output_init;

        let q_head = nn::linear(&root / "q_head", hidden_dim, n_actions, output_config);
        let device = vs.device();

        Self { vs, backbone, q_head, device, obs_dim, n_actions, hidden_dim }
    }

    /// Forward pass: compute `Q(s, a)` for every action `a`.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor of shape `[batch, obs_dim]`
    ///
    /// # Returns
    /// Q-value tensor of shape `[batch, n_actions]`.
    pub fn forward(&self, obs: &Tensor) -> Tensor {
        let features = self.backbone.forward(obs);
        self.q_head.forward(&features)
    }

    /// Get the number of discrete actions this Q-network covers.
    pub fn n_actions(&self) -> i64 {
        self.n_actions
    }

    /// Get the observation dimension this Q-network expects.
    pub fn obs_dim(&self) -> i64 {
        self.obs_dim
    }

    /// Get the hidden layer width.
    pub fn hidden_dim(&self) -> i64 {
        self.hidden_dim
    }

    /// Get the device this network is on (CPU or CUDA).
    pub fn device(&self) -> Device {
        self.device
    }

    /// Borrow the underlying `VarStore` (e.g. for optimizer construction).
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Mutably borrow the underlying `VarStore`.
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Create an Adam optimizer for this Q-network.
    pub fn optimizer(&mut self, learning_rate: f64) -> nn::Optimizer {
        nn::Adam::default().build(&self.vs, learning_rate).unwrap()
    }

    /// Copy all parameters from `source` into this network's `VarStore`.
    ///
    /// Used by [`crate::train::dqn::DQNTrainer`] to perform a hard target-net
    /// sync: every `target_update_interval` env steps, the target network's
    /// weights are overwritten with the online network's weights.
    ///
    /// Delegates to `tch::nn::VarStore::copy`, which performs a shape-checked,
    /// tensor-by-tensor copy and returns an error if the two VarStores
    /// disagree on shape.
    pub fn copy_params_from(&mut self, source: &QNetwork) -> Result<()> {
        self.vs.copy(&source.vs)?;
        Ok(())
    }

    /// Save model parameters to a file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model parameters from a file.
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    /// Freeze gradients (e.g. so the target network is not updated by the
    /// optimizer if it ever ends up sharing one).
    pub fn freeze(&mut self) {
        self.vs.freeze();
    }

    /// Unfreeze gradients.
    pub fn unfreeze(&mut self) {
        self.vs.unfreeze();
    }
}

#[cfg(test)]
mod tests {
    use tch::Kind;

    use super::*;

    #[test]
    fn test_q_network_creation() {
        let q_net = QNetwork::new(4, 2, 64);
        assert_eq!(q_net.obs_dim(), 4);
        assert_eq!(q_net.n_actions(), 2);
        assert_eq!(q_net.hidden_dim(), 64);
    }

    #[test]
    fn test_forward_shape() {
        let q_net = QNetwork::new(4, 2, 64);
        let obs = Tensor::randn([8, 4], (Kind::Float, q_net.device()));
        let q_values = q_net.forward(&obs);
        assert_eq!(q_values.size(), vec![8, 2]);
    }

    #[test]
    fn test_copy_params_from_byte_equal() {
        let mut online = QNetwork::new(4, 2, 32);
        let mut target = QNetwork::new(4, 2, 32);

        // Sanity: before the copy the two networks should produce different
        // outputs because they were initialized from different RNG draws.
        let obs = Tensor::randn([4, 4], (Kind::Float, online.device()));
        let q_online_before = online.forward(&obs);
        let q_target_before = target.forward(&obs);
        let pre_diff = (&q_online_before - &q_target_before).abs().sum(Kind::Float);
        let pre_diff_val: f64 = pre_diff.try_into().unwrap_or(0.0);
        assert!(pre_diff_val > 0.0, "expected fresh nets to disagree before copy");

        // Hard copy of online → target.
        target.copy_params_from(&online).expect("copy_params_from failed");

        // After the copy every named variable must match byte-for-byte.
        let online_vars = online.var_store().variables();
        let target_vars = target.var_store().variables();
        assert_eq!(online_vars.len(), target_vars.len());
        for (name, online_t) in &online_vars {
            let target_t = target_vars.get(name).expect("missing var in target");
            let diff = (online_t - target_t).abs().sum(Kind::Float);
            let diff_val: f64 = diff.try_into().unwrap_or(f64::INFINITY);
            assert_eq!(
                diff_val, 0.0,
                "var {} differs after copy_params_from (sum |diff| = {})",
                name, diff_val
            );
        }

        // And the forward pass must now agree exactly.
        let q_online_after = online.forward(&obs);
        let q_target_after = target.forward(&obs);
        let post_diff = (&q_online_after - &q_target_after).abs().sum(Kind::Float);
        let post_diff_val: f64 = post_diff.try_into().unwrap_or(f64::INFINITY);
        assert_eq!(post_diff_val, 0.0, "Q outputs differ after copy");

        // Mutating online should NOT now affect target (independent VarStores).
        let mut opt = online.optimizer(1e-2);
        let pred = online.forward(&obs);
        let loss = pred.square().mean(Kind::Float);
        opt.zero_grad();
        loss.backward();
        opt.step();
        let q_online_after_step = online.forward(&obs);
        let q_target_after_step = target.forward(&obs);
        let drift = (&q_online_after_step - &q_target_after_step).abs().sum(Kind::Float);
        let drift_val: f64 = drift.try_into().unwrap_or(0.0);
        assert!(drift_val > 0.0, "target unexpectedly tracked online net after grad step");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let q_net = QNetwork::new(4, 2, 32);
        let obs = Tensor::randn([4, 4], (Kind::Float, q_net.device()));
        let q_before = q_net.forward(&obs);
        let path = std::env::temp_dir().join("thrust_test_q_network.safetensors");
        q_net.save(&path).unwrap();

        let mut q_net2 = QNetwork::new(4, 2, 32);
        q_net2.load(&path).unwrap();
        let q_after = q_net2.forward(&obs);

        let diff = (&q_before - &q_after).abs().mean(Kind::Float);
        let diff_val: f64 = diff.try_into().unwrap();
        assert!(diff_val < 1e-6);

        std::fs::remove_file(&path).ok();
    }
}
