//! Multi-discrete actor-critic MLP for factored action spaces.
//!
//! Many environments use *factored* action spaces where a single decision
//! is composed of several independent discrete sub-decisions:
//!
//! - **Bucket Brigade**: `[house_index, mode] -> [10, 2]`
//! - **StarCraft micro**: `[action_type, target_unit, position]`
//!
//! Stuffing these into a single `Categorical` with cardinality `Π dims`
//! inflates the policy head dimension multiplicatively and discards the
//! factorization structure PPO can exploit.
//!
//! This module provides the multi-discrete analogue of [`MlpPolicy`]: a
//! shared trunk plus one [`nn::Linear`] action head per dimension. Per-step
//! log-probs are summed across dims (treating the dims as conditionally
//! independent), and per-step entropies are averaged.
//!
//! # Architecture
//!
//! ```text
//! Input (observations)
//!         |
//!     [Dense(64)]
//!         |
//!      Activation
//!         |
//!     [Dense(64)]
//!         |
//!      Activation
//!      /     |     \
//!  Action  Action  ...  Value
//!  Head_0  Head_1        Head
//!     |       |             |
//!  [Linear] [Linear]    [Linear]
//!     |       |             |
//!  Logits_0 Logits_1     Value
//! ```

use anyhow::Result;
use tch::{
    Device, Kind, Tensor,
    nn::{self, Init, Module, OptimizerConfig},
};

use super::mlp::{Activation, MlpConfig};

/// Multi-discrete MLP actor-critic policy.
///
/// Shares trunk-architecture parameters with [`MlpConfig`]. The only
/// per-instance difference is `action_dims: Vec<i64>` --- one head per
/// element, each producing logits over `action_dims[i]` discrete choices.
pub struct MultiDiscreteMlpPolicy {
    vs: nn::VarStore,
    shared: nn::Sequential,
    action_heads: Vec<nn::Linear>,
    value_head: nn::Linear,
    device: Device,
    action_dims: Vec<i64>,
    config: MlpConfig,
}

impl MultiDiscreteMlpPolicy {
    /// Create a new multi-discrete MLP policy with the default 2-layer
    /// architecture.
    pub fn new(obs_dim: i64, action_dims: Vec<i64>, hidden_dim: i64) -> Self {
        let config = MlpConfig { hidden_dim, ..Default::default() };
        Self::with_config(obs_dim, action_dims, config)
    }

    /// Create a new multi-discrete MLP policy with the given architecture
    /// configuration.
    pub fn with_config(obs_dim: i64, action_dims: Vec<i64>, config: MlpConfig) -> Self {
        assert!(!action_dims.is_empty(), "action_dims must have at least one element");
        for (i, d) in action_dims.iter().enumerate() {
            assert!(*d >= 1, "action_dims[{i}] = {d}; must be >= 1");
        }

        let device = Device::cuda_if_available();
        tracing::info!("MultiDiscreteMlpPolicy using device: {:?}", device);
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let hidden_init = if config.use_orthogonal_init {
            Init::Orthogonal { gain: 2.0_f64.sqrt() }
        } else {
            Init::Randn { mean: 0.0, stdev: 0.01 }
        };

        let mut linear_config = nn::LinearConfig::default();
        linear_config.ws_init = hidden_init;

        // Shared trunk (same construction as MlpPolicy --- keep them
        // structurally parallel so encoder-tap-based regularizers can mix
        // the two policy types).
        let mut shared = nn::seq();
        shared = shared
            .add(nn::linear(&root / "shared" / "fc1", obs_dim, config.hidden_dim, linear_config))
            .add_fn(move |x| match config.activation {
                Activation::ReLU => x.relu(),
                Activation::Tanh => x.tanh(),
            });
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

        let output_init = if config.use_orthogonal_init {
            Init::Orthogonal { gain: 0.01 }
        } else {
            Init::Randn { mean: 0.0, stdev: 0.01 }
        };
        let mut output_config = nn::LinearConfig::default();
        output_config.ws_init = output_init;

        let action_heads: Vec<nn::Linear> = action_dims
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                nn::linear(
                    &root / "policy" / format!("head_{i}"),
                    config.hidden_dim,
                    dim,
                    output_config,
                )
            })
            .collect();

        let value_head = nn::linear(&root / "value", config.hidden_dim, 1, output_config);
        let device = vs.device();

        Self { vs, shared, action_heads, value_head, device, action_dims, config }
    }

    /// Device the policy parameters live on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Action-dim layout (e.g. `[10, 2]` for Bucket Brigade `[house, mode]`).
    pub fn action_dims(&self) -> &[i64] {
        &self.action_dims
    }

    /// Forward pass: per-dim action logits plus value estimate.
    ///
    /// Returns `(Vec<logits_i>, value)` where `logits_i: [batch, action_dims[i]]`
    /// and `value: [batch]`.
    pub fn forward(&self, obs: &Tensor) -> (Vec<Tensor>, Tensor) {
        let features = self.shared.forward(obs);
        let logits: Vec<Tensor> =
            self.action_heads.iter().map(|h| h.forward(&features)).collect();
        let values = self.value_head.forward(&features).squeeze_dim(-1);
        (logits, values)
    }

    /// Shared-trunk features (same role as
    /// [`crate::policy::mlp::MlpPolicy::encoder_features`]).
    pub fn encoder_features(&self, obs: &Tensor) -> Tensor {
        self.shared.forward(obs)
    }

    /// Sample actions from the per-dim distributions.
    ///
    /// Returns:
    /// * `actions`   - `[batch, num_dims]` int64 tensor.
    /// * `log_probs` - `[batch]` summed log-probability across dims.
    /// * `values`    - `[batch]` value estimate.
    pub fn get_action(&self, obs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (logits_per_dim, values) = self.forward(obs);

        let mut per_dim_actions: Vec<Tensor> = Vec::with_capacity(logits_per_dim.len());
        let mut per_dim_log_probs: Vec<Tensor> = Vec::with_capacity(logits_per_dim.len());

        for logits in &logits_per_dim {
            let log_probs_all = logits.log_softmax(-1, Kind::Float);
            let probs = logits.softmax(-1, Kind::Float);
            let action = probs.multinomial(1, true).squeeze_dim(-1);
            let log_p = log_probs_all
                .gather(-1, &action.unsqueeze(-1), false)
                .squeeze_dim(-1);
            per_dim_actions.push(action);
            per_dim_log_probs.push(log_p);
        }

        // [batch, num_dims]
        let actions = Tensor::stack(&per_dim_actions, 1);
        // sum over dims to get joint log-prob (assumes conditional independence
        // across dims given the state).
        let log_probs = Tensor::stack(&per_dim_log_probs, 1).sum_dim_intlist(
            [1i64].as_slice(),
            false,
            Kind::Float,
        );

        (actions, log_probs, values)
    }

    /// Evaluate given actions: per-step summed log-prob, per-step mean
    /// entropy, and value.
    ///
    /// # Arguments
    /// * `obs`     - `[batch, obs_dim]`
    /// * `actions` - `[batch, num_dims]` int64 (one action per dim)
    ///
    /// # Returns
    /// `(log_probs [batch], entropy [batch], values [batch])`.
    /// `log_probs` is summed across dims; `entropy` is averaged across dims.
    pub fn evaluate_actions(
        &self,
        obs: &Tensor,
        actions: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let (logits_per_dim, values) = self.forward(obs);

        let num_dims = logits_per_dim.len() as i64;
        let mut summed_log_probs: Option<Tensor> = None;
        let mut summed_entropy: Option<Tensor> = None;

        for (i, logits) in logits_per_dim.iter().enumerate() {
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let probs = log_probs.exp();
            let per_dim_entropy = -(&probs * &log_probs).sum_dim_intlist(
                [-1i64].as_slice(),
                false,
                Kind::Float,
            );

            // actions[:, i]
            let actions_i = actions.select(1, i as i64);
            let per_dim_log_p = log_probs
                .gather(-1, &actions_i.unsqueeze(-1), false)
                .squeeze_dim(-1);

            summed_log_probs = Some(match summed_log_probs.take() {
                Some(acc) => acc + per_dim_log_p,
                None => per_dim_log_p,
            });
            summed_entropy = Some(match summed_entropy.take() {
                Some(acc) => acc + per_dim_entropy,
                None => per_dim_entropy,
            });
        }

        let log_probs = summed_log_probs.unwrap();
        // Mean entropy across dims (matches the convention used by the Python
        // joint trainer at bucket_brigade.training.joint_trainer).
        let entropy = summed_entropy.unwrap() / num_dims as f64;

        (log_probs, entropy, values)
    }

    /// Adam optimizer over the policy's parameters.
    pub fn optimizer(&mut self, learning_rate: f64) -> Result<nn::Optimizer> {
        Ok(nn::Adam::default().build(&self.vs, learning_rate)?)
    }

    /// Borrow the underlying var-store (for snapshotting / checkpointing).
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Total parameter count (handy for logging).
    pub fn num_parameters(&self) -> i64 {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum()
    }

    /// Architecture configuration (read-only).
    pub fn config(&self) -> &MlpConfig {
        &self.config
    }
}
