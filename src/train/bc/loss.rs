//! Behavioral Cloning loss math (Burn backend).
//!
//! Behavioral cloning is supervised learning: the policy's action head is
//! trained to reproduce expert action labels via cross-entropy. Unlike the
//! RL losses ([`crate::train::a2c::loss`], [`crate::train::ppo::loss`])
//! there is no advantage, no importance ratio, no entropy bonus, and no
//! value term — just the negative log-likelihood of the expert action under
//! the current policy.
//!
//! The loss is discrete-only, matching
//! [`MlpBurnPolicy`](crate::policy::mlp::MlpBurnPolicy)'s categorical action
//! head. We re-export PPO's host-side [`scalar_f64`] so callers can pull the
//! whole BC loss surface from `bc::loss` without reaching into `ppo::loss`.

use burn::tensor::{Int, Tensor, activation::log_softmax, backend::Backend};

// Re-export the shared host-side scalar extractor from the PPO loss module
// (the same one A2C re-exports), so the BC trainer and downstream callers
// can pull stats off the autograd tape without reaching into `ppo::loss`.
pub use crate::train::ppo::loss::scalar_f64;

/// Compute the supervised behavioral-cloning cross-entropy loss.
///
/// This is the mean negative log-likelihood of the expert action under the
/// current policy:
///
/// ```text
/// loss = -mean_i( log_softmax(logits_i)[action_i] )
/// ```
///
/// Concretely we take `log_softmax` over the action dimension, gather the
/// log-probability assigned to each example's expert action label, and
/// negate the batch mean. Minimizing this drives the policy's categorical
/// distribution toward the one-hot expert labels — the core of behavioral
/// cloning. It is the standard classification cross-entropy with integer
/// targets, specialized to the discrete action head of
/// [`MlpBurnPolicy`](crate::policy::mlp::MlpBurnPolicy) (the value head is
/// ignored for BC).
///
/// # Arguments
///
/// * `logits` - Pre-softmax action logits `[batch, action_dim]` from the policy
///   head (grad-bearing).
/// * `actions` - Expert action label per example `[batch]`, each in
///   `0..action_dim`.
///
/// # Returns
///
/// A scalar tensor (rank 1, shape `[1]`) carrying the grad-bearing
/// cross-entropy loss.
pub fn compute_bc_loss<B: Backend>(
    logits: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let [batch, _action_dim] = logits.dims();

    // Log-probabilities over the action dimension: [batch, action_dim].
    let log_probs = log_softmax(logits, 1);

    // Gather the log-prob of each example's expert action. `gather` needs a
    // matching-rank index tensor, so reshape labels [batch] -> [batch, 1].
    let action_index = actions.reshape([batch, 1]);
    let chosen = log_probs.gather(1, action_index); // [batch, 1]

    // NLL of the expert action, averaged over the batch.
    chosen.reshape([batch]).mean().neg()
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::TensorData,
    };

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn logits2d(data: &[f32], rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), &device)
    }

    fn actions1d(data: &[i64]) -> Tensor<B, 1, Int> {
        let device = Default::default();
        Tensor::<B, 1, Int>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
    }

    #[test]
    fn bc_loss_is_finite_scalar() {
        // 2 examples, 3 actions.
        let logits = logits2d(&[0.1, 0.2, 0.3, -0.5, 0.0, 0.5], 2, 3);
        let actions = actions1d(&[2, 0]);
        let loss = compute_bc_loss(logits, actions);
        assert_eq!(loss.dims(), [1]);
        assert!(scalar_f64(loss).is_finite());
    }

    #[test]
    fn bc_loss_is_nonnegative() {
        // Cross-entropy / NLL is always >= 0.
        let logits = logits2d(&[1.0, 2.0, -1.0, 0.0, 3.0, 0.5], 2, 3);
        let actions = actions1d(&[1, 1]);
        let loss_val = scalar_f64(compute_bc_loss(logits, actions));
        assert!(loss_val >= 0.0, "cross-entropy loss must be non-negative, got {loss_val}");
    }

    /// Pushing the logits toward the expert label must *decrease* the loss
    /// (the supervised-learning analogue of the A2C loss-direction tests).
    #[test]
    fn bc_loss_decreases_when_logits_match_labels() {
        // Both examples have expert action 0.
        let actions = actions1d(&[0, 0]);

        // Flat logits: uniform distribution over 3 actions -> loss ~= ln(3).
        let flat = scalar_f64(compute_bc_loss(logits2d(&[0.0; 6], 2, 3), actions.clone()));

        // Confident logits favoring action 0 -> much lower loss.
        let confident =
            scalar_f64(compute_bc_loss(logits2d(&[5.0, 0.0, 0.0, 5.0, 0.0, 0.0], 2, 3), actions));

        assert!(
            confident < flat,
            "logits favoring the expert action should lower loss: {confident} !< {flat}"
        );
    }

    /// A near-perfect classifier (huge logit on the correct action) drives
    /// the loss toward zero.
    #[test]
    fn bc_loss_approaches_zero_when_perfect() {
        let logits = logits2d(&[20.0, 0.0, 0.0, 0.0, 20.0, 0.0], 2, 3);
        let actions = actions1d(&[0, 1]);
        let loss_val = scalar_f64(compute_bc_loss(logits, actions));
        assert!(loss_val < 1e-3, "confident-correct logits should give ~0 loss, got {loss_val}");
    }
}
