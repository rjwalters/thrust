//! A2C loss math (Burn backend).
//!
//! A2C is the *single-update-per-rollout, no-clipping* special case of
//! the actor-critic family. Its loss diverges from PPO
//! ([`crate::train::ppo::loss`]) in exactly two ways:
//!
//! 1. **Policy loss** — the plain policy-gradient objective `-mean(log_prob *
//!    advantage)` with **no importance ratio** and **no clipping**. PPO's
//!    [`compute_policy_loss`] instead forms the clipped surrogate `min(ratio *
//!    adv, clip(ratio) * adv)`.
//! 2. **Value loss** — plain MSE `mean((V(s) - returns)^2)` with **no
//!    value-clipping**. PPO's [`compute_value_loss`] takes the pessimistic max
//!    over clipped/unclipped squared errors.
//!
//! The entropy bonus is shared verbatim with PPO, so we re-export PPO's
//! [`compute_entropy_loss`] and host-side [`scalar_f64`] rather than
//! re-implement them.
//!
//! [`compute_policy_loss`]: crate::train::ppo::loss::compute_policy_loss
//! [`compute_value_loss`]: crate::train::ppo::loss::compute_value_loss

use burn::tensor::{Tensor, backend::Backend};

// Re-export the shared helpers from the PPO loss module so the A2C
// trainer (and downstream callers) can pull the whole loss surface from
// `a2c::loss` without reaching into `ppo::loss`.
pub use crate::train::ppo::loss::{compute_entropy_loss, scalar_f64};

/// Compute the A2C policy-gradient loss.
///
/// This is the un-clipped, ratio-free actor objective:
/// `-mean(log_prob * advantage)`. Because A2C is on-policy and performs a
/// single gradient step per rollout there is no behaviour/target policy
/// split, hence no importance ratio and no clipping (the key divergence
/// from PPO's [`compute_policy_loss`]).
///
/// The advantages are treated as constants (detached) — they are the
/// critic's estimate of the rollout's advantage and must not propagate
/// gradients back through the value head via the policy term. The caller
/// passes advantages computed outside the autograd tape (host-side GAE),
/// so this is naturally satisfied; we additionally detach defensively.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of the taken actions under the current
///   policy. `[batch]`.
/// * `advantages` - Per-sample advantage estimates. `[batch]`.
///
/// # Returns
///
/// A scalar tensor (rank 1, shape `[1]`) carrying the grad-bearing
/// policy-gradient loss.
///
/// [`compute_policy_loss`]: crate::train::ppo::loss::compute_policy_loss
pub fn compute_a2c_policy_loss<B: Backend>(
    log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Advantages are constants w.r.t. the policy/value parameters.
    let advantages = advantages.detach();
    (log_probs * advantages).mean().neg()
}

/// Compute the A2C value-function loss: plain MSE `mean((V(s) - returns)^2)`.
///
/// Unlike PPO's [`compute_value_loss`], there is **no value-clipping** and
/// no pessimistic max — A2C uses the straight regression target. The
/// `returns` are detached (bootstrap targets are constants w.r.t. the
/// current value head).
///
/// # Arguments
///
/// * `values` - Current value-function predictions `V(s_t)`. `[batch]`.
/// * `returns` - Bootstrap/Monte-Carlo return targets. `[batch]`.
///
/// # Returns
///
/// A scalar tensor (rank 1, shape `[1]`) carrying the grad-bearing value
/// loss.
///
/// [`compute_value_loss`]: crate::train::ppo::loss::compute_value_loss
pub fn compute_a2c_value_loss<B: Backend>(
    values: Tensor<B, 1>,
    returns: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let returns = returns.detach();
    (values - returns).powf_scalar(2.0_f32).mean()
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Tensor, TensorData},
    };

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn tensor1d(data: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
    }

    #[test]
    fn a2c_policy_loss_is_scalar() {
        let log_probs = tensor1d(&[-0.5, -1.0, -0.2]);
        let advantages = tensor1d(&[1.0, -1.0, 0.5]);
        let loss = compute_a2c_policy_loss(log_probs, advantages);
        assert_eq!(loss.dims(), [1]);
        assert!(scalar_f64(loss).is_finite());
    }

    /// Sign sanity: with all-positive advantages and all-negative
    /// log-probs (the usual regime — `log p <= 0`), the PG loss
    /// `-mean(log_prob * advantage)` is positive.
    #[test]
    fn a2c_policy_loss_sign_with_positive_advantage() {
        let log_probs = tensor1d(&[-0.5, -1.0, -0.2]);
        let advantages = tensor1d(&[1.0, 2.0, 0.5]);
        let loss_val = scalar_f64(compute_a2c_policy_loss(log_probs, advantages));
        // -mean(neg * pos) = -mean(neg) = positive.
        assert!(loss_val > 0.0, "expected positive PG loss, got {loss_val}");
    }

    /// Increasing the log-prob of a positively-advantaged action must
    /// *decrease* the policy-gradient loss (gradient ascent on reward).
    #[test]
    fn a2c_policy_loss_decreases_when_good_action_more_likely() {
        let advantages = tensor1d(&[1.0, 1.0, 1.0]);
        let low =
            scalar_f64(compute_a2c_policy_loss(tensor1d(&[-2.0, -2.0, -2.0]), advantages.clone()));
        let high = scalar_f64(compute_a2c_policy_loss(tensor1d(&[-0.5, -0.5, -0.5]), advantages));
        assert!(high < low, "higher log-prob on good actions should lower loss: {high} !< {low}");
    }

    #[test]
    fn a2c_value_loss_is_plain_mse() {
        // values - returns = [0.0, 0.5, -0.3] → squares [0, 0.25, 0.09]
        // mean = 0.34 / 3 = 0.11333...
        let values = tensor1d(&[1.0, 2.0, 0.5]);
        let returns = tensor1d(&[1.0, 1.5, 0.8]);
        let loss = compute_a2c_value_loss(values, returns);
        assert_eq!(loss.dims(), [1]);
        let expected = (0.0_f64 + 0.25 + 0.09) / 3.0;
        let loss_val = scalar_f64(loss);
        assert!(
            (loss_val - expected).abs() < 1e-5,
            "A2C value loss should be plain MSE {expected}, got {loss_val}"
        );
    }

    #[test]
    fn a2c_value_loss_zero_when_perfect() {
        let values = tensor1d(&[1.0, 2.0, 0.5]);
        let returns = tensor1d(&[1.0, 2.0, 0.5]);
        let loss_val = scalar_f64(compute_a2c_value_loss(values, returns));
        assert!(loss_val.abs() < 1e-6, "perfect predictions should give ~0 loss, got {loss_val}");
    }
}
