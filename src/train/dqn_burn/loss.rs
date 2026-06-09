//! Burn-backend DQN loss math (phase 3 of the Burn migration, #80).
//!
//! Mirrors `crate::train::dqn::loss` (tch path) op-for-op. The two
//! sit side-by-side under their respective feature flags so the
//! existing tch trainer is not disturbed and the Burn-side trainer
//! [`crate::train::dqn_burn::trainer`] has a backend-agnostic surface
//! to call.
//!
//! # Numerical parity
//!
//! On identical inputs the Burn and tch implementations must agree to
//! `1e-4` absolute tolerance (DoD on issue #80). The
//! `crate::train::parity_tests` module enforces this when both
//! `training` and `training-burn` features are enabled.
//!
//! # tch::no_grad ↔ Burn detach
//!
//! tch's `tch::no_grad(|| ...)` is a thread-local autograd suppression
//! scope. Burn uses tape-based autograd (`Autodiff<Backend>`); the
//! analogous operation is to wrap a non-autodiff "inner backend" tensor
//! in the autodiff wrapper *after* the bootstrap computation, which the
//! Burn ecosystem expresses through `Tensor::detach()` (the boundary
//! between grad-bearing and grad-free tensors).
//!
//! The TD-target computation in [`compute_td_target`] /
//! [`compute_td_target_double`] returns a fully detached tensor — its
//! gradient (if any) is discarded inside the function. Callers therefore
//! see the same "the target does not flow gradients" semantics as the
//! tch path, without needing a `no_grad` scope at the call site.

use burn::tensor::{Int, Tensor, backend::Backend};

/// Compute the vanilla DQN TD target.
///
/// ```text
/// y = r + γ · (1 − done) · max_aʹ Q_target(sʹ, aʹ)
/// ```
///
/// Returns a `[batch]` tensor that is **detached** from the autograd
/// graph (gradients do not flow back through `next_q_target`).
pub fn compute_td_target<B: Backend>(
    rewards: Tensor<B, 1>,
    dones: Tensor<B, 1>,
    next_q_target: Tensor<B, 2>,
    gamma: f64,
) -> Tensor<B, 1> {
    // max_aʹ Q_target(sʹ, aʹ) — Burn returns a [batch, 1] tensor; squeeze.
    let next_max: Tensor<B, 1> = next_q_target.max_dim(1).squeeze_dim(1);
    let one = Tensor::<B, 1>::ones_like(&dones);
    let not_done = one - dones;
    let target = rewards + (not_done * next_max).mul_scalar(gamma as f32);
    target.detach()
}

/// Compute the Double-DQN TD target.
///
/// ```text
/// a* = argmax_aʹ Q_online(sʹ, aʹ)
/// y  = r + γ · (1 − done) · Q_target(sʹ, a*)
/// ```
///
/// Action selection uses the *online* network and action evaluation
/// uses the *target* network. Returns a detached `[batch]` tensor.
pub fn compute_td_target_double<B: Backend>(
    rewards: Tensor<B, 1>,
    dones: Tensor<B, 1>,
    next_q_online: Tensor<B, 2>,
    next_q_target: Tensor<B, 2>,
    gamma: f64,
) -> Tensor<B, 1> {
    // Choose the action with the online net, evaluate with the target net.
    let a_star: Tensor<B, 2, Int> = next_q_online.argmax(1); // [batch, 1]
    // `gather` over the action axis returns [batch, 1]; squeeze.
    let next_q: Tensor<B, 1> = next_q_target.gather(1, a_star).squeeze_dim(1);
    let one = Tensor::<B, 1>::ones_like(&dones);
    let not_done = one - dones;
    let target = rewards + (not_done * next_q).mul_scalar(gamma as f32);
    target.detach()
}

/// Gather the Q-value of the action that was actually taken.
///
/// `[batch, n_actions]` x `[batch]` → `[batch]`.
pub fn gather_action_q<B: Backend>(
    q_online_all: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let actions_2d: Tensor<B, 2, Int> = actions.unsqueeze_dim(1); // [batch, 1]
    q_online_all.gather(1, actions_2d).squeeze_dim(1)
}

/// Smooth-L1 (Huber) loss between Q(s, a) and the TD target.
///
/// Burn 0.21 ships `nn::loss::HuberLoss` but its API binds the delta
/// at config time; we instead express the math directly so the
/// function signature mirrors the tch path's `compute_loss` exactly.
///
/// For a delta of 1.0 (standard DQN):
/// - If `|x| <= 1`: loss = 0.5 * x²
/// - If `|x| > 1`:  loss = |x| - 0.5
///
/// Returns the mean across the batch as a scalar tensor.
pub fn compute_loss<B: Backend>(
    q_online_taken: Tensor<B, 1>,
    td_target: Tensor<B, 1>,
) -> Tensor<B, 1> {
    huber_per_sample(q_online_taken - td_target, 1.0).mean()
}

/// Per-sample Smooth-L1 (Huber) loss with explicit delta. Useful for
/// the prioritized-replay path which needs per-sample weights before
/// reduction.
pub fn huber_per_sample<B: Backend>(diff: Tensor<B, 1>, delta: f64) -> Tensor<B, 1> {
    let delta_f = delta as f32;
    let abs_diff = diff.clone().abs();
    let quadratic = abs_diff.clone().clamp(0.0, delta);
    let linear = abs_diff - quadratic.clone();
    // 0.5 * quadratic² + delta * linear  (continuous & differentiable
    // at |x| = delta).
    quadratic.clone().mul(quadratic).mul_scalar(0.5_f32) + linear.mul_scalar(delta_f)
}

/// One-shot convenience wrapper: full vanilla-DQN loss.
pub fn compute_dqn_loss<B: Backend>(
    q_online_all: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    rewards: Tensor<B, 1>,
    next_q_target_all: Tensor<B, 2>,
    dones: Tensor<B, 1>,
    gamma: f64,
) -> Tensor<B, 1> {
    let q_taken = gather_action_q(q_online_all, actions);
    let target = compute_td_target(rewards, dones, next_q_target_all, gamma);
    compute_loss(q_taken, target)
}

/// One-shot convenience wrapper: full Double-DQN loss.
#[allow(clippy::too_many_arguments)]
pub fn compute_dqn_loss_double<B: Backend>(
    q_online_all: Tensor<B, 2>,
    next_q_online_all: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
    rewards: Tensor<B, 1>,
    next_q_target_all: Tensor<B, 2>,
    dones: Tensor<B, 1>,
    gamma: f64,
) -> Tensor<B, 1> {
    let q_taken = gather_action_q(q_online_all, actions);
    let target = compute_td_target_double(
        rewards,
        dones,
        next_q_online_all,
        next_q_target_all,
        gamma,
    );
    compute_loss(q_taken, target)
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Int, Tensor, TensorData},
    };

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn tensor1d(data: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
    }

    fn tensor2d(data: &[f32], rows: usize, cols: usize) -> Tensor<B, 2> {
        let device = Default::default();
        Tensor::<B, 2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), &device)
    }

    fn actions1d(data: &[i64]) -> Tensor<B, 1, Int> {
        let device = Default::default();
        Tensor::<B, 1, Int>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
    }

    fn read_vec(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    #[test]
    fn test_td_target_terminal_drops_bootstrap() {
        let rewards = tensor1d(&[1.0, 2.0]);
        let dones = tensor1d(&[0.0, 1.0]);
        let next_q = tensor2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let target = compute_td_target(rewards, dones, next_q, 0.9);
        let vals = read_vec(target);
        // Sample 0: 1.0 + 0.9*3.0 = 3.7
        // Sample 1: done → 2.0
        assert!((vals[0] - 3.7).abs() < 1e-4);
        assert!((vals[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_gather_action_q_burn() {
        let q = tensor2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let actions = actions1d(&[0, 1, 0]);
        let gathered = gather_action_q(q, actions);
        let vals = read_vec(gathered);
        assert_eq!(vals, vec![1.0, 4.0, 5.0]);
    }

    #[test]
    fn test_double_dqn_target_hand_computed() {
        // Same setup as the tch test.
        let rewards = tensor1d(&[1.0, 0.5]);
        let dones = tensor1d(&[0.0, 1.0]);
        let q_online_next = tensor2d(&[0.5, 2.0, 1.0, 0.5], 2, 2);
        let q_target_next = tensor2d(&[10.0, 3.0, 7.0, 4.0], 2, 2);

        let target =
            compute_td_target_double(rewards, dones, q_online_next, q_target_next, 0.9);
        let vals = read_vec(target);
        assert!((vals[0] - 3.7).abs() < 1e-4, "double-Q target sample 0 = {}", vals[0]);
        assert!((vals[1] - 0.5).abs() < 1e-4, "double-Q target sample 1 = {}", vals[1]);
    }

    #[test]
    fn test_huber_per_sample_quadratic_branch() {
        // Small residuals: 0.5 * x² applies.
        let diff = tensor1d(&[0.1, -0.2, 0.5]);
        let loss = huber_per_sample(diff, 1.0);
        let vals = read_vec(loss);
        // Expected: 0.005, 0.02, 0.125
        assert!((vals[0] - 0.005).abs() < 1e-4);
        assert!((vals[1] - 0.02).abs() < 1e-4);
        assert!((vals[2] - 0.125).abs() < 1e-4);
    }

    #[test]
    fn test_huber_per_sample_linear_branch() {
        // Large residuals: |x| - 0.5 applies.
        let diff = tensor1d(&[2.0, -3.0]);
        let loss = huber_per_sample(diff, 1.0);
        let vals = read_vec(loss);
        // |x| - 0.5: 1.5, 2.5
        assert!((vals[0] - 1.5).abs() < 1e-4, "got {}", vals[0]);
        assert!((vals[1] - 2.5).abs() < 1e-4, "got {}", vals[1]);
    }

    #[test]
    fn test_compute_loss_finite_on_zero_residual() {
        let q = tensor1d(&[1.0, 2.0, 3.0]);
        let target = tensor1d(&[1.0, 2.0, 3.0]);
        let loss = compute_loss(q, target);
        let v = loss.into_scalar();
        assert!((v - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_dqn_loss_runs_end_to_end() {
        let q_online = tensor2d(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], 4, 3);
        let actions = actions1d(&[0, 1, 2, 1]);
        let rewards = tensor1d(&[1.0, 0.0, -1.0, 0.5]);
        let dones = tensor1d(&[0.0, 1.0, 0.0, 0.0]);
        let next_q_target =
            tensor2d(&[0.5, 0.1, 0.0, 0.2, 0.3, 0.4, 1.0, 0.5, 0.7, 0.2, 0.8, 0.6], 4, 3);
        let loss = compute_dqn_loss(q_online, actions, rewards, next_q_target, dones, 0.99);
        let v = loss.into_scalar();
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }
}
