//! Loss computation helpers for DQN.
//!
//! These functions are factored out as free functions (mirroring
//! `compute_policy_loss` / `compute_value_loss` in PPO) so they can be
//! unit-tested without instantiating a full trainer.

use tch::{Reduction, Tensor};

/// Compute the TD target
///
/// ```text
/// y = r + γ · (1 − done) · max_aʹ Q_target(sʹ, aʹ)
/// ```
///
/// The target tensor must be detached from the autograd graph — this
/// function wraps the bootstrap computation in `tch::no_grad` so gradients
/// only flow through the online network in
/// [`compute_loss`].
///
/// # Arguments
/// * `rewards` - `[batch]`, `Kind::Float`
/// * `dones`   - `[batch]`, `Kind::Float` (1.0 = terminal, 0.0 = continuing)
/// * `next_q_target` - `[batch, n_actions]`, the target net's Q-values at sʹ
/// * `gamma` - discount factor
///
/// # Returns
/// `[batch]` TD-target tensor, with `requires_grad = false`.
pub fn compute_td_target(
    rewards: &Tensor,
    dones: &Tensor,
    next_q_target: &Tensor,
    gamma: f64,
) -> Tensor {
    tch::no_grad(|| {
        let next_max = next_q_target.max_dim(-1, false).0; // [batch]
        let not_done = 1.0 - dones;
        rewards + gamma * &not_done * next_max
    })
}

/// Gather the Q-value of the action that was actually taken.
///
/// # Arguments
/// * `q_online_all` - `[batch, n_actions]`
/// * `actions`      - `[batch]`, `Kind::Int64`
///
/// # Returns
/// `[batch]` Q-value tensor.
pub fn gather_action_q(q_online_all: &Tensor, actions: &Tensor) -> Tensor {
    let actions_2d = actions.unsqueeze(-1); // [batch, 1]
    q_online_all.gather(-1, &actions_2d, false).squeeze_dim(-1)
}

/// Compute the Smooth-L1 (Huber) loss between the online Q-values
/// `Q(s, a)` and the TD target.
///
/// Smooth-L1 is the standard DQN loss; it behaves like MSE near zero and
/// like L1 in the tail, which makes it robust to TD-target outliers
/// during early training.
///
/// # Arguments
/// * `q_online_taken` - `[batch]`, output of [`gather_action_q`]
/// * `td_target`      - `[batch]`, output of [`compute_td_target`]
///
/// # Returns
/// A scalar tensor (mean Huber loss across the batch).
pub fn compute_loss(q_online_taken: &Tensor, td_target: &Tensor) -> Tensor {
    q_online_taken.smooth_l1_loss(td_target, Reduction::Mean, 1.0)
}

/// One-shot convenience wrapper used by [`crate::train::dqn::DQNTrainer`].
///
/// Computes the full DQN loss given the online network's all-action
/// Q-values, the target network's next-state all-action Q-values, the
/// actions taken, and the (reward, done, γ) tuple.
///
/// # Returns
/// A scalar Smooth-L1 loss tensor (requires_grad through `q_online_all`).
pub fn compute_dqn_loss(
    q_online_all: &Tensor,
    actions: &Tensor,
    rewards: &Tensor,
    next_q_target_all: &Tensor,
    dones: &Tensor,
    gamma: f64,
) -> Tensor {
    let q_taken = gather_action_q(q_online_all, actions);
    let target = compute_td_target(rewards, dones, next_q_target_all, gamma);
    compute_loss(&q_taken, &target)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind};

    use super::*;

    #[test]
    fn test_td_target_terminal_drops_bootstrap() {
        let rewards = Tensor::from_slice(&[1.0f32, 2.0]).to_device(Device::Cpu);
        let dones = Tensor::from_slice(&[0.0f32, 1.0]).to_device(Device::Cpu);
        // next_q has shape [2, 3]; max per row should be [3.0, 6.0].
        let next_q = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape([2, 3])
            .to_kind(Kind::Float);
        let target = compute_td_target(&rewards, &dones, &next_q, 0.9);
        let vals: Vec<f32> = Vec::try_from(target).unwrap();
        // Sample 0: r=1.0, not done, γ·max = 0.9·3.0 = 2.7 → 3.7
        // Sample 1: r=2.0, done → bootstrap zeroed → 2.0
        assert!((vals[0] - 3.7).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_action_q() {
        // q_online: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], actions [0, 1, 0]
        let q = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .reshape([3, 2])
            .to_kind(Kind::Float);
        let actions = Tensor::from_slice(&[0i64, 1, 0]).to_device(Device::Cpu);
        let gathered = gather_action_q(&q, &actions);
        let vals: Vec<f32> = Vec::try_from(gathered).unwrap();
        assert_eq!(vals, vec![1.0, 4.0, 5.0]);
    }

    #[test]
    fn test_compute_loss_finite_on_zero_residual() {
        let q = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).to_kind(Kind::Float);
        let target = q.copy();
        let loss = compute_loss(&q, &target);
        let v: f64 = loss.try_into().unwrap();
        assert!(v.is_finite());
        // Smooth-L1 of zero residual is zero.
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn test_compute_dqn_loss_runs_end_to_end() {
        let q_online_all = Tensor::randn([4, 3], (Kind::Float, Device::Cpu));
        let actions = Tensor::from_slice(&[0i64, 1, 2, 1]);
        let rewards = Tensor::from_slice(&[1.0f32, 0.0, -1.0, 0.5]);
        let dones = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 0.0]);
        let next_q_target_all = Tensor::randn([4, 3], (Kind::Float, Device::Cpu));
        let loss =
            compute_dqn_loss(&q_online_all, &actions, &rewards, &next_q_target_all, &dones, 0.99);
        let v: f64 = loss.try_into().unwrap();
        assert!(v.is_finite());
        assert!(v >= 0.0);
    }
}
