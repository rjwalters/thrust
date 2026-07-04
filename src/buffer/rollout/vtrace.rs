//! V-trace off-policy advantage/return estimation (IMPALA).
//!
//! This module implements V-trace targets (Espeholt et al. 2018,
//! "IMPALA: Scalable Distributed Deep-RL with Importance Weighted
//! Actor-Learner Architectures", arXiv:1802.01561, Algorithm 1) as an
//! alternative to [GAE](super::gae) for the same [`RolloutBuffer`].
//!
//! Where GAE assumes the data is on-policy, V-trace corrects for the
//! mismatch between the **behavior** policy `μ` that collected the
//! trajectory and the current **target** policy `π` that is being
//! optimized. This makes actor-critic learning sound on slightly-stale
//! trajectories (replayed or asynchronously generated), which is the
//! basis for scalable distributed actor-learner setups.
//!
//! # Algorithm (Espeholt et al. 2018, Algorithm 1)
//!
//! For each timestep `t` with importance ratio
//! `r_t = π(a_t|x_t) / μ(a_t|x_t) = exp(target_lp_t − behavior_lp_t)`:
//!
//! ```text
//! rho_t = min(rho_bar, r_t)                              // clipped IS ratio (TD weight)
//! c_t   = min(c_bar,   r_t)                              // clipped trace coefficient
//! delta_t = rho_t * (r_t^env + gamma * V(x_{t+1}) - V(x_t))   // corrected TD error
//!
//! // V-trace target (backward recursion, analogous to GAE):
//! v_t = V(x_t) + delta_t + gamma * c_t * (v_{t+1} - V(x_{t+1}))
//!
//! // V-trace advantage (note the outer rho_t):
//! A_t = rho_t * (r_t^env + gamma * v_{t+1} - V(x_t))
//! ```
//!
//! where `r_t^env` is the environment reward, `v_{T}` bootstraps from the
//! target-policy value `V(x_T)`, and the recursion resets across episode
//! boundaries so signal does not leak between episodes.
//!
//! # On-policy identity
//!
//! When `μ == π` at every step (on-policy) all ratios are `1`, so
//! `rho_t = c_t = 1` and the V-trace target reduces analytically to the
//! standard n-step return; the V-trace advantage reduces to the
//! GAE(`λ = 1`) advantage. See the `on_policy_recovers_nstep_returns`
//! test.

use super::storage::RolloutBuffer;

/// Compute V-trace targets and advantages for a single environment trajectory.
///
/// This is the inner per-env kernel (mirrors
/// [`compute_gae_single_env`](super::gae::compute_gae_single_env)),
/// visible to the parent `rollout` module so the sibling `tests` module
/// can exercise it directly.
///
/// The backward recursion resets across episode boundaries: when
/// `terminated[t]` is set, the bootstrap into `t+1` is zeroed and the
/// trace does not carry `v_{t+1}` from the next episode.
///
/// # Arguments
/// * `rewards` - Per-step rewards `[T]`
/// * `values` - Per-step value estimates from the target policy at collection
///   time `[T]`
/// * `behavior_log_probs` - Per-step `log μ(a_t | s_t)` under the behavior
///   (actor) policy `[T]`
/// * `target_log_probs` - Per-step `log π(a_t | s_t)` under the current target
///   (learner) policy `[T]`
/// * `terminated` - Per-step episode termination flags `[T]`
/// * `bootstrap_value` - `V(s_T)` from the target policy (value of the state
///   immediately after the final step)
/// * `gamma` - Discount factor
/// * `rho_bar` - IS ratio clip for the TD target (Espeholt eq. 1; typically
///   1.0)
/// * `c_bar` - IS ratio clip for the trace coefficient (Espeholt eq. 2;
///   typically 1.0)
/// * `vtrace_targets` - Output: V-trace return targets `[T]` (written in-place)
/// * `advantages` - Output: V-trace advantages `[T]` (written in-place)
// Each argument is a distinct trajectory field; bundling into a struct
// would add boilerplate at every call site without improving clarity.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_vtrace_single_env(
    rewards: &[f32],
    values: &[f32],
    behavior_log_probs: &[f32],
    target_log_probs: &[f32],
    terminated: &[bool],
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
    vtrace_targets: &mut [f32],
    advantages: &mut [f32],
) {
    let num_steps = rewards.len();
    debug_assert_eq!(values.len(), num_steps);
    debug_assert_eq!(behavior_log_probs.len(), num_steps);
    debug_assert_eq!(target_log_probs.len(), num_steps);
    debug_assert_eq!(terminated.len(), num_steps);
    debug_assert_eq!(vtrace_targets.len(), num_steps);
    debug_assert_eq!(advantages.len(), num_steps);

    // Backward pass (analogous to GAE). At step t we read the already
    // computed V-trace target at t+1, so interior episode boundaries are
    // handled by zeroing the t+1 bootstrap when step t is terminal.
    //
    // Final-step precedence matches `compute_gae_single_env`: the
    // `t == num_steps - 1` branch is checked *before* `terminated[t]`, so
    // the very last rollout row always bootstraps from `bootstrap_value`
    // (the value of the post-rollout state supplied by the caller). This
    // keeps V-trace an exact drop-in for GAE on on-policy data — a terminal
    // flag on the final row does not diverge the two estimators. Interior
    // terminal flags (`t < num_steps - 1`) still zero the bootstrap.
    for t in (0..num_steps).rev() {
        let is_last = t == num_steps - 1;
        let terminal = terminated[t];

        // Bootstrap value `V(x_{t+1})` (target policy) for the TD error.
        let next_value = if is_last {
            // Final rollout row: bootstrap from the post-rollout value
            // (matches GAE precedence — checked before `terminal`).
            bootstrap_value
        } else if terminal {
            // Interior episode boundary: no bootstrap into the next episode.
            0.0
        } else {
            values[t + 1]
        };

        // V-trace target at t+1 used by the advantage's n-step lookahead.
        let next_vtrace = if is_last {
            // v_T == bootstrap_value (the recursion terminates with d_T = 0).
            bootstrap_value
        } else if terminal {
            0.0
        } else {
            vtrace_targets[t + 1]
        };

        // Trace carry `v_{t+1} - V(x_{t+1})`. At the final step this is
        // `v_T - V(x_T) = 0`; across an interior terminal boundary it is
        // zeroed so the next episode's residual does not leak backward.
        let next_v_minus_baseline = if is_last || terminal {
            0.0
        } else {
            vtrace_targets[t + 1] - values[t + 1]
        };

        // Importance ratio r_t = exp(log π − log μ), clipped independently
        // for the TD weight (rho) and the trace coefficient (c).
        let ratio = (target_log_probs[t] - behavior_log_probs[t]).exp();
        let rho = ratio.min(rho_bar);
        let c = ratio.min(c_bar);

        // Corrected TD error and V-trace target recursion.
        let delta = rho * (rewards[t] + gamma * next_value - values[t]);
        let d = delta + gamma * c * next_v_minus_baseline;
        vtrace_targets[t] = values[t] + d;

        // V-trace advantage carries the *outer* rho_t (distinct from the
        // GAE delta): A_t = rho_t * (r_t + gamma * v_{t+1} - V(x_t)).
        advantages[t] = rho * (rewards[t] + gamma * next_vtrace - values[t]);
    }
}

/// Compute V-trace targets and advantages for all environments in the buffer.
///
/// Writes results into `buffer.advantages` and `buffer.returns` so the
/// trainer can call [`get_batch`](RolloutBuffer::get_batch) /
/// [`get_filled_batch`](RolloutBuffer::get_filled_batch) as usual after
/// this call — exactly the contract of
/// [`compute_advantages`](super::gae::compute_advantages).
///
/// # Arguments
/// * `buffer` - Rollout buffer (rewards, values, log_probs, terminated already
///   filled). The stored `log_probs` are the behavior-policy log-probs `log
///   μ(a_t | s_t)`.
/// * `target_log_probs` - Per-step target-policy log-probs
///   `[num_steps][num_envs]` (reevaluated by the learner's current policy over
///   the stored observations/actions)
/// * `last_values` - Bootstrap `V(s_{T+1})` per environment `[num_envs]`
/// * `gamma` - Discount factor
/// * `rho_bar` - Clip for rho (typically 1.0)
/// * `c_bar` - Clip for c (typically 1.0)
///
/// # Panics
/// Panics (via `assert`) if `target_log_probs` is not shaped
/// `[num_steps][num_envs]`.
pub fn compute_vtrace_advantages(
    buffer: &mut RolloutBuffer,
    target_log_probs: &[Vec<f32>],
    last_values: &[f32],
    gamma: f32,
    rho_bar: f32,
    c_bar: f32,
) {
    let (num_steps, num_envs, _) = buffer.shape();

    assert_eq!(
        target_log_probs.len(),
        num_steps,
        "target_log_probs must have num_steps ({}) rows, got {}",
        num_steps,
        target_log_probs.len()
    );
    debug_assert_eq!(last_values.len(), num_envs, "last_values length mismatch");

    if num_steps == 0 {
        return;
    }

    // Collect all immutable data first to avoid borrow-checker issues
    // (identical pattern to `gae::compute_advantages_partial`).
    let rewards: Vec<Vec<f32>> = buffer.rewards().iter().map(|step| step.to_vec()).collect();
    let values: Vec<Vec<f32>> = buffer.values().iter().map(|step| step.to_vec()).collect();
    let behavior_log_probs: Vec<Vec<f32>> =
        buffer.log_probs().iter().map(|step| step.to_vec()).collect();
    let terminated: Vec<Vec<bool>> = buffer.terminated().iter().map(|step| step.to_vec()).collect();

    for (row, tlp) in target_log_probs.iter().enumerate() {
        assert_eq!(
            tlp.len(),
            num_envs,
            "target_log_probs row {} must have num_envs ({}) columns, got {}",
            row,
            num_envs,
            tlp.len()
        );
    }

    // Now we can take mutable access to both advantages and returns.
    let (advantages, returns) = buffer.advantages_and_returns_mut();

    for env_id in 0..num_envs {
        let env_rewards: Vec<f32> = rewards.iter().map(|step| step[env_id]).collect();
        let env_values: Vec<f32> = values.iter().map(|step| step[env_id]).collect();
        let env_behavior: Vec<f32> = behavior_log_probs.iter().map(|step| step[env_id]).collect();
        let env_target: Vec<f32> = target_log_probs.iter().map(|step| step[env_id]).collect();
        let env_terminated: Vec<bool> = terminated.iter().map(|step| step[env_id]).collect();

        let mut env_targets: Vec<f32> = vec![0.0; num_steps];
        let mut env_advantages: Vec<f32> = vec![0.0; num_steps];

        compute_vtrace_single_env(
            &env_rewards,
            &env_values,
            &env_behavior,
            &env_target,
            &env_terminated,
            last_values[env_id],
            gamma,
            rho_bar,
            c_bar,
            &mut env_targets,
            &mut env_advantages,
        );

        // V-trace targets become the value-function targets (`returns`);
        // V-trace advantages become the policy-gradient `advantages`.
        for step in 0..num_steps {
            advantages[step][env_id] = env_advantages[step];
            returns[step][env_id] = env_targets[step];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::rollout::{gae::compute_gae_single_env, storage::RolloutBuffer};

    /// Tolerance for element-wise comparison against the numpy/Python
    /// reference (`scripts` under the issue #263 worktree). f32 rollout
    /// storage means ~1e-6 relative error is expected.
    const TOL: f32 = 1e-5;

    fn assert_slice_close(got: &[f32], expected: &[f32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < TOL, "{what}: mismatch at index {i}: got {g}, expected {e}");
        }
    }

    /// Reference test (a): 4-step single-env trajectory with `rho_bar = 1`,
    /// `c_bar = 1`. The importance ratios are
    /// `exp(target − behavior) = [1.34986, 0.60653, 1.34986, 0.67032]`, so
    /// rho is *clipped* to 1.0 at steps 0 and 2 (ratio > 1) and left
    /// unchanged at steps 1 and 3 — clipping is observable, not dead code.
    ///
    /// Expected values produced by `scripts/vtrace_ref.py` (Espeholt et al.
    /// 2018, Algorithm 1) on:
    /// ```text
    /// rewards   = [1.0, 0.0, -1.0, 2.0]
    /// values    = [0.5, 0.6,  0.7, 0.8]
    /// behavior  = [-0.5, -1.0, -0.7, -0.2]
    /// target    = [-0.2, -1.5, -0.4, -0.6]
    /// bootstrap = 0.9,  gamma = 0.99
    /// ```
    #[test]
    fn clipped_rho_matches_reference() {
        let rewards = [1.0_f32, 0.0, -1.0, 2.0];
        let values = [0.5_f32, 0.6, 0.7, 0.8];
        let behavior = [-0.5_f32, -1.0, -0.7, -0.2];
        let target = [-0.2_f32, -1.5, -0.4, -0.6];
        let terminated = [false, false, false, false];

        let mut vt = [0.0_f32; 4];
        let mut adv = [0.0_f32; 4];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &behavior,
            &target,
            &terminated,
            0.9,
            0.99,
            1.0,
            1.0,
            &mut vt,
            &mut adv,
        );

        // From scripts/vtrace_ref.py, Scenario B.
        let expected_vt = [1.9349601974_f32, 0.9444042398, 1.1796228241, 2.2016392163];
        let expected_adv = [1.4349601974_f32, 0.3444042398, 0.4796228241, 1.4016392163];
        assert_slice_close(&vt, &expected_vt, "vtrace_targets");
        assert_slice_close(&adv, &expected_adv, "advantages");
    }

    /// Reference test (b)+(e): same trajectory with `rho_bar = 0.5`,
    /// `c_bar = 1.5`. This exercises independent clipping:
    /// - rho is clipped to 0.5 at *every* step (all ratios exceed 0.5).
    /// - c is clipped to 1.5 nowhere (max ratio 1.35 < 1.5), so c follows the
    ///   raw ratios `[1.34986, 0.60653, 1.34986, 0.67032]`.
    ///
    /// If rho and c shared a single clip the outputs would differ; matching
    /// the reference confirms they are applied to their own thresholds.
    #[test]
    fn independent_rho_c_clipping_matches_reference() {
        let rewards = [1.0_f32, 0.0, -1.0, 2.0];
        let values = [0.5_f32, 0.6, 0.7, 0.8];
        let behavior = [-0.5_f32, -1.0, -0.7, -0.2];
        let target = [-0.2_f32, -1.5, -0.4, -0.6];
        let terminated = [false, false, false, false];

        let mut vt = [0.0_f32; 4];
        let mut adv = [0.0_f32; 4];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &behavior,
            &target,
            &terminated,
            0.9,
            0.99,
            0.5, // rho_bar
            1.5, // c_bar
            &mut vt,
            &mut adv,
        );

        // From scripts/vtrace_ref.py, Scenario C.
        let expected_vt = [1.8659718836_f32, 1.2128376703, 1.6431646095, 1.8455000000];
        let expected_adv = [0.8503546468_f32, 0.5133664817, 0.0635225000, 1.0455000000];
        assert_slice_close(&vt, &expected_vt, "vtrace_targets");
        assert_slice_close(&adv, &expected_adv, "advantages");

        // Cross-check: with rho_bar = c_bar = 1.0 the outputs are the
        // Scenario-B values, which differ from the above. This makes the
        // effect of the distinct thresholds explicit.
        let mut vt_b = [0.0_f32; 4];
        let mut adv_b = [0.0_f32; 4];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &behavior,
            &target,
            &terminated,
            0.9,
            0.99,
            1.0,
            1.0,
            &mut vt_b,
            &mut adv_b,
        );
        assert!(
            (vt_b[0] - vt[0]).abs() > 1e-3,
            "different (rho_bar, c_bar) must yield different targets"
        );
    }

    /// Reference test (c): on-policy recovery. With `behavior == target`
    /// every ratio is exactly 1, so `rho = c = 1`. In that regime the
    /// V-trace target is the standard n-step return and the V-trace
    /// advantage equals the GAE(`λ = 1`) advantage. We assert this against
    /// the existing, independently-tested `compute_gae_single_env` — an
    /// analytical identity, not a hand-typed constant.
    #[test]
    fn on_policy_recovers_nstep_returns() {
        let rewards = [1.0_f32, 0.0, -1.0, 2.0, 0.3];
        let values = [0.5_f32, 0.6, 0.7, 0.8, 0.4];
        let log_probs = [-0.5_f32, -1.0, -0.7, -0.2, -0.9];
        let terminated = [false, false, false, false, false];
        let bootstrap = 0.9_f32;
        let gamma = 0.99_f32;

        let mut vt = [0.0_f32; 5];
        let mut adv = [0.0_f32; 5];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &log_probs, // behavior
            &log_probs, // target == behavior => on-policy
            &terminated,
            bootstrap,
            gamma,
            1.0,
            1.0,
            &mut vt,
            &mut adv,
        );

        // GAE(lambda = 1.0) reference: returns == n-step returns, and the
        // advantage matches the V-trace advantage under rho = c = 1.
        let mut gae_adv = [0.0_f32; 5];
        let mut gae_ret = [0.0_f32; 5];
        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            bootstrap,
            gamma,
            1.0, // lambda = 1 => full n-step returns
            &mut gae_adv,
            &mut gae_ret,
        );

        assert_slice_close(&vt, &gae_ret, "vtrace_targets vs n-step returns");
        assert_slice_close(&adv, &gae_adv, "vtrace advantages vs GAE(lambda=1)");
    }

    /// Final-step precedence: a `terminated = true` on the *last* rollout
    /// row must still bootstrap from `bootstrap_value`, matching
    /// `compute_gae_single_env` (which checks `t == num_steps - 1` before
    /// the terminal flag). Without this alignment, on-policy V-trace would
    /// diverge from GAE exactly when an env terminates on the final rollout
    /// step — a subtle but real source of A2C instability. We assert
    /// element-wise equality against GAE(`λ = 1`) on-policy.
    #[test]
    fn terminal_last_step_bootstraps_like_gae() {
        let rewards = [1.0_f32, 0.5, -0.5, 2.0];
        let values = [0.4_f32, 0.6, 0.8, 0.3];
        let log_probs = [-0.5_f32, -1.0, -0.7, -0.2];
        // Interior terminal at step 1 AND a terminal on the final step 3.
        let terminated = [false, true, false, true];
        let bootstrap = 0.9_f32;
        let gamma = 0.99_f32;

        let mut vt = [0.0_f32; 4];
        let mut adv = [0.0_f32; 4];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &log_probs,
            &log_probs,
            &terminated,
            bootstrap,
            gamma,
            1.0,
            1.0,
            &mut vt,
            &mut adv,
        );

        let mut gae_adv = [0.0_f32; 4];
        let mut gae_ret = [0.0_f32; 4];
        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            bootstrap,
            gamma,
            1.0,
            &mut gae_adv,
            &mut gae_ret,
        );

        assert_slice_close(&vt, &gae_ret, "vtrace_targets vs GAE returns (terminal last step)");
        assert_slice_close(&adv, &gae_adv, "vtrace advantages vs GAE (terminal last step)");
    }

    /// Reference test (d): episode boundary. A `terminated = true` at
    /// step 1 of a 3-step trajectory must stop signal from step 2 leaking
    /// into steps 0-1.
    ///
    /// Expected values from `scripts/vtrace_ref.py`, Scenario D:
    /// ```text
    /// rewards   = [1.0, 0.5, -0.5]
    /// values    = [0.4, 0.6,  0.8]
    /// behavior  = [-0.5, -1.0, -0.7]
    /// target    = [-0.2, -1.5, -0.4]
    /// terminated= [false, true, false],  bootstrap = 0.9, gamma = 0.99
    /// ```
    #[test]
    fn episode_boundary_blocks_carryover() {
        let rewards = [1.0_f32, 0.5, -0.5];
        let values = [0.4_f32, 0.6, 0.8];
        let behavior = [-0.5_f32, -1.0, -0.7];
        let target = [-0.2_f32, -1.5, -0.4];
        let terminated = [false, true, false];

        let mut vt = [0.0_f32; 3];
        let mut adv = [0.0_f32; 3];
        compute_vtrace_single_env(
            &rewards,
            &values,
            &behavior,
            &target,
            &terminated,
            0.9,
            0.99,
            1.0,
            1.0,
            &mut vt,
            &mut adv,
        );

        let expected_vt = [1.5339534647_f32, 0.5393469340, 0.3910000000];
        let expected_adv = [1.1339534647_f32, -0.0606530660, -0.4090000000];
        assert_slice_close(&vt, &expected_vt, "vtrace_targets");
        assert_slice_close(&adv, &expected_adv, "advantages");

        // Direct no-leak assertion: perturbing step-2 reward across the
        // terminal boundary must leave steps 0 and 1 untouched.
        let rewards_perturbed = [1.0_f32, 0.5, 100.0];
        let mut vt2 = [0.0_f32; 3];
        let mut adv2 = [0.0_f32; 3];
        compute_vtrace_single_env(
            &rewards_perturbed,
            &values,
            &behavior,
            &target,
            &terminated,
            0.9,
            0.99,
            1.0,
            1.0,
            &mut vt2,
            &mut adv2,
        );
        assert!((vt2[0] - vt[0]).abs() < TOL, "step-0 target leaked across boundary");
        assert!((vt2[1] - vt[1]).abs() < TOL, "step-1 target leaked across boundary");
        assert!((adv2[0] - adv[0]).abs() < TOL, "step-0 advantage leaked across boundary");
    }

    /// Buffer-level `compute_vtrace_advantages` must reproduce the inner
    /// kernel's results and write them into `advantages`/`returns` in the
    /// `[step][env]` layout the trainer consumes.
    #[test]
    fn buffer_level_matches_single_env_kernel() {
        let num_steps = 4;
        let num_envs = 2;
        let obs_dim = 1;
        let mut buffer = RolloutBuffer::new(num_steps, num_envs, obs_dim);

        // Env 0: Scenario B trajectory. Env 1: a shifted copy so the two
        // columns are distinct.
        let rewards = [[1.0_f32, 0.0, -1.0, 2.0], [0.2, 0.4, 0.1, -0.3]];
        let values = [[0.5_f32, 0.6, 0.7, 0.8], [0.1, 0.2, 0.15, 0.05]];
        let behavior = [[-0.5_f32, -1.0, -0.7, -0.2], [-0.3, -0.9, -0.6, -0.4]];
        let target = [[-0.2_f32, -1.5, -0.4, -0.6], [-0.1, -1.1, -0.8, -0.2]];
        let last_values = [0.9_f32, 0.25];
        let gamma = 0.99_f32;
        let (rho_bar, c_bar) = (1.0_f32, 1.0_f32);

        for step in 0..num_steps {
            for env in 0..num_envs {
                buffer.add(
                    step,
                    env,
                    &[0.0],
                    0,
                    rewards[env][step],
                    values[env][step],
                    behavior[env][step], // stored as behavior log-prob
                    false,
                    false,
                );
            }
        }

        // target_log_probs as [num_steps][num_envs].
        let target_lp: Vec<Vec<f32>> = (0..num_steps)
            .map(|step| (0..num_envs).map(|env| target[env][step]).collect())
            .collect();

        compute_vtrace_advantages(&mut buffer, &target_lp, &last_values, gamma, rho_bar, c_bar);

        // Recompute per-env via the kernel and compare.
        for env in 0..num_envs {
            let mut vt = [0.0_f32; 4];
            let mut adv = [0.0_f32; 4];
            compute_vtrace_single_env(
                &rewards[env],
                &values[env],
                &behavior[env],
                &target[env],
                &[false; 4],
                last_values[env],
                gamma,
                rho_bar,
                c_bar,
                &mut vt,
                &mut adv,
            );
            for step in 0..num_steps {
                assert!(
                    (buffer.advantages()[step][env] - adv[step]).abs() < TOL,
                    "advantage mismatch env {env} step {step}"
                );
                assert!(
                    (buffer.returns()[step][env] - vt[step]).abs() < TOL,
                    "return mismatch env {env} step {step}"
                );
            }
        }
    }

    /// On-policy buffer path: `compute_vtrace_advantages` with
    /// `target == behavior` and `rho_bar = c_bar = 1` must match
    /// `compute_advantages` (GAE) with `lambda = 1.0` on the same buffer.
    /// This is the buffer-level analogue of the n-step recovery identity
    /// and the regression basis for the A2C learning-curve sanity check.
    #[test]
    fn buffer_on_policy_matches_gae_lambda_one() {
        let num_steps = 5;
        let num_envs = 2;
        let mut buf_vtrace = RolloutBuffer::new(num_steps, num_envs, 1);
        let mut buf_gae = RolloutBuffer::new(num_steps, num_envs, 1);

        let rewards = [[1.0_f32, 0.5, 0.2, -0.3, 0.7], [0.3, -0.4, 0.6, 0.1, -0.2]];
        let values = [[0.4_f32, 0.6, 0.8, 0.5, 0.35], [0.2, 0.1, 0.3, 0.4, 0.25]];
        let log_probs = [[-0.5_f32, -1.0, -0.7, -0.4, -0.9], [-0.3, -0.9, -0.6, -0.8, -0.5]];
        // Mixed terminations: env 0 has an interior boundary at step 2;
        // env 1 terminates on the *final* rollout row (exercises the
        // GAE-matching final-step bootstrap precedence). This mirrors the
        // realistic multi-env CartPole rollout where the two estimators
        // must stay identical on-policy.
        let terminated = [[false, false, true, false, false], [false, false, false, false, true]];
        let last_values = [0.7_f32, 0.5];
        let gamma = 0.99_f32;

        for step in 0..num_steps {
            for env in 0..num_envs {
                let args = (rewards[env][step], values[env][step], log_probs[env][step]);
                let term = terminated[env][step];
                buf_vtrace.add(step, env, &[0.0], 0, args.0, args.1, args.2, term, false);
                buf_gae.add(step, env, &[0.0], 0, args.0, args.1, args.2, term, false);
            }
        }

        // On-policy => target == behavior.
        let target_lp: Vec<Vec<f32>> = (0..num_steps)
            .map(|step| (0..num_envs).map(|env| log_probs[env][step]).collect())
            .collect();

        compute_vtrace_advantages(&mut buf_vtrace, &target_lp, &last_values, gamma, 1.0, 1.0);
        crate::buffer::rollout::gae::compute_advantages(&mut buf_gae, &last_values, gamma, 1.0);

        for step in 0..num_steps {
            for env in 0..num_envs {
                assert!(
                    (buf_vtrace.advantages()[step][env] - buf_gae.advantages()[step][env]).abs()
                        < TOL,
                    "advantage mismatch step {step} env {env}"
                );
                assert!(
                    (buf_vtrace.returns()[step][env] - buf_gae.returns()[step][env]).abs() < TOL,
                    "return mismatch step {step} env {env}"
                );
            }
        }
    }

    /// `target_log_probs` with the wrong number of rows must panic with a
    /// clear message.
    #[test]
    #[should_panic(expected = "target_log_probs")]
    fn wrong_target_log_probs_shape_panics() {
        let mut buffer = RolloutBuffer::new(4, 1, 1);
        // Only 2 rows for a 4-step buffer.
        let target_lp = vec![vec![0.0_f32], vec![0.0_f32]];
        compute_vtrace_advantages(&mut buffer, &target_lp, &[0.0], 0.99, 1.0, 1.0);
    }
}
