//! Tests for `PPOTrainer::train_step_with_aux`.
//!
//! Lives in `tests/` (separate binary) for the same reason as
//! `test_encoder_features.rs`: pre-existing in-module test failures on main
//! prevent unit tests in `src/train/ppo/trainer.rs` from compiling. See
//! upstream issue #7.

use tch::{Device, Kind, Tensor};
use thrust_rl::{
    policy::mlp::MlpPolicy,
    train::ppo::{PPOConfig, PPOTrainer},
};

fn build_trainer(
    obs_dim: i64,
    action_dim: i64,
) -> (PPOTrainer<MlpPolicy>, MlpPolicy, Device) {
    let mut policy = MlpPolicy::new(obs_dim, action_dim, 64);
    let device = policy.device();
    let opt = policy.optimizer(3e-4);

    // The trainer wants to own *a* policy for type inference but the closure
    // is what actually evaluates the live one. Use a throwaway dummy.
    let dummy = MlpPolicy::new(obs_dim, action_dim, 64);
    let mut trainer = PPOTrainer::new(PPOConfig::default(), dummy).unwrap();
    trainer.set_optimizer(opt);
    (trainer, policy, device)
}

fn make_batch(
    batch: i64,
    obs_dim: i64,
    action_dim: i64,
    device: Device,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let observations = Tensor::randn([batch, obs_dim], (Kind::Float, device));
    let actions = Tensor::randint(action_dim, [batch], (Kind::Int64, device));
    let old_log_probs = Tensor::randn([batch], (Kind::Float, device)) * 0.1;
    let old_values = Tensor::zeros([batch], (Kind::Float, device));
    let advantages = Tensor::randn([batch], (Kind::Float, device));
    let returns = Tensor::randn([batch], (Kind::Float, device));
    (observations, actions, old_log_probs, old_values, advantages, returns)
}

#[test]
fn train_step_unchanged_when_aux_is_none() {
    // `train_step` is supposed to delegate to `train_step_with_aux` with a
    // no-op aux closure. Verify by running it and checking
    // TrainingStats::aux_loss is zero.
    let (mut trainer, policy, device) = build_trainer(4, 2);
    let (obs, actions, old_lp, old_v, adv, ret) = make_batch(32, 4, 2, device);

    let stats = trainer
        .train_step(&obs, &actions, &old_lp, &old_v, &adv, &ret, |o, a| {
            policy.evaluate_actions(o, a)
        })
        .unwrap();

    assert_eq!(stats.aux_loss, 0.0);
}

#[test]
fn train_step_with_aux_records_aux_loss() {
    let (mut trainer, policy, device) = build_trainer(4, 2);
    let (obs, actions, old_lp, old_v, adv, ret) = make_batch(32, 4, 2, device);

    // Caller-supplied aux: a fixed-magnitude penalty on the obs norm. Doesn't
    // depend on the policy, just exercises the plumbing.
    let stats = trainer
        .train_step_with_aux(
            &obs,
            &actions,
            &old_lp,
            &old_v,
            &adv,
            &ret,
            |o, a| policy.evaluate_actions(o, a),
            |mb_obs| {
                let pen = mb_obs.pow_tensor_scalar(2).mean(Kind::Float) * 0.5;
                Some(pen)
            },
        )
        .unwrap();

    assert!(
        stats.aux_loss > 0.0,
        "aux_loss should be recorded, got {}",
        stats.aux_loss
    );
}

#[test]
fn aux_loss_drives_features_toward_zero() {
    // A large aux term that pulls features toward zero (||features||^2) should
    // measurably shrink the trunk activations after one update.
    let (mut trainer, policy, device) = build_trainer(4, 2);
    let (obs, actions, old_lp, old_v, adv, ret) = make_batch(64, 4, 2, device);

    let features_before = policy.encoder_features(&obs).abs().sum(Kind::Float);
    let mass_before: f64 = f64::try_from(&features_before).unwrap();

    let _stats = trainer
        .train_step_with_aux(
            &obs,
            &actions,
            &old_lp,
            &old_v,
            &adv,
            &ret,
            |o, a| policy.evaluate_actions(o, a),
            |mb_obs| {
                // 100 * mean(features^2) --- aggressive shrink term.
                let feats = policy.encoder_features(mb_obs);
                Some(feats.pow_tensor_scalar(2).mean(Kind::Float) * 100.0)
            },
        )
        .unwrap();

    let features_after = policy.encoder_features(&obs).abs().sum(Kind::Float);
    let mass_after: f64 = f64::try_from(&features_after).unwrap();

    assert!(
        mass_after < mass_before,
        "aux training did not shrink features (before {}, after {})",
        mass_before,
        mass_after
    );
}
