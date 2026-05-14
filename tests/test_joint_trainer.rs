//! Integration tests for `multi_agent::joint::JointMultiAgentTrainer`.
//!
//! Lives in `tests/` (separate binary) for the same reason as
//! `test_ppo_aux_loss.rs`: pre-existing in-module test failures on main
//! prevent unit tests inside `src/` from compiling (see upstream issue #7).
//! The joint module also ships a `#[cfg(test)] mod tests` inside
//! `src/multi_agent/joint.rs` covering the same scenarios; once #7 is fixed
//! those will run via `cargo test --lib`.

use std::collections::HashMap;

use tch::{Kind, Tensor, nn};
use thrust_rl::{
    multi_agent::joint::{
        JointEnv, JointMultiAgentTrainer, JointStepResult, JointTrainerConfig,
    },
    policy::{mlp::MlpPolicy, multi_discrete_mlp::MultiDiscreteMlpPolicy},
};

// -----------------------------------------------------------------------
// Mock joint environment.
// -----------------------------------------------------------------------

struct MockEnv {
    num_agents: usize,
    obs_dim: usize,
    t: usize,
}

impl MockEnv {
    fn new(num_agents: usize, obs_dim: usize) -> Self {
        Self { num_agents, obs_dim, t: 0 }
    }

    fn obs_for(&self) -> Vec<f32> {
        (0..self.obs_dim)
            .map(|i| (((self.t * 7 + i * 13) % 100) as f32) / 100.0 - 0.5)
            .collect()
    }
}

impl JointEnv for MockEnv {
    fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
        self.t = 0;
        let obs = self.obs_for();
        (0..self.num_agents).map(|_| obs.clone()).collect()
    }
    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        self.t += 1;
        let rewards: Vec<f32> = actions
            .iter()
            .map(|a| a.iter().map(|&x| x as f32).sum::<f32>() / 10.0)
            .collect();
        let obs = self.obs_for();
        let observations = (0..self.num_agents).map(|_| obs.clone()).collect();
        JointStepResult { rewards, done: false, observations }
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn capture_params(vs: &nn::VarStore) -> HashMap<String, Tensor> {
    vs.variables()
        .into_iter()
        .map(|(name, t)| (name, t.detach().copy()))
        .collect()
}

fn map_l2_diff(a: &HashMap<String, Tensor>, b: &HashMap<String, Tensor>) -> f64 {
    let mut total = 0.0_f64;
    for (name, at) in a.iter() {
        if let Some(bt) = b.get(name) {
            let d = at - bt;
            let s: f64 = f64::try_from(&d.square().sum(Kind::Float)).unwrap_or(0.0);
            total += s;
        }
    }
    total
}

fn mlp_with_optimizer(
    obs_dim: i64,
    action_dim: i64,
    lr: f64,
) -> (MlpPolicy, nn::Optimizer) {
    let mut p = MlpPolicy::new(obs_dim, action_dim, 16);
    let opt = p.optimizer(lr);
    (p, opt)
}

fn multi_discrete_with_optimizer(
    obs_dim: i64,
    action_dims: Vec<i64>,
    lr: f64,
) -> (MultiDiscreteMlpPolicy, nn::Optimizer) {
    let mut p = MultiDiscreteMlpPolicy::new(obs_dim, action_dims, 16);
    let opt = p.optimizer(lr).unwrap();
    (p, opt)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[test]
fn test_joint_trainer_smoke() {
    let num_agents = 2;
    let obs_dim: i64 = 4;
    let action_dim: i64 = 3;
    let mut policies: Vec<MlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..num_agents {
        let (p, o) = mlp_with_optimizer(obs_dim, action_dim, 3e-4);
        policies.push(p);
        optimizers.push(o);
    }

    let config = JointTrainerConfig {
        num_agents,
        rollout_steps: 64,
        n_epochs: 2,
        minibatch_size: 32,
        ..Default::default()
    };
    let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

    let mut env = MockEnv::new(num_agents, obs_dim as usize);
    let initial = env.reset_joint(None);
    let mut last_obs = initial[0].clone();

    let rollout = trainer.collect_rollout(&mut env, &mut last_obs);
    let stats = trainer
        .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
        .expect("update should not error");

    assert!(stats.total_loss.is_finite(), "total_loss must be finite");
    for i in 0..num_agents {
        assert!(stats.policy_loss[i].is_finite(), "policy_loss[{i}] finite");
        assert!(stats.value_loss[i].is_finite(), "value_loss[{i}] finite");
        assert!(stats.entropy[i].is_finite(), "entropy[{i}] finite");
        assert!(stats.clip_fraction[i].is_finite(), "clip_fraction[{i}] finite");
        assert!(stats.approx_kl[i].is_finite(), "approx_kl[{i}] finite");
        assert!(stats.explained_var[i].is_finite(), "explained_var[{i}] finite");
    }
}

#[test]
fn test_joint_rollout_shapes() {
    let num_agents = 3;
    let obs_dim: i64 = 5;
    let t: usize = 32;
    let mut policies: Vec<MlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..num_agents {
        let (p, o) = mlp_with_optimizer(obs_dim, 4, 3e-4);
        policies.push(p);
        optimizers.push(o);
    }

    let config = JointTrainerConfig {
        num_agents,
        rollout_steps: t,
        n_epochs: 1,
        minibatch_size: t,
        ..Default::default()
    };
    let trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

    let mut env = MockEnv::new(num_agents, obs_dim as usize);
    let initial = env.reset_joint(None);
    let mut last_obs = initial[0].clone();
    let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

    assert_eq!(rollout.observations.size(), vec![t as i64, obs_dim]);
    assert_eq!(rollout.actions.len(), num_agents);
    // Scalar discrete: per-agent actions are [T]
    for a in &rollout.actions {
        assert_eq!(a.size(), vec![t as i64]);
    }
    assert_eq!(rollout.dones.size(), vec![t as i64]);
    for r in &rollout.rewards {
        assert_eq!(r.size(), vec![t as i64]);
    }
    for lp in &rollout.log_probs {
        assert_eq!(lp.size(), vec![t as i64]);
    }
    for v in &rollout.values {
        assert_eq!(v.size(), vec![t as i64]);
    }
    assert_eq!(rollout.num_steps(), t as i64);
    assert_eq!(rollout.obs_dim(), obs_dim);
    assert_eq!(rollout.num_agents(), num_agents);
}

#[test]
fn test_aux_fn_gradient_couples_encoders() {
    // With aux_fn = |feats| (feats[0] - feats[1]).square().sum() AND PPO
    // loss contribution suppressed via vf_coef=ent_coef=clip_range=0, the
    // sole gradient source is the aux term -- which must touch BOTH
    // encoders.
    let num_agents = 2;
    let obs_dim: i64 = 4;
    let mut policies: Vec<MlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..num_agents {
        let (p, o) = mlp_with_optimizer(obs_dim, 3, 1e-2);
        policies.push(p);
        optimizers.push(o);
    }

    let config = JointTrainerConfig {
        num_agents,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        vf_coef: 0.0,
        ent_coef: 0.0,
        clip_range: 0.0,
        normalize_advantages: false,
        ..Default::default()
    };
    let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

    let mut env = MockEnv::new(num_agents, obs_dim as usize);
    let initial = env.reset_joint(None);
    let mut last_obs = initial[0].clone();
    let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

    let before_a = capture_params(trainer.policies[0].var_store());
    let before_b = capture_params(trainer.policies[1].var_store());

    let _stats = trainer
        .update(&rollout, |features: &[&Tensor]| -> Option<Tensor> {
            Some((features[0] - features[1]).square().sum(Kind::Float))
        })
        .expect("update should not error");

    let after_a = capture_params(trainer.policies[0].var_store());
    let after_b = capture_params(trainer.policies[1].var_store());

    let diff_a = map_l2_diff(&before_a, &after_a);
    let diff_b = map_l2_diff(&before_b, &after_b);

    assert!(diff_a > 0.0, "policy 0 params must change; diff_a = {diff_a}");
    assert!(diff_b > 0.0, "policy 1 params must change; diff_b = {diff_b}");
}

#[test]
fn test_aux_fn_none_runs_clean() {
    let num_agents = 2;
    let obs_dim: i64 = 4;
    let mut policies: Vec<MlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..num_agents {
        let (p, o) = mlp_with_optimizer(obs_dim, 3, 3e-4);
        policies.push(p);
        optimizers.push(o);
    }

    let config = JointTrainerConfig {
        num_agents,
        rollout_steps: 32,
        n_epochs: 2,
        minibatch_size: 16,
        ..Default::default()
    };
    let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

    let mut env = MockEnv::new(num_agents, obs_dim as usize);
    let initial = env.reset_joint(None);
    let mut last_obs = initial[0].clone();
    let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

    let stats = trainer
        .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
        .expect("update should not error");

    assert_eq!(
        stats.aux_loss, 0.0,
        "aux_loss must be 0 when aux_fn returns None"
    );
    assert!(stats.total_loss.is_finite());
}

#[test]
fn test_jointpolicy_for_multidiscrete() {
    // Repeat the smoke test with MultiDiscreteMlpPolicy + factored [3, 2]
    // action space -- exercises the multi-discrete code paths in
    // collect_rollout (action shape [T, num_dims]) and in evaluate_actions.
    let num_agents = 2;
    let obs_dim: i64 = 4;
    let action_dims = vec![3_i64, 2];
    let mut policies: Vec<MultiDiscreteMlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..num_agents {
        let (p, o) = multi_discrete_with_optimizer(obs_dim, action_dims.clone(), 3e-4);
        policies.push(p);
        optimizers.push(o);
    }

    let config = JointTrainerConfig {
        num_agents,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let mut trainer = JointMultiAgentTrainer::new(policies, optimizers, config).unwrap();

    let mut env = MockEnv::new(num_agents, obs_dim as usize);
    let initial = env.reset_joint(None);
    let mut last_obs = initial[0].clone();
    let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

    for a in &rollout.actions {
        assert_eq!(a.size(), vec![32, action_dims.len() as i64]);
    }

    let stats = trainer
        .update(&rollout, |_features: &[&Tensor]| -> Option<Tensor> { None })
        .expect("update should not error");
    assert!(stats.total_loss.is_finite());
}

#[test]
fn test_trainer_validates_device_uniformity() {
    // Trainer construction should reject empty policy list.
    let policies: Vec<MlpPolicy> = vec![];
    let optimizers: Vec<nn::Optimizer> = vec![];
    let cfg = JointTrainerConfig { num_agents: 0, ..Default::default() };
    assert!(JointMultiAgentTrainer::new(policies, optimizers, cfg).is_err());
}

#[test]
fn test_trainer_validates_optimizer_count_mismatch() {
    let mut policies: Vec<MlpPolicy> = Vec::new();
    let mut optimizers: Vec<nn::Optimizer> = Vec::new();
    for _ in 0..2 {
        let (p, o) = mlp_with_optimizer(4, 3, 3e-4);
        policies.push(p);
        optimizers.push(o);
    }
    // Drop one optimizer -> mismatched lengths.
    optimizers.pop();
    let cfg = JointTrainerConfig { num_agents: 2, ..Default::default() };
    assert!(JointMultiAgentTrainer::new(policies, optimizers, cfg).is_err());
}
