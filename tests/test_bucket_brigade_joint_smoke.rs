//! End-to-end smoke test: drive [`BucketBrigadeMaEnv`] through the joint
//! multi-agent PPO trainer for a few iterations on the versioned
//! `minimal_specialization-v1` scenario.
//!
//! This test exercises the acceptance criterion from upstream issue #55:
//!
//! > Can be used by a Thrust PPO trainer end-to-end on
//! > `minimal_specialization-v1` (smoke run only; full training is a
//! > separate issue)
//!
//! We do not assert on the learning curve — the test just verifies that
//! every wiring step works end to end (env construction from a versioned
//! ID, joint rollout collection, joint update with no aux loss, no
//! panics, no dim mismatches). A full training-quality run is tracked
//! separately.
//!
//! Run with:
//! `cargo test --features "training,env-bucket-brigade" --test
//! test_bucket_brigade_joint_smoke`.

#![cfg(all(feature = "env-bucket-brigade", feature = "training"))]

use tch::nn::OptimizerConfig;
use thrust_rl::{
    env::games::bucket_brigade::BucketBrigadeMaEnv,
    multi_agent::joint::{JointEnv, JointMultiAgentTrainer, JointStepResult, JointTrainerConfig},
    policy::multi_discrete_mlp::MultiDiscreteMlpPolicy,
};

/// Local `JointEnv` adapter (kept inline so this test doesn't depend on the
/// `train_p3` example). Mirrors the shape used in
/// `examples/games/bucket_brigade/train_p3.rs`.
struct BbJointEnv {
    inner: BucketBrigadeMaEnv,
}

impl JointEnv for BbJointEnv {
    fn reset_joint(&mut self, seed: Option<u64>) -> Vec<Vec<f32>> {
        self.inner.reset(seed)
    }
    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        let joint: Vec<[u8; 3]> = actions
            .iter()
            .map(|a| {
                assert_eq!(
                    a.len(),
                    3,
                    "BbJointEnv expects 3 action dims (house, mode, signal); got {}",
                    a.len()
                );
                [a[0] as u8, a[1] as u8, a[2] as u8]
            })
            .collect();
        let result = self.inner.step(&joint);
        JointStepResult {
            rewards: result.rewards,
            done: result.done,
            observations: result.observations,
        }
    }
}

#[test]
fn joint_ppo_smoke_runs_two_iterations_on_minimal_specialization_v1() {
    // Construct the env via the versioned-ID registry — the
    // reproducibility surface from bucket-brigade PR #379.
    let env_inner = BucketBrigadeMaEnv::from_scenario_id(
        "minimal_specialization-v1",
        None, // pick up frozen default (4 agents)
        Some(42),
    )
    .expect("registered scenario ID resolves");

    let num_agents = env_inner.num_agents();
    let obs_dim = env_inner.obs_dim();
    let action_dims: Vec<i64> = env_inner.action_dims();

    // One policy + one optimizer per agent. Tiny hidden_dim keeps the test
    // fast; numerical quality is out of scope here.
    let mut policies: Vec<MultiDiscreteMlpPolicy> = Vec::with_capacity(num_agents);
    let mut optimizers: Vec<tch::nn::Optimizer> = Vec::with_capacity(num_agents);
    for _ in 0..num_agents {
        let p = MultiDiscreteMlpPolicy::new(obs_dim as i64, action_dims.clone(), 16);
        let opt = tch::nn::Adam::default().build(p.var_store(), 1e-3).expect("Adam built");
        policies.push(p);
        optimizers.push(opt);
    }

    let trainer_config = JointTrainerConfig {
        num_agents,
        rollout_steps: 64,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_range: 0.2,
        clip_range_vf: 0.0,
        vf_coef: 0.5,
        ent_coef: 0.0,
        n_epochs: 1,
        minibatch_size: 32,
        max_grad_norm: 0.5,
        normalize_advantages: true,
    };
    let mut trainer =
        JointMultiAgentTrainer::new(policies, optimizers, trainer_config).expect("trainer built");

    let mut env = BbJointEnv { inner: env_inner };
    let initial = env.reset_joint(Some(42));
    let mut last_obs = initial[0].clone();

    // Two iterations: enough to exercise rollout collection + update
    // without taking long.
    for _ in 0..2 {
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);
        assert_eq!(
            rollout.rewards.len(),
            num_agents,
            "rollout must contain a rewards tensor per agent"
        );
        let _stats = trainer.update(&rollout, |_features| None).expect("update did not error");
    }
}
