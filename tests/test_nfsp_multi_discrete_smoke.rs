//! NFSP multi-discrete smoke test (issue #127).
//!
//! Verifies that the post-#127 NFSP trainer can drive a
//! `MultiDiscreteMlpBurnPolicy` end-to-end on a small factored env
//! without panicking. The pre-#127 trainer would panic at the
//! supervised AP-update step because it built the actions tensor at
//! shape `[mb, 1]` regardless of `num_action_dims`, which caused the
//! `MultiDiscreteMlpBurnPolicy::evaluate_actions` per-dim slice
//! `actions.slice([0..mb, i..i+1])` to go out of bounds when
//! `num_action_dims > 1`.
//!
//! # Env
//!
//! A minimal 2-agent toy env with `action_dims = [2, 3]` (factored
//! multi-discrete) and a 1-d agent-index observation. The reward is a
//! deterministic function of the joint action; the env's only job is to
//! exercise the multi-discrete reservoir / supervised-loss path through
//! the trainer.
//!
//! # Assertions
//!
//! 1. NFSP runs 2 outer iterations without panicking.
//! 2. Reservoirs grow under non-zero η (≥ 0 entries after the run, and
//!    `cumulative_br_pushes > 0` at η = 0.5).
//! 3. Reservoir entries carry length-2 action vectors (`num_action_dims`
//!    matches `MultiDiscreteMlpBurnPolicy::action_dims_joint`).
//! 4. Supervised AP losses are finite for every iteration where the reservoir
//!    was non-empty.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
};
use thrust_rl::{
    multi_agent::{
        JointTrainerConfig, NfspConfig, NfspTrainer,
        joint::{JointEnv, JointPolicy, JointStepResult},
    },
    policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

type B = Autodiff<NdArray<f32>>;

const OBS_DIM: usize = 1;
const NUM_AGENTS: usize = 2;
const HIDDEN_DIM: usize = 8;
/// Factored action cardinalities. Length > 1 is the regression target —
/// the pre-#127 trainer panicked for any `num_action_dims > 1`.
const ACTION_DIMS: [usize; 2] = [2, 3];

/// Minimal 2-agent factored-action env. Each `step_joint` returns a
/// deterministic per-agent reward derived from the joint action and
/// terminates after [`EPISODE_LEN`] steps.
#[derive(Debug, Clone)]
struct ToyMultiDiscreteEnv {
    step_idx: usize,
}

impl ToyMultiDiscreteEnv {
    const EPISODE_LEN: usize = 8;

    fn new() -> Self {
        Self { step_idx: 0 }
    }

    fn per_agent_obs() -> Vec<Vec<f32>> {
        (0..NUM_AGENTS)
            .map(|i| vec![i as f32 / (NUM_AGENTS.saturating_sub(1).max(1)) as f32])
            .collect()
    }
}

impl JointEnv for ToyMultiDiscreteEnv {
    fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
        self.step_idx = 0;
        Self::per_agent_obs()
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        debug_assert_eq!(actions.len(), NUM_AGENTS);
        for (i, a) in actions.iter().enumerate() {
            debug_assert_eq!(
                a.len(),
                ACTION_DIMS.len(),
                "agent {i} must supply a {}-d action vec, got {}",
                ACTION_DIMS.len(),
                a.len()
            );
            for (d, &v) in a.iter().enumerate() {
                debug_assert!(
                    v >= 0 && (v as usize) < ACTION_DIMS[d],
                    "agent {i} action dim {d} out of range [0, {})",
                    ACTION_DIMS[d]
                );
            }
        }

        // Reward: small deterministic function of own action coordinates.
        let rewards: Vec<f32> = actions
            .iter()
            .map(|a| {
                let head = a[0] as f32;
                let body = a[1] as f32;
                // Rewards favor (1, 2) for both agents — a symmetric
                // optimum. The exact shape doesn't matter for the
                // smoke test; what matters is that NFSP can complete
                // its outer loop with a multi-discrete policy.
                head * 0.5 + body * 0.25
            })
            .collect();

        self.step_idx += 1;
        let done = self.step_idx >= Self::EPISODE_LEN;
        JointStepResult { rewards, done, observations: Self::per_agent_obs() }
    }
}

#[allow(clippy::type_complexity)]
fn build_trainer(
    eta: f32,
    max_iterations: usize,
    seed: u64,
) -> NfspTrainer<
    B,
    MultiDiscreteMlpBurnPolicy<B>,
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MultiDiscreteMlpBurnPolicy<B>, B>,
    ToyMultiDiscreteEnv,
    impl Fn(&NdArrayDevice, u64) -> MultiDiscreteMlpBurnPolicy<B>,
    impl Fn() -> BurnOptimizer<
        B,
        MultiDiscreteMlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MultiDiscreteMlpBurnPolicy<B>, B>,
    >,
    impl Fn() -> ToyMultiDiscreteEnv,
> {
    let device: NdArrayDevice = Default::default();
    let nfsp_config = NfspConfig {
        max_iterations,
        anticipatory_param: eta,
        reservoir_capacity: 1_024,
        br_train_steps_per_iteration: 1,
        avg_policy_train_steps_per_iteration: 4,
        avg_policy_minibatch_size: 16,
        avg_policy_lr: 5e-3,
        avg_policy_min_reservoir_coverage: 0.0,
        br_reward_scale: 1.0,
        seed,
    };
    let joint_config = JointTrainerConfig {
        num_agents: NUM_AGENTS,
        rollout_steps: 64,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    NfspTrainer::new(
        nfsp_config,
        joint_config,
        device,
        |dev: &NdArrayDevice, seed: u64| {
            MultiDiscreteMlpBurnPolicy::<B>::new_seeded(
                OBS_DIM,
                ACTION_DIMS.to_vec(),
                HIDDEN_DIM,
                seed,
                dev,
            )
        },
        || {
            let inner = AdamConfig::new().init();
            BurnOptimizer::new(inner, 5e-3)
        },
        ToyMultiDiscreteEnv::new,
    )
    .expect("NfspTrainer::new should succeed for multi-discrete policy")
}

/// Primary regression: NFSP outer loop with a multi-discrete policy
/// runs 2 iterations end-to-end without panicking. The pre-#127
/// trainer panicked inside `train_average_policies` on the very first
/// non-empty AP step.
#[test]
fn test_nfsp_multi_discrete_runs_without_panic() {
    let mut trainer = build_trainer(0.5, 2, 7);
    let stats = trainer
        .run_silent()
        .expect("NFSP run with multi-discrete policy must not error");
    assert_eq!(stats.iterations.len(), 2, "expected 2 outer iterations");

    // Reservoirs grew (η = 0.5 → ~50% of rollout steps push).
    let total_pushes: usize = trainer.cumulative_br_pushes();
    assert!(
        total_pushes > 0,
        "expected non-zero BR pushes at η=0.5 over 2 iter × 64 rollout × {NUM_AGENTS} agents, got 0"
    );

    // Reservoir entries carry length-`ACTION_DIMS.len()` action vecs.
    let expected_num_dims = ACTION_DIMS.len();
    for i in 0..NUM_AGENTS {
        let reservoir = trainer.reservoir(i);
        if !reservoir.is_empty() {
            for (k, (_, a)) in reservoir.items().iter().enumerate() {
                assert_eq!(
                    a.len(),
                    expected_num_dims,
                    "agent {i} reservoir item {k}: action vec length {} != num_action_dims {}",
                    a.len(),
                    expected_num_dims
                );
                // Each component must be in-range for its dim.
                for (d, &v) in a.iter().enumerate() {
                    assert!(
                        v >= 0 && (v as usize) < ACTION_DIMS[d],
                        "agent {i} reservoir item {k} dim {d}: action {} out of range [0, {})",
                        v,
                        ACTION_DIMS[d]
                    );
                }
            }
        }
    }

    // Supervised AP losses, when present, are finite. (`None` is
    // allowed for an iteration where the reservoir was empty.)
    for (k, it) in stats.iterations.iter().enumerate() {
        for (i, loss) in it.avg_policy_loss.iter().enumerate() {
            if let Some(l) = loss {
                assert!(l.is_finite(), "iter {} agent {i} AP loss must be finite, got {l}", k + 1);
            }
        }
    }
}

/// Probes the BR policy's reported `action_dims_joint()` and confirms
/// it matches the factored shape passed at construction. This is the
/// source of truth `train_average_policies` uses to size its actions
/// tensor.
#[test]
fn test_multi_discrete_policy_action_dims_match_construction() {
    let trainer = build_trainer(0.5, 1, 11);
    for i in 0..NUM_AGENTS {
        let dims = trainer.br_policy(i).action_dims_joint();
        let expected: Vec<i64> = ACTION_DIMS.iter().map(|&d| d as i64).collect();
        assert_eq!(
            dims, expected,
            "agent {i} action_dims_joint mismatch: got {dims:?}, expected {expected:?}"
        );
    }
}
