//! End-to-end bit-exact reproducibility tests for PSRO and NFSP under
//! a fixed `PsroConfig::seed` / `NfspConfig::seed` (issue #135).
//!
//! These close the last determinism gap: with the seeded host-side
//! policy-init shim (`MlpBurnPolicy::new_seeded` /
//! `MlpBurnConfig::with_seed`, routed through
//! `crate::policy::seeded_init`) plumbed into the trainers' per-
//! construction seed derivation, two runs with the same seed must
//! produce **bit-identical** trajectories — not merely a bounded trend.
//!
//! The rollout/training RNG sites were already seeded in PRs
//! #113/#116/#122/#123/#125; before this PR the policy-init RNG (Burn
//! 0.21's unseedable `Initializer`) injected residual nondeterminism
//! that forced the matching-pennies tolerance bands to stay loose. With
//! init seeded the curves are reproducible to the last bit.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
};
use thrust_rl::{
    env::games::matching_pennies::MatchingPennies,
    multi_agent::{
        FictitiousPlayMetaSolver, JointTrainerConfig, MetaSolver, NfspConfig, NfspTrainer,
        PsroConfig, PsroTrainer,
    },
    policy::mlp::MlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

type B = Autodiff<NdArray<f32>>;

/// Run PSRO on matching pennies with the given seed and return the
/// per-iteration exploitability trace.
fn psro_exploitability_trace(seed: u64) -> Vec<f32> {
    let device: NdArrayDevice = Default::default();
    let psro_config = PsroConfig {
        max_iterations: 6,
        max_population_size: 50,
        br_train_steps_per_iteration: 2,
        payoff_eval_episodes: 4,
        max_payoff_evals_per_iteration: None,
        br_reward_scale: 1.0,
        seed,
        serialize_br_updates: true,
    };
    let joint_config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let mut trainer = PsroTrainer::new(
        psro_config,
        joint_config,
        Box::new(FictitiousPlayMetaSolver::new(500)) as Box<dyn MetaSolver>,
        device,
        |dev: &NdArrayDevice, s: u64| {
            MlpBurnPolicy::<B>::new_seeded(
                MatchingPennies::OBS_DIM,
                MatchingPennies::ACTION_DIM,
                16,
                s,
                dev,
            )
        },
        || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
        MatchingPennies::new,
    )
    .expect("PsroTrainer::new should succeed");
    let stats = trainer.run_silent().expect("PSRO run should not error");
    stats.iterations.iter().map(|it| it.exploitability).collect()
}

/// Two PSRO runs with the same seed produce a bit-identical
/// exploitability trace; a different seed produces a different trace.
#[test]
fn test_psro_exploitability_trace_is_bit_identical() {
    let a = psro_exploitability_trace(42);
    let b = psro_exploitability_trace(42);
    assert_eq!(a.len(), 6);
    assert_eq!(a, b, "same PsroConfig::seed must yield a bit-identical exploitability trace");

    let c = psro_exploitability_trace(43);
    assert_ne!(a, c, "different seed should change the trace");
}

/// Run NFSP on matching pennies with the given seed and return the
/// per-iteration average-policy action marginals (agent 0) as a flat
/// trace.
#[allow(clippy::type_complexity)]
fn nfsp_avg_marginal_trace(seed: u64) -> Vec<f32> {
    let device: NdArrayDevice = Default::default();
    let nfsp_config = NfspConfig {
        max_iterations: 6,
        anticipatory_param: 0.1,
        reservoir_capacity: 4_096,
        br_train_steps_per_iteration: 1,
        avg_policy_train_steps_per_iteration: 4,
        avg_policy_minibatch_size: 32,
        avg_policy_lr: 5e-3,
        avg_policy_min_reservoir_coverage: 0.0,
        br_reward_scale: 1.0,
        seed,
    };
    let joint_config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: 64,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let mut trainer: NfspTrainer<
        B,
        MlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
        MatchingPennies,
        _,
        _,
        _,
    > = NfspTrainer::new(
        nfsp_config,
        joint_config,
        device,
        |dev: &NdArrayDevice, s: u64| {
            MlpBurnPolicy::<B>::new_seeded(
                MatchingPennies::OBS_DIM,
                MatchingPennies::ACTION_DIM,
                16,
                s,
                dev,
            )
        },
        || BurnOptimizer::new(AdamConfig::new().init(), 5e-3),
        MatchingPennies::new,
    )
    .expect("NfspTrainer::new should succeed");
    let stats = trainer.run_silent().expect("NFSP run should not error");
    let mut trace = Vec::new();
    for it in &stats.iterations {
        if let Some(Some(m)) = it.avg_action_marginal.first() {
            trace.extend_from_slice(m);
        }
    }
    trace
}

/// Two NFSP runs with the same seed produce a bit-identical
/// average-policy marginal trace; a different seed differs.
#[test]
fn test_nfsp_avg_marginal_trace_is_bit_identical() {
    let a = nfsp_avg_marginal_trace(7);
    let b = nfsp_avg_marginal_trace(7);
    assert!(!a.is_empty(), "trace should be non-empty");
    assert_eq!(a, b, "same NfspConfig::seed must yield a bit-identical marginal trace");

    let c = nfsp_avg_marginal_trace(8);
    assert_ne!(a, c, "different seed should change the trace");
}

/// Golden-reference pin for the NFSP avg-policy action stream (issue #235).
///
/// The marginal trace is the empirical action distribution accumulated by
/// `NfspTrainer::action_marginal_for`'s seeded probe loop, which flows
/// through the **batched** seeded sampler
/// (`JointPolicy::get_actions_host_seeded_batched`) introduced in #235.
/// Pinning it to a literal captured from `main` *before* the batching
/// refactor proves the forward/sample decoupling + single-batched-forward
/// probe is **bit-for-bit** identical to the previous per-row batch-1
/// loop — the load-bearing determinism guarantee the issue requires.
///
/// Whereas `test_nfsp_avg_marginal_trace_is_bit_identical` only checks
/// run-vs-run equality at a fixed seed (which a *consistent* RNG-reordering
/// would still pass), this test would fail if the batched path changed the
/// action stream at all. If a future change intentionally alters the
/// stream, re-capture the literal and document the change.
#[test]
fn test_nfsp_avg_marginal_stream_matches_golden_reference() {
    // Reference captured on `main` @ 5beae49 (pre-#235) by running
    // `nfsp_avg_marginal_trace(7)` with the *old* per-probe batch-1
    // sampler. The #235 batched sampler reproduces it exactly.
    let golden: Vec<f32> = vec![
        0.46875_f32,
        0.53125_f32,
        0.5625_f32,
        0.4375_f32,
        0.5703125_f32,
        0.4296875_f32,
        0.625_f32,
        0.375_f32,
        0.5234375_f32,
        0.4765625_f32,
        0.5390625_f32,
        0.4609375_f32,
    ];
    let trace = nfsp_avg_marginal_trace(7);
    assert_eq!(
        trace, golden,
        "NFSP avg-marginal action stream (seed 7) must be bit-identical to the pre-#235 \
         reference; the batched seeded sampler must not change the RNG draw order"
    );
}
