//! NFSP smoke test on the N-player matching-pennies env (issue #119, AC item
//! E).
//!
//! Verifies that the post-#119 NFSP trainer — with the
//! `num_agents != 2` guard lifted and the hardcoded `let num_agents = 2usize;`
//! constants lifted to `self.joint_config.num_agents` — still
//! converges on the symmetric mixed-equilibrium target for N = 4.
//!
//! The N-player majority game ("N-player matching pennies") has a
//! unique symmetric mixed equilibrium at `p = 0.5` per agent.
//! Per-agent average-policy (AP) marginals are expected to be within
//! `0.05` TV distance of `(0.5, 0.5)` averaged over the last 3 outer
//! iterations (post-iteration convergence rather than final-iteration
//! to absorb the BR/AP oscillation).
//!
//! ## CI gating (issue #208)
//!
//! A fast always-on smoke test
//! (`nfsp_n4_smoke_runs_and_is_finite`, tiny 2-iteration budget) keeps
//! the trainer wired up on every CI run. The full-budget convergence
//! (AC E) and η-mixing aggregate tests are `#[ignore]`d and run on
//! demand:
//!
//! ```text
//! cargo test --release --features training \
//!     --test test_nfsp_n_player_matching_pennies -- --ignored
//! ```
//!
//! Tracking issue: #119 (PR 2 of #117's chain).

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
};
use thrust_rl::{
    env::games::n_player_matching_pennies::NPlayerMatchingPennies,
    multi_agent::{JointTrainerConfig, NfspConfig, NfspTrainer},
    policy::mlp::MlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

type B = Autodiff<NdArray<f32>>;

/// Total variation distance between two distributions of the same length.
fn total_variation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        s += (x - y).abs();
    }
    s * 0.5
}

#[allow(clippy::type_complexity)]
fn build_n_player_trainer(
    num_agents: usize,
    max_iterations: usize,
    eta: f32,
    seed: u64,
) -> NfspTrainer<
    B,
    MlpBurnPolicy<B>,
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
    NPlayerMatchingPennies,
    impl Fn(&NdArrayDevice, u64) -> MlpBurnPolicy<B>,
    impl Fn() -> BurnOptimizer<
        B,
        MlpBurnPolicy<B>,
        burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>,
    >,
    impl Fn() -> NPlayerMatchingPennies,
> {
    let device: NdArrayDevice = Default::default();
    let nfsp_config = NfspConfig {
        max_iterations,
        anticipatory_param: eta,
        reservoir_capacity: 8_192,
        br_train_steps_per_iteration: 1,
        avg_policy_train_steps_per_iteration: 8,
        avg_policy_minibatch_size: 64,
        avg_policy_lr: 5e-3,
        seed,
    };
    let joint_config = JointTrainerConfig {
        num_agents,
        rollout_steps: 128,
        n_epochs: 1,
        minibatch_size: 64,
        ..Default::default()
    };
    NfspTrainer::new(
        nfsp_config,
        joint_config,
        device,
        |dev: &NdArrayDevice, seed: u64| {
            MlpBurnPolicy::<B>::new_seeded(
                NPlayerMatchingPennies::OBS_DIM,
                NPlayerMatchingPennies::ACTION_DIM,
                16,
                seed,
                dev,
            )
        },
        || {
            let inner = AdamConfig::new().init();
            BurnOptimizer::new(inner, 5e-3)
        },
        move || NPlayerMatchingPennies::new(num_agents),
    )
    .expect("NfspTrainer::new should succeed for N-player config")
}

/// Fast, always-on smoke test (issue #208): a tiny-budget (2 outer
/// iterations) N=4 NFSP run executes end-to-end and produces FINITE,
/// structurally-valid outputs — every per-agent AP action marginal is
/// a valid simplex distribution. No convergence bar.
#[test]
fn nfsp_n4_smoke_runs_and_is_finite() {
    let num_agents = 4_usize;
    let max_iterations = 2_usize;
    let mut trainer = build_n_player_trainer(num_agents, max_iterations, 0.1, 19);
    let stats = trainer.run_silent().expect("NFSP run should not error");
    assert_eq!(stats.iterations.len(), max_iterations, "smoke run records 2 iterations");

    let last = stats.iterations.last().unwrap();
    for agent in 0..num_agents {
        let m = last.avg_action_marginal[agent]
            .as_ref()
            .expect("AP marginal should be present on N-player matching pennies");
        let mut sum = 0.0_f32;
        for &p in m {
            assert!(p.is_finite(), "agent {agent} AP marginal entry finite, got {p}");
            assert!(
                (0.0..=1.0001).contains(&p),
                "agent {agent} AP marginal entry out of range: {p}"
            );
            sum += p;
        }
        assert!((sum - 1.0).abs() <= 1e-3, "agent {agent} AP marginal must sum to ~1, got {sum}");
    }
}

/// AC item E: NFSP per-agent AP marginals converge to `(0.5, 0.5)`
/// within `0.05` TV averaged over the last 3 outer iterations.
///
/// `#[ignore]`d in CI per issue #208 (run with `--ignored`).
///
/// **Calibration**: The 5-run release-mode sweep (`cargo test
/// --features training --release --test test_nfsp_n_player_matching_pennies`)
/// produced last-3-iteration average TV in `[0.018, 0.044]` for every
/// agent. The 0.05 bound gives ~14% headroom; the issue body's stated
/// `0.05` bar is therefore met with the documented seed and
/// hyperparameter set.
#[test]
#[ignore = "multi-iteration NFSP convergence run; opt in with --ignored (prefer --release)"]
fn test_nfsp_n4_converges_to_uniform_on_majority_game() {
    let num_agents = 4_usize;
    let max_iterations = 10_usize;
    let mut trainer = build_n_player_trainer(num_agents, max_iterations, 0.1, 19);
    let stats = trainer.run_silent().expect("NFSP run should not error");
    assert_eq!(stats.iterations.len(), max_iterations);

    let uniform = vec![0.5_f32; NPlayerMatchingPennies::ACTION_DIM];
    let last_k = 3_usize;
    let window_start = stats.iterations.len().saturating_sub(last_k);

    for agent in 0..num_agents {
        // Average TV from uniform over the last `last_k` AP marginals
        // for this agent.
        let mut sum_tv = 0.0_f32;
        let mut count = 0_usize;
        for it in &stats.iterations[window_start..] {
            let m = it.avg_action_marginal[agent]
                .as_ref()
                .expect("AP marginal should be present on N-player matching pennies");
            sum_tv += total_variation(m, &uniform);
            count += 1;
        }
        let avg_tv = sum_tv / count as f32;
        println!("agent {agent}: last-{last_k} avg AP TV from uniform = {avg_tv:.4}");
        assert!(
            avg_tv <= 0.05,
            "agent {agent} AP marginal TV averaged over last {last_k} iters = {avg_tv}; \
             must be <= 0.05 (NFSP convergence bar for N=4 majority game)"
        );
    }
}

/// Reservoir / push-rate sanity: with η = 0.1 and 128 rollout steps
/// per iteration over 10 iterations, the total BR push count across
/// all 4 agents should be `4 × 128 × 10 × 0.1 ≈ 512 ± 4σ`.
///
/// `#[ignore]`d in CI per issue #208 (run with `--ignored`).
#[test]
#[ignore = "multi-iteration NFSP run; opt in with --ignored (prefer --release)"]
fn test_nfsp_n4_eta_mixing_rate_in_aggregate() {
    let num_agents = 4_usize;
    let max_iterations = 10_usize;
    let eta = 0.1_f32;
    let mut trainer = build_n_player_trainer(num_agents, max_iterations, eta, 41);
    let _ = trainer.run_silent().expect("NFSP run should not error");
    let total_steps_per_agent = 128 * max_iterations;
    let total_steps = (total_steps_per_agent * num_agents) as f64;
    let p_emp = trainer.cumulative_br_pushes() as f64 / total_steps;
    let p_target = eta as f64;
    let std = (p_target * (1.0 - p_target) / total_steps).sqrt();
    let tol = 4.0 * std;
    println!(
        "N=4 NFSP cumulative BR pushes = {}, p_emp = {:.4}, p_target = {:.4}",
        trainer.cumulative_br_pushes(),
        p_emp,
        p_target
    );
    assert!(
        (p_emp - p_target).abs() <= tol,
        "η-mixing rate for N=4 deviates: p_emp={p_emp:.4}, p_target={p_target:.4}, tol={tol:.4}"
    );
}
