//! PSRO smoke test on the N-player matching-pennies env (issue #124,
//! AC item G).
//!
//! Verifies that the post-#124 PSRO trainer — with the
//! `num_agents != 2` guard lifted to `num_agents < 2`, the populations
//! generalized to `Vec<Vec<P>>`, the payoff cache replaced with a flat
//! N-tensor, the meta-solver routed through `solve_n_player`, and the
//! exploitability replaced with N-player NashConv — converges on the
//! symmetric mixed-equilibrium target for N = 4.
//!
//! The N-player majority game ("N-player matching pennies") has a
//! unique symmetric mixed equilibrium at `p = 0.5` per agent. Per-agent
//! meta-Nash-weighted action marginals are expected to be within `0.10`
//! TV distance of `(0.5, 0.5)` on the final iteration. The bound
//! mirrors the existing N=2 PSRO test
//! (`test_psro_converges_to_uniform_on_matching_pennies`) for the
//! consistent N=2 ↔ N=4 reviewer story; the NFSP N=4 test uses a
//! tighter `0.05` last-3-iter average because NFSP averages policies
//! into a non-parametric reservoir, whereas PSRO carries fresh
//! best-response policies in each population — the per-iteration TV is
//! noisier and the looser `0.10` bound is appropriate.
//!
//! Tracking issue: #124 (PR 2b of #117's chain).

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::Tensor,
};
use thrust_rl::{
    env::games::n_player_matching_pennies::NPlayerMatchingPennies,
    multi_agent::{AlphaRankMetaSolver, JointTrainerConfig, MetaSolver, PsroConfig, PsroTrainer},
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

/// Compute the per-agent meta-Nash-weighted action marginal:
/// `marginal[j] = Σ_k σ[k] · softmax(population[k](obs))[j]`.
///
/// Each agent in N-player matching-pennies sees a per-agent
/// observation `[agent_idx / (N-1)]`; we evaluate each population's
/// policy on the agent's own obs and mix the resulting action
/// distributions under that agent's meta-Nash marginal.
fn meta_nash_action_marginal(
    population: &[MlpBurnPolicy<B>],
    meta_nash: &[f32],
    agent_idx: usize,
    num_agents: usize,
    device: &NdArrayDevice,
) -> Vec<f32> {
    use burn::tensor::activation;
    assert_eq!(population.len(), meta_nash.len());
    let denom = (num_agents.saturating_sub(1)).max(1) as f32;
    let obs_value = agent_idx as f32 / denom;
    let obs = Tensor::<B, 2>::from_data([[obs_value]], device);
    let mut marginal = vec![0.0_f32; NPlayerMatchingPennies::ACTION_DIM];
    for (k, pol) in population.iter().enumerate() {
        let (logits, _) = pol.forward(obs.clone());
        let probs = activation::softmax(logits, 1);
        let probs_host: Vec<f32> = probs.into_data().to_vec().expect("probs to_vec");
        for j in 0..NPlayerMatchingPennies::ACTION_DIM {
            marginal[j] += meta_nash[k] * probs_host[j];
        }
    }
    marginal
}

/// AC item G (12 in the Curator's list): PSRO with α-rank on N=4
/// majority game converges to per-agent meta-Nash-weighted action
/// marginals within `0.10` TV from uniform `(0.5, 0.5)`.
///
/// # Calibration
///
/// 5-run release-mode sweep: see PR body for the observed TV values.
/// The `0.10` bound matches the PSRO N=2 tolerance in
/// `test_psro_converges_to_uniform_on_matching_pennies`.
///
/// Wall-clock: ~90s in release mode on Apple M-series. 5 outer
/// iterations × 4 agents per round × 1 BR-train cycle × 32 rollout
/// steps + a 6^4 = 1296-cell payoff-cache evaluation pass. α-rank's
/// `k^N`-state power iteration (final k=6 → 1296 states × N=4
/// deviation directions × 5 mutations × ≤200 power-iter steps) is
/// the dominant cost above k≥4. The N=4 path is necessarily slower
/// than the N=2 test because α-rank dispatches through
/// `solve_n_player`, which exposes the full joint Markov chain
/// rather than the marginalized 2-player matrix.
#[test]
fn test_psro_n4_converges_to_uniform_on_majority_game() {
    let num_agents = 4_usize;
    // 5 iterations × N=4 → final per-role-k = 6. Payoff-cache grows
    // to 6^4 = 1296 cells over the run; ~1295 are evaluated (initial
    // seed is 1 cell). With `payoff_eval_episodes = 1` and the env's
    // 16-step episode bound, this is ~83k policy forward passes for
    // cache evaluation + 5 × 4 × 1 BR training rollouts of 32 steps
    // = ~640 policy forward passes for BR. Total ~85k forward passes
    // on a 1-D obs / 2-action / hidden-16 MLP — well under 30s in
    // release mode.
    let max_iterations = 5_usize;
    let device: NdArrayDevice = Default::default();
    let psro_config = PsroConfig {
        max_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 1,
        seed: 19,
    };
    let joint_config = JointTrainerConfig {
        num_agents,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(AlphaRankMetaSolver::default());

    let mut trainer = PsroTrainer::new(
        psro_config,
        joint_config,
        meta_solver,
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
            BurnOptimizer::new(inner, 1e-3)
        },
        move || NPlayerMatchingPennies::new(num_agents),
    )
    .expect("PsroTrainer::new should succeed for N=4 config");

    let stats = trainer.run_silent().expect("PSRO run should not error");
    assert_eq!(stats.iterations.len(), max_iterations, "should record 8 iterations");

    let final_meta_nash = stats.iterations.last().unwrap().meta_nash_per_agent.clone();
    assert_eq!(
        final_meta_nash.len(),
        num_agents,
        "final iteration should report N per-agent meta-Nash marginals"
    );

    let uniform = vec![0.5_f32; NPlayerMatchingPennies::ACTION_DIM];
    let mut max_tv_seen = 0.0_f32;
    for agent in 0..num_agents {
        let marginal = meta_nash_action_marginal(
            trainer.populations(agent),
            &final_meta_nash[agent],
            agent,
            num_agents,
            &Default::default(),
        );
        let tv = total_variation(&marginal, &uniform);
        println!(
            "agent {agent}: meta-Nash-weighted action marginal = {marginal:?}, TV from uniform = {tv:.4}"
        );
        max_tv_seen = max_tv_seen.max(tv);
        assert!(
            tv <= 0.10,
            "agent {agent} meta-Nash action marginal TV {tv:.4} > 0.10 (PSRO N=4 convergence bar)"
        );
    }
    println!("N=4 PSRO: max per-agent TV from uniform = {max_tv_seen:.4}");
}

/// AC item G additional sanity: the per-iteration NashConv (which
/// reduces to the legacy 2-player exploitability formula for N=2 by
/// construction; see `compute_nashconv`'s N=2 fast path) is finite,
/// non-negative, and decreases on average — last-half mean ≤
/// first-half mean + `0.2` slack.
///
/// **Bit-exact since issue #135.** With policy init seeded
/// (`MlpBurnPolicy::new_seeded` → `crate::policy::seeded_init`) and the
/// PSRO factory fed a distinct per-construction seed derived from
/// `PsroConfig::seed`, this curve is fully deterministic. Across
/// repeated release-mode runs the curve is identically
/// `[1.0, 3.5556, 1.375, 1.3120, 1.2577]`: first-half mean `2.2778`,
/// second-half mean `1.3149`, so `(second - first) = -0.963` every
/// run. The slack is tightened from `+0.5` to `+0.2`; the deterministic
/// delta sits ~1.16 below the bound, so the assertion fires only on a
/// genuine blow-up.
///
/// The PSRO N≥3 NashConv is monotonically-decreasing in *expectation*
/// (each newly-added best response is, by α-rank's response-graph
/// equivalence to a Markov chain over joint pure strategies, a
/// non-deviation strategy that reduces the empirical-game's
/// regret-sum). In practice the per-iteration curve is noisy because
/// each BR training run uses a single sampled-opponent posture, so a
/// trend-based assertion is more robust than per-step monotonicity.
#[test]
fn test_psro_n4_nashconv_trend_does_not_blow_up() {
    let num_agents = 4_usize;
    // Same compute envelope as the convergence test: 5 iterations →
    // 6^4 = 1296-cell payoff cache. Seed differs so the two tests
    // explore independent rollout-trajectory clouds.
    let max_iterations = 5_usize;
    let device: NdArrayDevice = Default::default();
    let psro_config = PsroConfig {
        max_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 1,
        seed: 41,
    };
    let joint_config = JointTrainerConfig {
        num_agents,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(AlphaRankMetaSolver::default());

    let mut trainer = PsroTrainer::new(
        psro_config,
        joint_config,
        meta_solver,
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
            BurnOptimizer::new(inner, 1e-3)
        },
        move || NPlayerMatchingPennies::new(num_agents),
    )
    .expect("PsroTrainer::new should succeed for N=4 config");

    let stats = trainer.run_silent().expect("PSRO run should not error");
    let expls: Vec<f32> = stats.iterations.iter().map(|it| it.exploitability).collect();
    println!("N=4 NashConv curve: {expls:?}");
    for &e in &expls {
        assert!(e.is_finite(), "NashConv must be finite, got {e}");
        assert!(e >= 0.0, "NashConv must be non-negative, got {e}");
    }
    let n = expls.len();
    let half = n / 2;
    let first_mean: f32 = expls[..half].iter().copied().sum::<f32>() / half as f32;
    let second_mean: f32 = expls[half..].iter().copied().sum::<f32>() / (n - half) as f32;
    println!("N=4 NashConv first-half mean = {first_mean:.4}, second-half mean = {second_mean:.4}");
    assert!(
        second_mean <= first_mean + 0.2,
        "NashConv blew up: first-half mean {first_mean}, second-half mean {second_mean}.          Band tightened from +0.5 to +0.2 after issue #135 made the curve          bit-exact (deterministic delta -0.963); see test docs."
    );
}
