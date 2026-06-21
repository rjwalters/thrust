//! PSRO smoke + convergence tests on the N-player matching-pennies env
//! (issue #124, AC item G).
//!
//! Two tiers, mirroring the SAC/A2C convention
//! (`tests/test_a2c_cartpole.rs`, `tests/test_sac_pendulum.rs`) and the
//! gating applied in issue #208:
//!
//! 1. **`psro_n4_smoke_runs_and_is_finite`** (always runs) — a tiny-budget (2
//!    outer iterations) check that the N-player PSRO trainer wires together
//!    end-to-end and produces FINITE, structurally-valid outputs (per-agent
//!    meta-Nash marginals are valid simplex distributions, the NashConv curve
//!    is finite and non-negative). Runs in well under a minute, so it stays on
//!    every CI run.
//!
//! 2. **`test_psro_n4_converges_to_uniform_on_majority_game`** and
//!    **`test_psro_n4_nashconv_trend_does_not_blow_up`** (`#[ignore]`) — the
//!    full-budget (5 outer iteration) convergence bars, kept verbatim.
//!
//! ## Why the heavy bars are `#[ignore]`d
//!
//! The N=4 PSRO convergence run trains the α-rank `population⁴` payoff
//! tensor (the final k=6 → `6^4 = 1296`-cell power iteration); on the CPU
//! `NdArray` backend this was the single dominant cost of the CI `Tests`
//! job (~68 min on Linux). It is kept opt-in:
//!
//! ```text
//! cargo test --release --features training \
//!     --test test_psro_n_player_matching_pennies -- --ignored
//! ```
//!
//! The fast smoke test above guarantees the trainer keeps working on every
//! CI run; the convergence tests are the periodic / release-gate check.
//!
//! ## Convergence target
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
    multi_agent::{
        AlphaRankMetaSolver, JointTrainerConfig, MetaSolver, PsroConfig, PsroStats, PsroTrainer,
    },
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

/// Concrete Burn optimizer the N=4 PSRO trainer uses for each
/// best-response policy. Lifted to a type alias so the shared
/// `run_psro_n4` helper can name `PsroTrainer`'s `O` parameter.
type Opt = burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, MlpBurnPolicy<B>, B>;

/// Build and run an N=4 PSRO trainer for `max_iterations` outer rounds
/// against the majority game, returning the trainer (for population /
/// meta-Nash inspection) and its run stats. Centralizes the trainer
/// wiring shared by the smoke and convergence tests so that only the
/// iteration budget differs between tiers.
#[allow(clippy::type_complexity)]
fn run_psro_n4(
    max_iterations: usize,
    seed: u64,
) -> (
    PsroTrainer<
        B,
        MlpBurnPolicy<B>,
        Opt,
        NPlayerMatchingPennies,
        impl Fn(&NdArrayDevice, u64) -> MlpBurnPolicy<B>,
        impl Fn() -> BurnOptimizer<B, MlpBurnPolicy<B>, Opt>,
        impl Fn() -> NPlayerMatchingPennies,
    >,
    PsroStats,
) {
    let num_agents = 4_usize;
    let device: NdArrayDevice = Default::default();
    let psro_config = PsroConfig {
        max_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 1,
        max_payoff_evals_per_iteration: None,
        seed,
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
    (trainer, stats)
}

/// Fast, always-on smoke test (issue #208): drive a tiny-budget (2 outer
/// iterations) N=4 PSRO run end-to-end and assert it produces FINITE,
/// structurally-valid outputs without asserting any convergence bar.
///
/// With `max_iterations = 2` the α-rank payoff cache only grows to
/// `3^4 = 81` cells (vs `6^4 = 1296` at the full 5-iteration budget),
/// so this runs in a fraction of a second on the CPU `NdArray` backend
/// while still exercising the full N-player PSRO control flow:
/// best-response training, the flat N-tensor payoff evaluation, the
/// `solve_n_player` α-rank meta-solve, and N-player NashConv.
#[test]
fn psro_n4_smoke_runs_and_is_finite() {
    let num_agents = 4_usize;
    let max_iterations = 2_usize;
    let (trainer, stats) = run_psro_n4(max_iterations, 19);

    assert_eq!(
        stats.iterations.len(),
        max_iterations,
        "smoke run should record {max_iterations} iterations"
    );

    // NashConv (== exploitability field for the N-player path) must be
    // finite and non-negative on every iteration.
    for it in &stats.iterations {
        assert!(
            it.exploitability.is_finite(),
            "NashConv must be finite, got {}",
            it.exploitability
        );
        assert!(
            it.exploitability >= 0.0,
            "NashConv must be non-negative, got {}",
            it.exploitability
        );
    }

    let final_meta_nash = stats.iterations.last().unwrap().meta_nash_per_agent.clone();
    assert_eq!(
        final_meta_nash.len(),
        num_agents,
        "final iteration should report N per-agent meta-Nash marginals"
    );

    // Each agent's meta-Nash-weighted action marginal must be a valid
    // distribution: finite, in `[0, 1]`, and summing to ~1.
    for agent in 0..num_agents {
        let marginal = meta_nash_action_marginal(
            trainer.populations(agent),
            &final_meta_nash[agent],
            agent,
            num_agents,
            &Default::default(),
        );
        assert_eq!(
            marginal.len(),
            NPlayerMatchingPennies::ACTION_DIM,
            "agent {agent} marginal must have ACTION_DIM entries"
        );
        let mut sum = 0.0_f32;
        for &p in &marginal {
            assert!(p.is_finite(), "agent {agent} marginal entry must be finite, got {p}");
            assert!((0.0..=1.0001).contains(&p), "agent {agent} marginal entry out of range: {p}");
            sum += p;
        }
        assert!(
            (sum - 1.0).abs() <= 1e-3,
            "agent {agent} marginal must sum to ~1, got {sum} on {marginal:?}"
        );
    }
}

/// AC item G (12 in the Curator's list): PSRO with α-rank on N=4
/// majority game converges to per-agent meta-Nash-weighted action
/// marginals within `0.10` TV from uniform `(0.5, 0.5)`.
///
/// `#[ignore]`d in CI per issue #208 — this is the dominant cost of the
/// `Tests` job (the `6^4 = 1296`-cell α-rank power iteration). Run with:
///
/// ```text
/// cargo test --release --features training \
///     --test test_psro_n_player_matching_pennies -- --ignored
/// ```
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
#[ignore = "multi-minute N=4 PSRO convergence run; opt in with --ignored (prefer --release)"]
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
    let (trainer, stats) = run_psro_n4(max_iterations, 19);
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
/// `#[ignore]`d in CI per issue #208 (full 5-iteration `6^4`-cell
/// budget). Run with `--ignored` (see the convergence test above).
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
#[ignore = "multi-minute N=4 PSRO convergence run; opt in with --ignored (prefer --release)"]
fn test_psro_n4_nashconv_trend_does_not_blow_up() {
    // Same compute envelope as the convergence test: 5 iterations →
    // 6^4 = 1296-cell payoff cache. Seed differs so the two tests
    // explore independent rollout-trajectory clouds.
    let max_iterations = 5_usize;
    let (_trainer, stats) = run_psro_n4(max_iterations, 41);
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
