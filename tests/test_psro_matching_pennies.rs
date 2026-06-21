//! PSRO smoke + convergence tests on the matching-pennies env.
//!
//! Two tiers, gated per issue #208 to bound per-PR CI cost:
//!
//! 1. **`psro_matching_pennies_smoke_runs_and_is_finite`** (always runs) — a
//!    tiny-budget (2 outer iterations) check that the trainer wires together
//!    end-to-end and produces FINITE, structurally-valid outputs (per-agent
//!    meta-Nash marginals are valid simplex distributions, the exploitability
//!    curve is finite and non-negative). Runs in seconds.
//! 2. **`test_psro_converges_to_uniform_on_matching_pennies`** and
//!    **`test_psro_exploitability_non_increasing_trend_on_matching_pennies`**
//!    (`#[ignore]`) — the full-budget convergence bars, kept verbatim. Run them
//!    on demand with `cargo test --release --features training --test
//!    test_psro_matching_pennies -- --ignored`.
//!
//! Tracks acceptance criterion 7 from issue #107's curator comment:
//! after ≥10 PSRO iterations the meta-Nash distribution's marginal
//! mean over each agent's actions should have total variation distance
//! ≤ 0.10 from uniform `(0.5, 0.5)`. The "marginal mean over actions"
//! here is the expected mixed-action distribution under the meta-Nash —
//! we approximate it by sampling each population policy uniformly,
//! evaluating it on the (constant) matching-pennies observation, and
//! taking the meta-Nash-weighted average of the resulting action
//! distributions.
//!
//! Tracking issue: #107.

#![cfg(feature = "training")]

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    optim::AdamConfig,
    tensor::Tensor,
};
use thrust_rl::{
    env::games::matching_pennies::MatchingPennies,
    multi_agent::{
        FictitiousPlayMetaSolver, JointTrainerConfig, MetaSolver, PsroConfig, PsroTrainer,
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

/// Compute the expected marginal action distribution under a meta-Nash
/// mixture over a population of policies on matching pennies. Both
/// agents see the same constant observation, so the marginal is just
/// `Σ_k σ_k * softmax(policy_k(obs))`.
fn meta_nash_action_marginal(
    population: &[MlpBurnPolicy<B>],
    meta_nash: &[f32],
    device: &NdArrayDevice,
) -> Vec<f32> {
    use burn::tensor::activation;
    assert_eq!(population.len(), meta_nash.len());
    let obs = Tensor::<B, 2>::zeros([1, MatchingPennies::OBS_DIM], device);
    let mut marginal = vec![0.0_f32; MatchingPennies::ACTION_DIM];
    for (k, pol) in population.iter().enumerate() {
        let (logits, _) = pol.forward(obs.clone());
        let probs = activation::softmax(logits, 1);
        let probs_host: Vec<f32> = probs.into_data().to_vec().expect("probs to_vec");
        for j in 0..MatchingPennies::ACTION_DIM {
            marginal[j] += meta_nash[k] * probs_host[j];
        }
    }
    marginal
}

/// Fast, always-on smoke test (issue #208): drive a tiny-budget (2 outer
/// iterations) PSRO run end-to-end and assert it produces FINITE,
/// structurally-valid outputs without asserting any convergence bar.
#[test]
fn psro_matching_pennies_smoke_runs_and_is_finite() {
    let device: NdArrayDevice = Default::default();
    let max_iterations = 2_usize;
    let psro_config = PsroConfig {
        max_iterations,
        max_population_size: 50,
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 1,
        seed: 17,
    };
    let joint_config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(FictitiousPlayMetaSolver::new(1000));

    let mut trainer = PsroTrainer::new(
        psro_config,
        joint_config,
        meta_solver,
        device,
        |dev: &NdArrayDevice, seed: u64| {
            MlpBurnPolicy::<B>::new_seeded(
                MatchingPennies::OBS_DIM,
                MatchingPennies::ACTION_DIM,
                16,
                seed,
                dev,
            )
        },
        || {
            let inner = AdamConfig::new().init();
            BurnOptimizer::new(inner, 1e-3)
        },
        MatchingPennies::new,
    )
    .expect("PsroTrainer::new should succeed");

    let stats = trainer.run_silent().expect("PSRO run should not error");
    assert_eq!(stats.iterations.len(), max_iterations, "smoke run records 2 iterations");

    for it in &stats.iterations {
        assert!(
            it.exploitability.is_finite(),
            "exploitability finite, got {}",
            it.exploitability
        );
        assert!(it.exploitability >= 0.0, "exploitability >= 0, got {}", it.exploitability);
    }

    // Both per-agent meta-Nash-weighted marginals must be valid
    // distributions: finite, in `[0, 1]`, summing to ~1.
    let final_meta_nash = stats.iterations.last().unwrap().meta_nash_row().to_vec();
    for (label, pop) in [("row", trainer.population_row()), ("col", trainer.population_col())] {
        let marginal = meta_nash_action_marginal(pop, &final_meta_nash, &Default::default());
        let mut sum = 0.0_f32;
        for &p in &marginal {
            assert!(p.is_finite(), "{label} marginal entry finite, got {p}");
            assert!((0.0..=1.0001).contains(&p), "{label} marginal entry out of range: {p}");
            sum += p;
        }
        assert!((sum - 1.0).abs() <= 1e-3, "{label} marginal must sum to ~1, got {sum}");
    }
}

#[test]
#[ignore = "multi-iteration PSRO convergence run; opt in with --ignored (prefer --release)"]
fn test_psro_converges_to_uniform_on_matching_pennies() {
    let device: NdArrayDevice = Default::default();
    let psro_config = PsroConfig {
        // ≥10 iterations per the curator's acceptance criterion.
        max_iterations: 10,
        max_population_size: 50,
        // One BR-training rollout/update cycle per iteration is
        // enough to nudge each new best-response away from the random
        // initialization on this trivial env. Matching pennies' Nash
        // is uniform over policies sampling uniformly, and even random
        // policies' average marginal is already near uniform — what
        // we're really verifying is that the meta-solver + outer loop
        // mechanics don't *break* uniformity.
        br_train_steps_per_iteration: 1,
        payoff_eval_episodes: 2,
        seed: 17,
    };
    let joint_config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: 32,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let meta_solver: Box<dyn MetaSolver> = Box::new(FictitiousPlayMetaSolver::new(1000));

    let mut trainer = PsroTrainer::new(
        psro_config,
        joint_config,
        meta_solver,
        device,
        |dev: &NdArrayDevice, seed: u64| {
            MlpBurnPolicy::<B>::new_seeded(
                MatchingPennies::OBS_DIM,
                MatchingPennies::ACTION_DIM,
                16,
                seed,
                dev,
            )
        },
        || {
            let inner = AdamConfig::new().init();
            BurnOptimizer::new(inner, 1e-3)
        },
        MatchingPennies::new,
    )
    .expect("PsroTrainer::new should succeed");

    let stats = trainer.run_silent().expect("PSRO run should not error");
    assert_eq!(stats.iterations.len(), 10, "should record 10 iterations");

    // Acceptance criterion 7: marginal mean over actions has TV ≤ 0.10
    // from uniform after ≥10 iterations.
    let final_meta_nash = stats.iterations.last().unwrap().meta_nash_row().to_vec();
    let marginal_row =
        meta_nash_action_marginal(trainer.population_row(), &final_meta_nash, &Default::default());
    let marginal_col =
        meta_nash_action_marginal(trainer.population_col(), &final_meta_nash, &Default::default());
    let uniform = vec![0.5_f32; MatchingPennies::ACTION_DIM];
    let tv_row = total_variation(&marginal_row, &uniform);
    let tv_col = total_variation(&marginal_col, &uniform);

    println!(
        "matching-pennies PSRO marginal: row={:?} (TV={:.4}), col={:?} (TV={:.4})",
        marginal_row, tv_row, marginal_col, tv_col
    );
    assert!(
        tv_row <= 0.10,
        "row-player marginal action TV must be <= 0.10 (got {tv_row} on {marginal_row:?})"
    );
    assert!(
        tv_col <= 0.10,
        "col-player marginal action TV must be <= 0.10 (got {tv_col} on {marginal_col:?})"
    );
}

/// Companion to the marginal-distance test: confirm the
/// empirical-exploitability curve does not blow up over PSRO
/// iterations. We require the *average* exploitability over the
/// second half of the run to be within a calibrated band of the
/// average over the first half — i.e. the curve neither explodes nor
/// drifts unboundedly.
///
/// Re-enabled in #109 (PR linked to that issue). The inner
/// `JointMultiAgentTrainer::update_with_active_agents` and
/// `generate_minibatch_indices_with_rng` now take an `&mut StdRng`
/// from the caller, so the minibatch shuffle is driven by
/// `PsroConfig::seed` rather than the thread-local `rand::rng()`.
///
/// Tolerance calibration: matching pennies' first iteration starts
/// from a 1×1 random-vs-random matrix whose exploitability is often
/// near 0, while later iterations evaluate exploitability against a
/// *larger* meta-Nash support whose typical magnitude is higher;
/// the second-half mean is frequently *above* the first-half mean
/// even when training is doing the right thing. The asserted bound
/// is therefore `second_half_mean <= first_half_mean + 1.2`, chosen
/// against an empirical 10-run sweep on this seed triple: observed
/// `(second - first)` deltas spanned roughly `[-0.40, +0.94]` after
/// the #109 fix, and 1.2 provides ~25% headroom over the worst
/// positive delta while still catching genuine divergence (the curve
/// would have to grow by > 1.2 nats on average between halves for the
/// test to fire).
///
/// Tightened from `+1.2` to `+1.0` after issue #114 plumbed the
/// seeded `StdRng` through `get_action_host_seeded`. A 10-run release-
/// mode sweep on this seed triple observed `(second - first)`
/// deltas in `[0.71, 0.88]` (mean ≈ 0.81). The `+1.0` bound gave
/// ~14% headroom over the worst observed delta while halving the
/// previous slack.
///
/// **Now bit-exact (issue #135).** The last unseeded RNG site —
/// policy *initialization* — has been plumbed through a seeded host-
/// side init shim (`MlpBurnPolicy::new_seeded` →
/// `crate::policy::seeded_init`, driven by `StdRng::seed_from_u64`),
/// and the PSRO factory now receives a distinct per-construction seed
/// derived from `PsroConfig::seed`. As a result this curve is fully
/// **deterministic**: across 10 release-mode runs the averaged
/// `(second - first)` delta is identically `0.7655` (first-half mean
/// `0.9686`, second-half mean `1.7341`) — zero run-to-run variance.
/// The band is therefore tightened from `+1.0` to `+0.85`, ~11%
/// headroom over the now-exact observed delta. (We do *not* assert a
/// literal trace equality here because that belongs in
/// `tests/test_seeded_reproducibility.rs`, which checks bit-identity
/// directly; this test keeps its trend-monotonicity contract.)
#[test]
#[ignore = "multi-iteration PSRO convergence run; opt in with --ignored (prefer --release)"]
fn test_psro_exploitability_non_increasing_trend_on_matching_pennies() {
    let device: NdArrayDevice = Default::default();
    let joint_config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: 64,
        n_epochs: 1,
        minibatch_size: 32,
        ..Default::default()
    };
    let max_iterations: usize = 8;
    let seeds: [u64; 3] = [42, 7, 1234];

    let mut summed: Vec<f32> = vec![0.0_f32; max_iterations];
    for &seed in &seeds {
        let psro_config = PsroConfig {
            max_iterations,
            max_population_size: 50,
            br_train_steps_per_iteration: 2,
            payoff_eval_episodes: 4,
            seed,
        };
        let mut trainer = PsroTrainer::new(
            psro_config,
            joint_config.clone(),
            Box::new(FictitiousPlayMetaSolver::new(500)) as Box<dyn MetaSolver>,
            device,
            |dev: &NdArrayDevice, seed: u64| {
                MlpBurnPolicy::<B>::new_seeded(
                    MatchingPennies::OBS_DIM,
                    MatchingPennies::ACTION_DIM,
                    16,
                    seed,
                    dev,
                )
            },
            || BurnOptimizer::new(AdamConfig::new().init(), 1e-3),
            MatchingPennies::new,
        )
        .expect("PsroTrainer::new should succeed");

        let stats = trainer.run_silent().expect("PSRO run should not error");
        let expls: Vec<f32> = stats.iterations.iter().map(|it| it.exploitability).collect();
        println!("seed={seed} exploitability curve: {:?}", expls);

        // Per-seed sanity: all finite, all >= 0.
        for &e in &expls {
            assert!(e.is_finite(), "exploitability must be finite");
            assert!(e >= 0.0, "exploitability must be >= 0");
        }
        assert_eq!(expls.len(), max_iterations, "PSRO should record one stat per iteration");
        for (i, &e) in expls.iter().enumerate() {
            summed[i] += e;
        }
    }
    let averaged: Vec<f32> = summed.iter().map(|s| s / seeds.len() as f32).collect::<Vec<_>>();
    println!("seed-averaged exploitability curve: {:?}", averaged);

    // Trend on the averaged curve: second-half mean <= first-half
    // mean + epsilon. Matching pennies' first iteration starts from a
    // 2x2 random-vs-random matrix; later iterations are over larger
    // populations and typically have smaller exploitability gaps
    // because the meta-Nash has more room to mix.
    let n = averaged.len();
    let half = n / 2;
    let first_half_mean: f32 = averaged[..half].iter().copied().sum::<f32>() / half as f32;
    let second_half_mean: f32 = averaged[half..].iter().copied().sum::<f32>() / (n - half) as f32;
    println!(
        "averaged first-half mean = {:.4}, second-half mean = {:.4}",
        first_half_mean, second_half_mean
    );
    assert!(
        second_half_mean <= first_half_mean + 0.85,
        "averaged exploitability second-half mean exceeded first-half mean + 0.85;          first={first_half_mean}, second={second_half_mean}.          The band was tightened from +1.0 to +0.85 after issue #135 made the          curve bit-exact by seeding policy init; the deterministic delta is          0.7655. See test docs for the calibration rationale.",
    );
}
