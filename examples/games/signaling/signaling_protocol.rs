//! SignalingGame protocol-emergence experiment (issue #304, Phase 1-2 research
//! run).
//!
//! Drives the shipped [`SignalingGame`] reference env (#292) through the
//! shipped joint multi-agent PPO trainer (#295) and asks the epic-#266
//! question: **does a discrete communication protocol emerge?** The trainer,
//! env, comms routing (`split_action` / `place_message`), and the
//! message-entropy comms-loss hook (`JointTrainerConfig::comms_coef`) are all
//! used exactly as shipped — this binary only wires them together and measures
//! the outcome. No core `src/` code is added.
//!
//! # What is measured
//!
//! For each experimental arm we train two independent PPO policies (a speaker
//! and a listener) and report, aggregated over seeds:
//!
//! - **Reward** vs the **chance floor** `1/V` and vs a **no-comms ablation**
//!   (the channel severed — the listener's message slot is forced to the
//!   sentinel). If reward is identical with the channel ablated, no protocol is
//!   load-bearing (issue #304's control).
//! - **Mutual information** `I(referent; message)` (does the speaker *encode*
//!   the referent?) and `I(referent; guess)` (does information survive
//!   end-to-end?), both in bits (max `log2(V)`).
//! - A learning curve (mean rollout reward vs iteration).
//!
//! # Why a referent-randomizing, message-persisting wrapper
//!
//! The shipped `impl JointEnv for SignalingGame` keeps the hidden referent
//! **fixed** (`reset_joint` ignores its seed) and terminates **every step**
//! (single-shot). Driven through `collect_rollout` — which resets on `done` and
//! moves both agents simultaneously — that means (a) there is only ever one
//! referent to transmit, and (b) the message the speaker emits is erased by the
//! reset before the listener can ever observe it. Neither is a signaling game.
//!
//! This harness therefore wraps `SignalingGame` in an [`Arena`] that:
//! 1. draws a fresh referent each episode (the definition of a Lewis signaling
//!    game — the env's own `set_hidden` API and docstring intend a varying
//!    referent), and
//! 2. optionally holds an episode open for `episode_len` steps so the message
//!    *persists* into the listener's next observation.
//!
//! `episode_len == 1` faithfully reproduces the **shipped single-shot surface**
//! (the message never reaches the listener); `episode_len > 1` is the
//! **diagnostic** arm that isolates whether the bottleneck is the turn
//! structure or the learning dynamics. All message routing still flows through
//! the unmodified `SignalingGame::step_multi`.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example signaling_protocol --features training
//!
//! # Quick smoke (fewer iters / one seed):
//! ITERATIONS=10 SEEDS=1 cargo run --release --example signaling_protocol \
//!     --features training
//!
//! # Custom output path:
//! OUT=docs/research/data/2026-07-signaling-protocol-emergence.json \
//!     cargo run --release --example signaling_protocol --features training
//! ```

use anyhow::Result;
use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;
use thrust_rl::{
    env::SignalingGame,
    multi_agent::{
        MultiAgentEnvironment,
        joint::{JointEnv, JointMultiAgentTrainer, JointStepResult, JointTrainerConfig},
    },
    policy::multi_discrete_mlp::MultiDiscreteMlpBurnPolicy,
    train::optimizer::BurnOptimizer,
};

type Inner = NdArray<f32>;
type B = Autodiff<Inner>;
type Policy = MultiDiscreteMlpBurnPolicy<B>;

const SPEAKER: usize = 0;
const LISTENER: usize = 1;

// --- Experiment constants (env-overridable where noted) ------------------
const VOCAB: usize = 4;
const HIDDEN_DIM: usize = 32;
const ROLLOUT_STEPS: usize = 128;
const DEFAULT_ITERATIONS: usize = 500;
const DEFAULT_SEEDS: usize = 3;
const RECORD_EVERY: usize = 10;
const LR: f64 = 3e-3;
/// Sentinel written into the listener's observation when it has not (yet)
/// received a message — matches `SignalingGame`'s own pre-step sentinel.
const NO_MESSAGE: f32 = -1.0;

/// Referent-randomizing, message-persisting arena over [`SignalingGame`].
///
/// See the module docs for why this wrapper exists. It reuses the unmodified
/// `SignalingGame::step_multi` (and its `split_action` / `place_message` comms
/// routing); it only manages episode length, referent randomization, and the
/// ablation switch.
struct Arena {
    inner: SignalingGame,
    vocab: usize,
    episode_len: usize,
    /// When `false`, the listener's received-message slot is forced to
    /// [`NO_MESSAGE`] every step — the channel is severed (the control arm).
    comms_enabled: bool,
    rng: StdRng,
    step_in_ep: usize,
}

impl Arena {
    fn new(vocab: usize, episode_len: usize, comms_enabled: bool, seed: u64) -> Self {
        Self {
            inner: SignalingGame::new(vocab),
            vocab,
            episode_len,
            comms_enabled,
            rng: StdRng::seed_from_u64(seed),
            step_in_ep: 0,
        }
    }

    /// Per-agent observations, applying the ablation to the listener's slot.
    fn observe(&self) -> Vec<Vec<f32>> {
        let mut obs = vec![
            self.inner.get_agent_observation(SPEAKER),
            self.inner.get_agent_observation(LISTENER),
        ];
        if !self.comms_enabled {
            obs[LISTENER].iter_mut().for_each(|v| *v = NO_MESSAGE);
        }
        obs
    }
}

impl JointEnv for Arena {
    fn reset_joint(&mut self, _seed: Option<u64>) -> Vec<Vec<f32>> {
        let referent = self.rng.random_range(0..self.vocab) as i64;
        self.inner.set_hidden(referent);
        self.step_in_ep = 0;
        self.observe()
    }

    fn step_joint(&mut self, actions: &[Vec<i64>]) -> JointStepResult {
        // All comms routing (message slice + broadcast + place into the
        // listener obs) happens inside the shipped `step_multi`.
        let result = self.inner.step_multi(actions);
        self.step_in_ep += 1;
        let done = self.step_in_ep >= self.episode_len;
        let mut observations = result.observations;
        if !self.comms_enabled {
            observations[LISTENER].iter_mut().for_each(|v| *v = NO_MESSAGE);
        }
        JointStepResult { rewards: result.rewards, done, observations }
    }
}

/// One experimental arm's configuration.
struct Arm {
    name: &'static str,
    episode_len: usize,
    comms_enabled: bool,
    comms_coef: f64,
}

/// Aggregated result of running an arm over several seeds.
struct ArmResult {
    reward: Vec<f64>,
    i_tm: Vec<f64>,
    i_mg: Vec<f64>,
    i_tg: Vec<f64>,
    decode_ceiling: Vec<f64>,
    /// Mean learning curve across seeds (one entry per `RECORD_EVERY` iters).
    curve: Vec<f64>,
}

fn one_hot(t: i64, vocab: usize, device: &burn::tensor::Device<Inner>) -> Tensor<B, 2> {
    let mut v = vec![0.0_f32; vocab];
    v[t as usize] = 1.0;
    Tensor::<B, 2>::from_data(TensorData::new(v, [1, vocab]), device)
}

fn scalar_obs(x: f32, device: &burn::tensor::Device<Inner>) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(vec![x], [1, 1]), device)
}

/// Mutual information (bits) of a joint count table `joint[x][y]`.
fn mutual_information_bits(joint: &[Vec<f64>]) -> f64 {
    let total: f64 = joint.iter().flat_map(|r| r.iter()).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let rows = joint.len();
    let cols = joint[0].len();
    let px: Vec<f64> = joint.iter().map(|r| r.iter().sum::<f64>() / total).collect();
    let mut py = vec![0.0_f64; cols];
    for row in joint {
        for (j, &c) in row.iter().enumerate() {
            py[j] += c / total;
        }
    }
    let mut mi = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let pxy = joint[i][j] / total;
            if pxy > 0.0 && px[i] > 0.0 && py[j] > 0.0 {
                mi += pxy * (pxy / (px[i] * py[j])).log2();
            }
        }
    }
    mi.max(0.0)
}

/// Row-normalize a count table into conditional distributions `p(y | x)`.
fn normalize_rows(counts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    counts
        .iter()
        .map(|row| {
            let s: f64 = row.iter().sum();
            if s > 0.0 {
                row.iter().map(|&c| c / s).collect()
            } else {
                row.clone()
            }
        })
        .collect()
}

/// Protocol metrics for a trained (speaker, listener) pair.
struct Metrics {
    /// `I(referent; message)` — does the speaker *encode* the referent? (bits)
    i_tm: f64,
    /// `I(message; guess)` — can the listener *decode* a received token? (bits)
    i_mg: f64,
    /// `I(referent; guess)` — composed end-to-end channel. (bits)
    i_tg: f64,
    /// Decode ceiling: `Σ_t (1/V) p(guess = t | referent = t)` — the accuracy
    /// an informed listener would achieve (turn-structure penalty
    /// excluded).
    decode_ceiling: f64,
}

/// Probe-based evaluation of a trained (speaker, listener) pair.
///
/// Rather than replay episodes (whose autodiff-graph cost blows up with episode
/// length), we probe each policy directly at batch-1 and *compose* the channel:
/// - `p(message | referent)` from a speaker probe over all `V` referents,
/// - `p(guess | message)` from a listener probe over all `V` message tokens,
/// - `p(guess | referent) = Σ_m p(m | t) p(g | m)`.
///
/// All three mutual informations then follow from the discrete tables under a
/// uniform prior. This isolates the *learned protocol* (encode + decode
/// capacity) from the turn-structure penalty, which is captured separately by
/// the achieved training reward.
fn evaluate(
    speaker: &Policy,
    listener: &Policy,
    vocab: usize,
    device: &burn::tensor::Device<Inner>,
    seed: u64,
) -> Metrics {
    let mut rng = StdRng::seed_from_u64(seed);
    let probe = 300_usize;

    // Speaker encoding: p(message | referent).
    let mut tm = vec![vec![0.0_f64; vocab]; vocab];
    for (t, row) in tm.iter_mut().enumerate() {
        for _ in 0..probe {
            let (msg, _, _) =
                speaker.get_action_host_seeded(one_hot(t as i64, vocab, device), &mut rng);
            row[msg[0] as usize] += 1.0;
        }
    }

    // Listener decoding: p(guess | message) over real message tokens.
    let mut mg = vec![vec![0.0_f64; vocab]; vocab];
    for (m, row) in mg.iter_mut().enumerate() {
        for _ in 0..probe {
            let (guess, _, _) =
                listener.get_action_host_seeded(scalar_obs(m as f32, device), &mut rng);
            row[guess[0] as usize] += 1.0;
        }
    }

    // Composed channel: p(guess | referent) = Σ_m p(m | t) p(g | m).
    let p_m_t = normalize_rows(&tm);
    let p_g_m = normalize_rows(&mg);
    let mut tg = vec![vec![0.0_f64; vocab]; vocab];
    for t in 0..vocab {
        for g in 0..vocab {
            tg[t][g] = (0..vocab).map(|m| p_m_t[t][m] * p_g_m[m][g]).sum();
        }
    }
    let decode_ceiling = (0..vocab).map(|t| tg[t][t]).sum::<f64>() / vocab as f64;

    Metrics {
        i_tm: mutual_information_bits(&tm),
        i_mg: mutual_information_bits(&mg),
        i_tg: mutual_information_bits(&tg),
        decode_ceiling,
    }
}

/// Outcome of one training run (one arm, one seed).
struct SeedRun {
    /// Achieved reward: mean rollout reward over the final quarter of training
    /// (turn-structure penalty included — this is what the trainer optimizes).
    reward: f64,
    metrics: Metrics,
    /// Mean rollout reward per recorded iteration.
    curve: Vec<f64>,
}

/// Train one (speaker, listener) pair for `iterations` and return the achieved
/// reward, protocol metrics, and learning curve.
fn run_seed(
    arm: &Arm,
    iterations: usize,
    seed: u64,
    device: &burn::tensor::Device<Inner>,
) -> Result<SeedRun> {
    // Speaker observes a length-`VOCAB` one-hot; listener observes a single
    // received-token slot. Both emit one token over the vocabulary.
    let speaker = Policy::new_seeded(
        VOCAB,
        vec![VOCAB],
        HIDDEN_DIM,
        seed.wrapping_mul(2).wrapping_add(1),
        device,
    );
    let listener = Policy::new_seeded(
        1,
        vec![VOCAB],
        HIDDEN_DIM,
        seed.wrapping_mul(2).wrapping_add(2),
        device,
    );

    let optimizers = vec![
        BurnOptimizer::new(AdamConfig::new().init(), LR),
        BurnOptimizer::new(AdamConfig::new().init(), LR),
    ];

    let config = JointTrainerConfig {
        num_agents: 2,
        rollout_steps: ROLLOUT_STEPS,
        gamma: 0.9,
        gae_lambda: 0.95,
        ent_coef: 0.01,
        n_epochs: 4,
        minibatch_size: 64,
        iterate_all_minibatches: true,
        comms_coef: arm.comms_coef,
        ..Default::default()
    };

    let mut trainer =
        JointMultiAgentTrainer::new(vec![speaker, listener], optimizers, config, *device)?;

    let mut arena = Arena::new(
        VOCAB,
        arm.episode_len,
        arm.comms_enabled,
        seed.wrapping_mul(7).wrapping_add(11),
    );
    let mut last_obs = arena.reset_joint(None);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(13).wrapping_add(17));

    let tail_start = iterations - iterations / 4; // final quarter
    let mut curve = Vec::new();
    let mut tail_sum = 0.0_f64;
    let mut tail_n = 0.0_f64;
    for iter in 0..iterations {
        let rollout = trainer.collect_rollout(&mut arena, &mut last_obs, &mut rng);
        let rewards = &rollout.rewards[0];
        let mean_r = rewards.iter().map(|&r| r as f64).sum::<f64>() / rewards.len().max(1) as f64;
        if iter % RECORD_EVERY == 0 {
            curve.push(mean_r);
        }
        if iter >= tail_start {
            tail_sum += mean_r;
            tail_n += 1.0;
        }
        trainer.update(&rollout, &mut rng, |_: &[Tensor<B, 2>]| None)?;
    }

    let metrics = evaluate(
        trainer.policy(SPEAKER),
        trainer.policy(LISTENER),
        VOCAB,
        device,
        seed.wrapping_mul(19).wrapping_add(23),
    );

    Ok(SeedRun { reward: tail_sum / tail_n.max(1.0), metrics, curve })
}

fn run_arm(
    arm: &Arm,
    iterations: usize,
    n_seeds: usize,
    device: &burn::tensor::Device<Inner>,
) -> Result<ArmResult> {
    let mut reward = Vec::new();
    let mut i_tm = Vec::new();
    let mut i_mg = Vec::new();
    let mut i_tg = Vec::new();
    let mut decode_ceiling = Vec::new();
    let mut curves: Vec<Vec<f64>> = Vec::new();

    for s in 0..n_seeds {
        let seed = 1000 + s as u64;
        let run = run_seed(arm, iterations, seed, device)?;
        tracing::info!(
            "  arm={:<24} seed={seed}  reward={:.3}  I(T;M)={:.3}  I(M;G)={:.3}  I(T;G)={:.3}  \
             decode={:.3}",
            arm.name,
            run.reward,
            run.metrics.i_tm,
            run.metrics.i_mg,
            run.metrics.i_tg,
            run.metrics.decode_ceiling,
        );
        reward.push(run.reward);
        i_tm.push(run.metrics.i_tm);
        i_mg.push(run.metrics.i_mg);
        i_tg.push(run.metrics.i_tg);
        decode_ceiling.push(run.metrics.decode_ceiling);
        curves.push(run.curve);
    }

    // Mean learning curve across seeds (curves share length).
    let len = curves.iter().map(|c| c.len()).min().unwrap_or(0);
    let curve = (0..len)
        .map(|i| curves.iter().map(|c| c[i]).sum::<f64>() / n_seeds.max(1) as f64)
        .collect();

    Ok(ArmResult { reward, i_tm, i_mg, i_tg, decode_ceiling, curve })
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let iterations: usize = std::env::var("ITERATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let n_seeds: usize = std::env::var("SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SEEDS);
    let out_path = std::env::var("OUT")
        .unwrap_or_else(|_| "docs/research/data/2026-07-signaling-protocol-emergence.json".into());

    let device: burn::tensor::Device<Inner> = Default::default();
    let chance = 1.0 / VOCAB as f64;
    let max_mi = (VOCAB as f64).log2();

    tracing::info!("SignalingGame protocol-emergence experiment (issue #304)");
    tracing::info!("  vocab={VOCAB}  hidden_dim={HIDDEN_DIM}  rollout_steps={ROLLOUT_STEPS}");
    tracing::info!("  iterations={iterations}  seeds={n_seeds}  lr={LR}");
    tracing::info!("  chance reward = 1/V = {chance:.3}   max MI = log2(V) = {max_mi:.3} bits");

    // The experimental matrix:
    //  - single_shot_*     : episode_len 1 = the shipped single-shot surface.
    //  - persistent_*      : episode_len 8 = message persists to the listener.
    //  - *_ablated         : channel severed (the no-comms control).
    //  - *_coef_{default,high}: comms-loss-weight sweep on the live channel.
    let arms = [
        Arm {
            name: "single_shot_comms_on",
            episode_len: 1,
            comms_enabled: true,
            comms_coef: 0.0,
        },
        Arm {
            name: "single_shot_ablated",
            episode_len: 1,
            comms_enabled: false,
            comms_coef: 0.0,
        },
        Arm {
            name: "persistent_comms_on",
            episode_len: 8,
            comms_enabled: true,
            comms_coef: 0.0,
        },
        Arm {
            name: "persistent_ablated",
            episode_len: 8,
            comms_enabled: false,
            comms_coef: 0.0,
        },
        Arm {
            name: "persistent_coef_default",
            episode_len: 8,
            comms_enabled: true,
            comms_coef: 0.01,
        },
        Arm {
            name: "persistent_coef_high",
            episode_len: 8,
            comms_enabled: true,
            comms_coef: 0.1,
        },
    ];

    let mut arm_json = Vec::new();
    for arm in &arms {
        tracing::info!("running arm: {}", arm.name);
        let res = run_arm(arm, iterations, n_seeds, &device)?;
        tracing::info!(
            "  => reward={:.3}±{:.3}  I(T;M)={:.3}  I(M;G)={:.3}  I(T;G)={:.3}  decode={:.3}",
            mean(&res.reward),
            std_dev(&res.reward),
            mean(&res.i_tm),
            mean(&res.i_mg),
            mean(&res.i_tg),
            mean(&res.decode_ceiling),
        );
        arm_json.push(json!({
            "name": arm.name,
            "episode_len": arm.episode_len,
            "comms_enabled": arm.comms_enabled,
            "comms_coef": arm.comms_coef,
            "reward_mean": mean(&res.reward),
            "reward_std": std_dev(&res.reward),
            "reward_per_seed": res.reward,
            "i_tm_bits_mean": mean(&res.i_tm),
            "i_tm_bits_per_seed": res.i_tm,
            "i_mg_bits_mean": mean(&res.i_mg),
            "i_mg_bits_per_seed": res.i_mg,
            "i_tg_bits_mean": mean(&res.i_tg),
            "i_tg_bits_per_seed": res.i_tg,
            "decode_ceiling_mean": mean(&res.decode_ceiling),
            "decode_ceiling_per_seed": res.decode_ceiling,
            "learning_curve_mean": res.curve,
        }));
    }

    let out = json!({
        "schema_version": 1,
        "generated": "2026-07-06",
        "issue": 304,
        "config": {
            "vocab": VOCAB,
            "hidden_dim": HIDDEN_DIM,
            "rollout_steps": ROLLOUT_STEPS,
            "iterations": iterations,
            "seeds": n_seeds,
            "lr": LR,
            "gamma": 0.9,
            "ent_coef": 0.01,
            "n_epochs": 4,
            "minibatch_size": 64,
            "record_every": RECORD_EVERY,
            "mi_probe_samples": 300,
            "backend": "NdArray<f32> + Autodiff (CPU)",
        },
        "chance_reward": chance,
        "max_mi_bits": max_mi,
        "arms": arm_json,
    });

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    tracing::info!("wrote {out_path}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent variables carry zero mutual information.
    #[test]
    fn mi_of_independent_table_is_zero() {
        // Uniform joint = independent marginals.
        let joint = vec![vec![1.0; 4]; 4];
        assert!(mutual_information_bits(&joint).abs() < 1e-12);
    }

    /// A diagonal (perfectly coupled) table over V symbols carries log2(V)
    /// bits.
    #[test]
    fn mi_of_diagonal_table_is_log2_v() {
        let v = 4;
        let mut joint = vec![vec![0.0; v]; v];
        for (i, row) in joint.iter_mut().enumerate() {
            row[i] = 10.0;
        }
        assert!((mutual_information_bits(&joint) - (v as f64).log2()).abs() < 1e-12);
    }

    /// Empty / all-zero tables degrade to 0 bits rather than NaN.
    #[test]
    fn mi_of_zero_table_is_zero() {
        let joint = vec![vec![0.0; 3]; 3];
        assert_eq!(mutual_information_bits(&joint), 0.0);
    }

    #[test]
    fn normalize_rows_produces_conditionals() {
        let counts = vec![vec![2.0, 2.0], vec![0.0, 4.0], vec![0.0, 0.0]];
        let p = normalize_rows(&counts);
        assert_eq!(p[0], vec![0.5, 0.5]);
        assert_eq!(p[1], vec![0.0, 1.0]);
        // All-zero row passes through untouched (no division by zero).
        assert_eq!(p[2], vec![0.0, 0.0]);
    }

    /// The ablated arena severs the channel: the listener's observation is the
    /// sentinel even after the speaker emits a message.
    #[test]
    fn ablated_arena_forces_sentinel_listener_obs() {
        let mut arena = Arena::new(4, 8, false, 7);
        let obs0 = arena.reset_joint(None);
        assert_eq!(obs0[LISTENER], vec![NO_MESSAGE]);
        // Speaker emits token 2; listener guesses 0.
        let step = arena.step_joint(&[vec![2], vec![0]]);
        assert_eq!(step.observations[LISTENER], vec![NO_MESSAGE]);
    }

    /// The live persistent arena delivers the emitted token into the
    /// listener's next observation and holds the episode open.
    #[test]
    fn live_arena_delivers_message_and_persists_episode() {
        let mut arena = Arena::new(4, 8, true, 7);
        arena.reset_joint(None);
        let step = arena.step_joint(&[vec![3], vec![0]]);
        assert_eq!(step.observations[LISTENER], vec![3.0]);
        assert!(!step.done, "episode_len 8 must not terminate after one step");
    }

    /// `episode_len == 1` reproduces the shipped single-shot surface: done
    /// after every step (so `collect_rollout` resets and erases the message).
    #[test]
    fn single_shot_arena_terminates_every_step() {
        let mut arena = Arena::new(4, 1, true, 7);
        arena.reset_joint(None);
        let step = arena.step_joint(&[vec![1], vec![0]]);
        assert!(step.done);
    }

    /// The arena redraws the referent across episodes (seeded): over many
    /// resets every token must appear.
    #[test]
    fn arena_randomizes_referent_across_episodes() {
        let mut arena = Arena::new(4, 1, true, 42);
        let mut seen = [false; 4];
        for _ in 0..64 {
            let obs = arena.reset_joint(None);
            let referent = obs[SPEAKER].iter().position(|&x| x == 1.0).expect("one-hot referent");
            seen[referent] = true;
        }
        assert!(seen.iter().all(|&s| s), "all 4 referents should be drawn: {seen:?}");
    }
}
