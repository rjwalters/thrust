//! P3 specialization experiment: joint multi-agent PPO with cross-agent
//! redundancy penalty, in Rust. Supports two modes via `--centralized-critic`:
//!
//! 1. **No-critic (default)** — IPPO-style independent learners that share a
//!    synchronized rollout buffer so the cross-agent representational
//!    regularizer can be added as a single backward pass.
//! 2. **Centralized critic** (`--centralized-critic`) — MAPPO-style CT-DE: a
//!    shared value function `V_c(s_joint) -> R` is attached to the joint
//!    trainer. Its loss is added to the joint update and reported alongside
//!    the per-agent stats. See `src/multi_agent/centralized_critic.rs` for
//!    the design rationale (CT-DE, MAPPO/COMA references). With
//!    `--centralized-critic --critic-vf-coef 0` the binary recovers the
//!    no-critic baseline numerically (the critic's own VarStore is gradient-
//!    isolated from per-agent params).
//!
//! In both modes the cross-agent representational regularizer
//!
//! ```text
//! L_red = lambda_red * sum_{i<j} || corr(Z_i, Z_j) ||_F^2 / (d^2)
//! ```
//!
//! can be added to the joint loss via the `--lambda-red` flag. The gradient
//! couples every policy's encoder through the auxiliary term while leaving
//! inference strictly per-agent.
//!
//! The core training machinery lives in [`thrust_rl::multi_agent::joint`]
//! (see issue #5); this binary is responsible only for arg parsing, env
//! construction, the experiment-specific `redundancy_penalty` aux loss, and
//! checkpoint I/O.
//!
//! Usage (after libtorch is set up, see issue #8 for current macOS notes):
//!
//! ```bash
//! # No-critic baseline (IPPO).
//! cargo run --release \
//!     --example train_p3 \
//!     --features "training env-bucket-brigade" \
//!     -- --scenario default --lambda-red 0.01 --seed 42
//!
//! # Centralized-critic mode (MAPPO).
//! cargo run --release \
//!     --example train_p3 \
//!     --features "training env-bucket-brigade" \
//!     -- --scenario default --lambda-red 0.01 --seed 42 --centralized-critic
//! ```

use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use bucket_brigade_core::SCENARIOS;
use tch::{Kind, Tensor, nn::OptimizerConfig};
use thrust_rl::{
    env::games::bucket_brigade::BucketBrigadeMaEnv,
    multi_agent::{
        CentralizedCritic, CentralizedCriticConfig,
        joint::{JointEnv, JointMultiAgentTrainer, JointStepResult, JointTrainerConfig},
    },
    policy::multi_discrete_mlp::MultiDiscreteMlpPolicy,
};

// =====================================================================
// Configuration (experiment-local)
// =====================================================================

#[derive(Debug, Clone)]
struct Config {
    scenario: String,
    lambda_red: f64,
    seed: u64,
    num_iterations: usize,
    rollout_steps: usize,
    num_agents: usize,
    hidden_dim: i64,
    lr: f64,
    gamma: f64,
    gae_lambda: f64,
    clip_range: f64,
    vf_coef: f64,
    ent_coef: f64,
    ppo_epochs: usize,
    minibatch_size: usize,
    // Centralized critic knobs (only used when `centralized_critic` is true;
    // defaults match the PPO baseline values).
    centralized_critic: bool,
    critic_hidden_dim: i64,
    critic_lr: f64,
    critic_vf_coef: f64,
    critic_clip_range_vf: f64,
    output_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            scenario: "default".into(),
            lambda_red: 0.01,
            seed: 42,
            num_iterations: 50,
            rollout_steps: 2048,
            num_agents: 4,
            hidden_dim: 64,
            lr: 3e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            ppo_epochs: 4,
            minibatch_size: 256,
            centralized_critic: false,
            critic_hidden_dim: 64,
            critic_lr: 3e-4,
            critic_vf_coef: 0.5,
            critic_clip_range_vf: 0.0,
            output_dir: PathBuf::from("runs/p3"),
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    // The `--centralized-critic` flag is a boolean toggle (no value); all
    // other flags take a single value. Iterate manually so we can branch
    // on the flag shape.
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        // Boolean flag: no value to consume.
        if flag == "--centralized-critic" {
            cfg.centralized_critic = true;
            // If the caller chose `--centralized-critic`, default the output
            // directory to a distinct path so the two modes don't clobber
            // each other's checkpoints when invoked back-to-back. Caller can
            // still override with `--output-dir`.
            if cfg.output_dir == PathBuf::from("runs/p3") {
                cfg.output_dir = PathBuf::from("runs/p3_centralized");
            }
            continue;
        }
        let val = args.next().expect("flag without value");
        match flag.as_str() {
            "--scenario" => cfg.scenario = val,
            "--lambda-red" => cfg.lambda_red = val.parse().unwrap(),
            "--seed" => cfg.seed = val.parse().unwrap(),
            "--num-iterations" => cfg.num_iterations = val.parse().unwrap(),
            "--num-updates" => cfg.num_iterations = val.parse().unwrap(),
            "--rollout-steps" => cfg.rollout_steps = val.parse().unwrap(),
            "--num-agents" => cfg.num_agents = val.parse().unwrap(),
            "--critic-hidden-dim" => cfg.critic_hidden_dim = val.parse().unwrap(),
            "--critic-lr" => cfg.critic_lr = val.parse().unwrap(),
            "--critic-vf-coef" => cfg.critic_vf_coef = val.parse().unwrap(),
            "--critic-clip-range-vf" => cfg.critic_clip_range_vf = val.parse().unwrap(),
            "--output-dir" => cfg.output_dir = PathBuf::from(val),
            other => panic!("unknown flag: {}", other),
        }
    }
    cfg
}

// =====================================================================
// Env adapter: wrap BucketBrigadeMaEnv to satisfy the JointEnv trait.
// =====================================================================
//
// BucketBrigadeMaEnv natively takes `&[[u8; 3]]` actions
// (`[house, mode, signal]` after upstream issue #235), whereas the joint
// trainer hands the env a `&[Vec<i64>]` (one entry per agent, length =
// num_action_dims). This shim translates between the two; once the
// MultiAgentEnvironment trait grows multi-discrete action support (issue
// #3 / #6) the shim collapses to a one-line impl on BucketBrigadeMaEnv
// itself.
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

// =====================================================================
// Redundancy penalty: pairwise cross-correlation Frobenius norm.
// =====================================================================

/// Cross-agent redundancy penalty (differentiable). Identical formula to the
/// Python `JointPPOTrainer::redundancy_penalty`.
fn redundancy_penalty(features: &[&Tensor]) -> Tensor {
    let n = features.len();
    let device = features[0].device();
    if n < 2 {
        return Tensor::zeros([], (Kind::Float, device));
    }

    // Standardize each per-feature column (zero mean, unit std per dim).
    let standardized: Vec<Tensor> = features
        .iter()
        .map(|z| {
            let mean = z.mean_dim([0i64].as_slice(), true, Kind::Float);
            let centered = *z - &mean;
            let std = centered.std_dim([0i64].as_slice(), false, true);
            let std = std.clamp_min(1e-6);
            centered / std
        })
        .collect();

    let batch_size = standardized[0].size()[0] as f64;
    let d = standardized[0].size()[1] as f64;

    let mut total = Tensor::zeros([], (Kind::Float, device));
    let mut num_pairs: usize = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            // [d, d] cross-correlation.
            let cross = standardized[i].transpose(0, 1).matmul(&standardized[j]) / batch_size;
            total = total + cross.pow_tensor_scalar(2).sum(Kind::Float);
            num_pairs += 1;
        }
    }
    total / (num_pairs as f64 * d * d)
}

// =====================================================================
// Driver
// =====================================================================

fn main() -> Result<()> {
    let cfg = parse_args();
    tracing_subscriber::fmt::init();

    let scenario = SCENARIOS.get(cfg.scenario.as_str()).expect("unknown scenario").clone();

    let env_inner = BucketBrigadeMaEnv::new(scenario, cfg.num_agents, Some(cfg.seed));
    let obs_dim = env_inner.obs_dim();
    let action_dims: Vec<i64> = env_inner.action_dims();

    // Build N policies + N optimizers, all on the same device (CUDA if
    // available, else CPU).
    let mut policies: Vec<MultiDiscreteMlpPolicy> = Vec::with_capacity(cfg.num_agents);
    let mut optimizers: Vec<tch::nn::Optimizer> = Vec::with_capacity(cfg.num_agents);
    for _ in 0..cfg.num_agents {
        let p = MultiDiscreteMlpPolicy::new(obs_dim as i64, action_dims.clone(), cfg.hidden_dim);
        let opt = tch::nn::Adam::default().build(p.var_store(), cfg.lr)?;
        policies.push(p);
        optimizers.push(opt);
    }
    let device = policies[0].device();

    // Centralized-critic config (only consumed when `--centralized-critic`
    // is set). Joint-obs dim is the env's per-agent obs_dim — observations
    // are shared across agents in bucket-brigade so the joint state IS the
    // per-agent state.
    let critic_cfg = CentralizedCriticConfig {
        hidden_dim: cfg.critic_hidden_dim,
        learning_rate: cfg.critic_lr,
        vf_coef: cfg.critic_vf_coef,
        clip_range_vf: cfg.critic_clip_range_vf,
    };

    let trainer_config = JointTrainerConfig {
        num_agents: cfg.num_agents,
        rollout_steps: cfg.rollout_steps,
        gamma: cfg.gamma,
        gae_lambda: cfg.gae_lambda,
        clip_range: cfg.clip_range,
        clip_range_vf: 0.0,
        vf_coef: cfg.vf_coef,
        ent_coef: cfg.ent_coef,
        n_epochs: cfg.ppo_epochs,
        minibatch_size: cfg.minibatch_size,
        max_grad_norm: 0.5,
        normalize_advantages: true,
        centralized_critic: if cfg.centralized_critic { Some(critic_cfg.clone()) } else { None },
    };

    let mut trainer = if cfg.centralized_critic {
        let mut critic =
            CentralizedCritic::new_on_device(obs_dim as i64, critic_cfg.hidden_dim, device);
        let critic_opt = critic.build_optimizer(critic_cfg.learning_rate)?;
        JointMultiAgentTrainer::new_with_centralized_critic(
            policies,
            optimizers,
            critic,
            critic_opt,
            trainer_config,
        )?
    } else {
        JointMultiAgentTrainer::new(policies, optimizers, trainer_config)?
    };

    let mut env = BbJointEnv { inner: env_inner };
    let initial = env.reset_joint(Some(cfg.seed));
    let mut last_obs = initial[0].clone();

    let mode_tag = if cfg.centralized_critic { " (centralized critic)" } else { "" };
    println!(
        "== thrust P3{}: scenario={} lambda_red={} seed={} num_agents={} ==",
        mode_tag, cfg.scenario, cfg.lambda_red, cfg.seed, cfg.num_agents
    );
    println!("obs_dim={} action_dims={:?} device={:?}", obs_dim, action_dims, device);
    if cfg.centralized_critic {
        println!(
            "centralized critic: hidden_dim={} lr={} vf_coef={} clip_range_vf={}",
            cfg.critic_hidden_dim, cfg.critic_lr, cfg.critic_vf_coef, cfg.critic_clip_range_vf,
        );
    }

    let lambda_red = cfg.lambda_red;
    let t_total = Instant::now();
    for it in 0..cfg.num_iterations {
        let t0 = Instant::now();
        let rollout = trainer.collect_rollout(&mut env, &mut last_obs);

        let mean_reward = rollout
            .rewards
            .iter()
            .map(|r| f64::try_from(&r.sum(Kind::Float)).unwrap_or(0.0))
            .sum::<f64>()
            / cfg.rollout_steps as f64;

        let stats = trainer.update(&rollout, |features: &[&Tensor]| {
            if lambda_red > 0.0 {
                Some(lambda_red * redundancy_penalty(features))
            } else {
                None
            }
        })?;

        let dt = t0.elapsed();
        if it % cfg.num_iterations.max(10) / 10 == 0 || it == cfg.num_iterations - 1 {
            // Aux loss = lambda_red * redundancy_penalty (the trainer reports
            // the scaled scalar). Divide out lambda_red for display so the
            // log line matches the pre-migration format.
            let red_loss = if lambda_red > 0.0 {
                stats.aux_loss / lambda_red
            } else {
                0.0
            };
            let critic_str = if cfg.centralized_critic {
                format!(
                    " | c_v_loss {:.4} | c_ev {:+.3}",
                    stats.centralized_value_loss, stats.centralized_explained_var
                )
            } else {
                String::new()
            };
            println!(
                "  iter {:4} | team_reward {:8.3} | red_loss {:.4}{} | total {:.3} | {:.1}s",
                it,
                mean_reward,
                red_loss,
                critic_str,
                stats.total_loss,
                dt.as_secs_f64(),
            );
        }
    }
    let elapsed = t_total.elapsed();
    println!("done in {:.1}s ({} iters)", elapsed.as_secs_f64(), cfg.num_iterations);

    // Persist checkpoints (policies + optional centralized critic).
    std::fs::create_dir_all(&cfg.output_dir)?;
    for (i, policy) in trainer.policies.iter().enumerate() {
        let path = cfg.output_dir.join(format!("agent_{i}.safetensors"));
        policy
            .var_store()
            .save(&path)
            .map_err(|e| anyhow::anyhow!("save policy {}: {}", i, e))?;
    }
    if let Some(critic) = trainer.centralized_critic.as_ref() {
        let path = cfg.output_dir.join("centralized_critic.safetensors");
        critic
            .var_store()
            .save(&path)
            .map_err(|e| anyhow::anyhow!("save centralized critic: {}", e))?;
        println!("saved centralized critic checkpoint to {:?}", path);
    }
    println!("saved {} policy checkpoints to {:?}", cfg.num_agents, cfg.output_dir);
    Ok(())
}
