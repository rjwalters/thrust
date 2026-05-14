//! P3 specialization experiment: joint multi-agent PPO with cross-agent
//! redundancy penalty, in Rust.
//!
//! This is the thrust counterpart of the Python prototype at
//! `slepian-wolf-marl/bucket-brigade/bucket_brigade/training/joint_trainer.py`
//! plus `experiments/p3_specialization/train.py`. The architecture is
//! independent learners (IPPO) reorganized to share a synchronized rollout
//! buffer so the cross-agent representational regularizer
//!
//! ```text
//! L_red = lambda_red * sum_{i<j} || corr(Z_i, Z_j) ||_F^2 / (d^2)
//! ```
//!
//! can be added to the joint loss in a single backward pass --- the gradient
//! couples every policy's encoder through the auxiliary term while leaving
//! inference strictly per-agent.
//!
//! Usage (after libtorch is set up, see issue #8 for current macOS notes):
//!
//! ```bash
//! cargo run --release \
//!     --example train_p3 \
//!     --features "training env-bucket-brigade" \
//!     -- --scenario default --lambda-red 0.01 --seed 42
//! ```

use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;

use tch::{Device, Kind, Tensor};
use thrust_rl::{
    env::games::bucket_brigade::BucketBrigadeMaEnv,
    policy::multi_discrete_mlp::MultiDiscreteMlpPolicy,
    train::ppo::{compute_policy_loss, compute_value_loss},
};

use bucket_brigade_core::SCENARIOS;

// =====================================================================
// Configuration
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
            output_dir: PathBuf::from("runs/p3"),
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag without value");
        match flag.as_str() {
            "--scenario" => cfg.scenario = val,
            "--lambda-red" => cfg.lambda_red = val.parse().unwrap(),
            "--seed" => cfg.seed = val.parse().unwrap(),
            "--num-iterations" => cfg.num_iterations = val.parse().unwrap(),
            "--rollout-steps" => cfg.rollout_steps = val.parse().unwrap(),
            "--num-agents" => cfg.num_agents = val.parse().unwrap(),
            "--output-dir" => cfg.output_dir = PathBuf::from(val),
            other => panic!("unknown flag: {}", other),
        }
    }
    cfg
}

// =====================================================================
// Rollout collection
// =====================================================================

/// Synchronized rollout: each step every agent acts on the same global obs;
/// the env steps with the joint action; per-agent rewards / values / log-probs
/// are recorded in parallel tensors.
struct Rollout {
    /// Shared observations: `[T, obs_dim]`. Same for every agent.
    observations: Tensor,
    /// Per-agent actions: `[N, T, num_action_dims]`.
    actions: Vec<Tensor>,
    /// Per-agent old log-probs from rollout-time policy: `[N, T]`.
    log_probs: Vec<Tensor>,
    /// Per-agent value estimates: `[N, T]`.
    values: Vec<Tensor>,
    /// Per-agent rewards: `[N, T]`.
    rewards: Vec<Tensor>,
    /// Episode-termination flag (shared across agents): `[T]`.
    dones: Tensor,
}

fn collect_rollout(
    env: &mut BucketBrigadeMaEnv,
    policies: &[MultiDiscreteMlpPolicy],
    num_steps: usize,
    obs_dim: usize,
    num_agents: usize,
    num_action_dims: i64,
    device: Device,
    last_obs: &mut Vec<f32>,
) -> Rollout {
    let mut obs_buf = vec![0.0_f32; num_steps * obs_dim];
    let mut act_buf: Vec<Vec<i64>> =
        (0..num_agents).map(|_| vec![0_i64; num_steps * num_action_dims as usize]).collect();
    let mut lp_buf: Vec<Vec<f32>> = (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
    let mut val_buf: Vec<Vec<f32>> = (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
    let mut rew_buf: Vec<Vec<f32>> = (0..num_agents).map(|_| vec![0.0_f32; num_steps]).collect();
    let mut done_buf = vec![0.0_f32; num_steps];

    for t in 0..num_steps {
        // Copy current global obs into the rolling buffer.
        let start = t * obs_dim;
        obs_buf[start..start + obs_dim].copy_from_slice(last_obs);

        // Each policy samples an action conditioned on the global obs.
        let obs_t = Tensor::from_slice(last_obs).to_device(device).view([1, obs_dim as i64]);
        let mut joint_action = vec![[0_u8, 0]; num_agents];
        for (i, policy) in policies.iter().enumerate() {
            let (actions_t, log_p_t, value_t) = tch::no_grad(|| policy.get_action(&obs_t));
            // actions_t : [1, num_action_dims]
            let row: Vec<i64> = Vec::try_from(&actions_t.view([num_action_dims])).unwrap();
            joint_action[i] = [row[0] as u8, row[1] as u8];

            let off = t * num_action_dims as usize;
            for (k, &a) in row.iter().enumerate() {
                act_buf[i][off + k] = a;
            }

            let lp: f32 = f32::try_from(&log_p_t).unwrap_or(0.0);
            let v: f32 = f32::try_from(&value_t).unwrap_or(0.0);
            lp_buf[i][t] = lp;
            val_buf[i][t] = v;
        }

        let result = env.step(&joint_action);
        for i in 0..num_agents {
            rew_buf[i][t] = result.rewards[i];
        }
        done_buf[t] = if result.done { 1.0 } else { 0.0 };

        // Update last_obs for the next step --- shared global view (every
        // agent has the same observation in Bucket Brigade).
        if result.done {
            let fresh = env.reset(None);
            // All agents have the same obs in BB; we take agent 0's view.
            *last_obs = fresh[0].clone();
        } else {
            *last_obs = result.observations[0].clone();
        }
    }

    // Materialize tensors.
    let observations =
        Tensor::from_slice(&obs_buf).to_device(device).view([num_steps as i64, obs_dim as i64]);
    let actions = act_buf
        .into_iter()
        .map(|a| {
            Tensor::from_slice(&a)
                .to_device(device)
                .view([num_steps as i64, num_action_dims])
        })
        .collect();
    let log_probs = lp_buf
        .into_iter()
        .map(|v| Tensor::from_slice(&v).to_device(device))
        .collect();
    let values = val_buf
        .into_iter()
        .map(|v| Tensor::from_slice(&v).to_device(device))
        .collect();
    let rewards = rew_buf
        .into_iter()
        .map(|v| Tensor::from_slice(&v).to_device(device))
        .collect();
    let dones = Tensor::from_slice(&done_buf).to_device(device);

    Rollout { observations, actions, log_probs, values, rewards, dones }
}

// =====================================================================
// Advantage estimation (single-env GAE, computed per agent)
// =====================================================================

fn compute_gae_single_agent(
    rewards: &Tensor,
    values: &Tensor,
    dones: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> (Tensor, Tensor) {
    let rewards_v: Vec<f32> = Vec::try_from(rewards).unwrap();
    let values_v: Vec<f32> = Vec::try_from(values).unwrap();
    let dones_v: Vec<f32> = Vec::try_from(dones).unwrap();
    let t = rewards_v.len();

    let mut advantages = vec![0.0_f32; t];
    let mut gae = 0.0_f32;
    for i in (0..t).rev() {
        let next_v = if i == t - 1 { 0.0 } else { values_v[i + 1] };
        let delta =
            rewards_v[i] + (gamma as f32) * next_v * (1.0 - dones_v[i]) - values_v[i];
        gae = delta + (gamma as f32) * (gae_lambda as f32) * (1.0 - dones_v[i]) * gae;
        advantages[i] = gae;
    }
    let returns: Vec<f32> = advantages.iter().zip(&values_v).map(|(&a, &v)| a + v).collect();

    let device = rewards.device();
    let adv_t = Tensor::from_slice(&advantages).to_device(device);
    let ret_t = Tensor::from_slice(&returns).to_device(device);
    (adv_t, ret_t)
}

// =====================================================================
// Redundancy penalty: pairwise cross-correlation Frobenius norm.
// =====================================================================

/// Cross-agent redundancy penalty (differentiable). Identical formula to the
/// Python `JointPPOTrainer::redundancy_penalty`.
fn redundancy_penalty(features: &[Tensor]) -> Tensor {
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
            let centered = z - &mean;
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
// PPO update (joint over all agents)
// =====================================================================

#[allow(clippy::too_many_arguments)]
fn ppo_update(
    cfg: &Config,
    policies: &[MultiDiscreteMlpPolicy],
    optimizers: &mut [tch::nn::Optimizer],
    rollout: &Rollout,
) -> JointStats {
    let t_total = rollout.observations.size()[0];
    let device = rollout.observations.device();

    // Per-agent advantages and returns.
    let mut advantages: Vec<Tensor> = Vec::with_capacity(cfg.num_agents);
    let mut returns: Vec<Tensor> = Vec::with_capacity(cfg.num_agents);
    for i in 0..cfg.num_agents {
        let (adv, ret) = compute_gae_single_agent(
            &rollout.rewards[i],
            &rollout.values[i],
            &rollout.dones,
            cfg.gamma,
            cfg.gae_lambda,
        );
        // Normalize.
        let adv_mean = adv.mean(Kind::Float);
        let adv_std = adv.std(false).clamp_min(1e-8);
        let adv = (adv - adv_mean) / adv_std;
        advantages.push(adv);
        returns.push(ret);
    }

    let mut stats = JointStats::zeros(cfg.num_agents);
    let mb = cfg.minibatch_size.min(t_total as usize);

    for _epoch in 0..cfg.ppo_epochs {
        let idx_full = Tensor::randperm(t_total, (Kind::Int64, device));
        let idx = idx_full.slice(0, 0, mb as i64, 1);

        let obs_mb = rollout.observations.index_select(0, &idx);

        // Recompute encoder features per agent on this minibatch, for both
        // the PPO heads' forward and the cross-agent redundancy penalty.
        let mut per_agent_losses: Vec<Tensor> = Vec::with_capacity(cfg.num_agents);
        let mut features: Vec<Tensor> = Vec::with_capacity(cfg.num_agents);

        for (i, policy) in policies.iter().enumerate() {
            let actions_mb = rollout.actions[i].index_select(0, &idx);
            let old_lp_mb = rollout.log_probs[i].index_select(0, &idx);
            let adv_mb = advantages[i].index_select(0, &idx);
            let ret_mb = returns[i].index_select(0, &idx);
            let old_v_mb = rollout.values[i].index_select(0, &idx);

            let (new_lp, entropy, values_mb) = policy.evaluate_actions(&obs_mb, &actions_mb);
            let feat = policy.encoder_features(&obs_mb);
            features.push(feat);

            let (policy_loss, _clip_frac, _kl) =
                compute_policy_loss(&new_lp, &old_lp_mb, &adv_mb, cfg.clip_range);
            let (value_loss, _ev) =
                compute_value_loss(&values_mb, &old_v_mb, &ret_mb, 0.0);
            let entropy_mean = entropy.mean(Kind::Float);

            let agent_loss = &policy_loss + cfg.vf_coef * &value_loss
                - cfg.ent_coef * &entropy_mean;

            stats.policy_loss[i] += f64::try_from(&policy_loss).unwrap_or(0.0);
            stats.value_loss[i] += f64::try_from(&value_loss).unwrap_or(0.0);
            stats.entropy[i] += f64::try_from(&entropy_mean).unwrap_or(0.0);

            per_agent_losses.push(agent_loss);
        }

        // Aggregate per-agent losses, add the cross-agent redundancy penalty.
        let mut joint_loss = Tensor::zeros([], (Kind::Float, device));
        for l in &per_agent_losses {
            joint_loss = joint_loss + l;
        }
        let red = if cfg.lambda_red > 0.0 {
            redundancy_penalty(&features)
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };
        stats.redundancy_loss += f64::try_from(&red).unwrap_or(0.0);
        joint_loss = joint_loss + cfg.lambda_red * &red;
        stats.total_loss += f64::try_from(&joint_loss).unwrap_or(0.0);

        // Zero, backward, step every optimizer. Each optimizer's clip_grad
        // and step affect only its own var-store, so the gradients from
        // policy i's heads stay inside policy i. The redundancy penalty's
        // gradient flows into every encoder by construction.
        for opt in optimizers.iter_mut() {
            opt.zero_grad();
        }
        joint_loss.backward();
        for opt in optimizers.iter_mut() {
            opt.clip_grad_norm(0.5);
            opt.step();
        }
    }

    // Average across epochs.
    let n = cfg.ppo_epochs as f64;
    for i in 0..cfg.num_agents {
        stats.policy_loss[i] /= n;
        stats.value_loss[i] /= n;
        stats.entropy[i] /= n;
    }
    stats.redundancy_loss /= n;
    stats.total_loss /= n;

    stats
}

#[derive(Debug, Clone)]
struct JointStats {
    policy_loss: Vec<f64>,
    value_loss: Vec<f64>,
    entropy: Vec<f64>,
    redundancy_loss: f64,
    total_loss: f64,
}

impl JointStats {
    fn zeros(num_agents: usize) -> Self {
        Self {
            policy_loss: vec![0.0; num_agents],
            value_loss: vec![0.0; num_agents],
            entropy: vec![0.0; num_agents],
            redundancy_loss: 0.0,
            total_loss: 0.0,
        }
    }
}

// =====================================================================
// Driver
// =====================================================================

fn main() -> Result<()> {
    let cfg = parse_args();
    tracing_subscriber::fmt::init();

    let scenario = SCENARIOS
        .get(cfg.scenario.as_str())
        .expect("unknown scenario")
        .clone();

    let mut env = BucketBrigadeMaEnv::new(scenario, cfg.num_agents, Some(cfg.seed));
    let obs_dim = env.obs_dim();
    let action_dims = env.action_dims();
    let num_action_dims = action_dims.len() as i64;

    // Build N policies + N optimizers. All on the same device (CUDA if
    // available, else CPU).
    let mut policies: Vec<MultiDiscreteMlpPolicy> = Vec::with_capacity(cfg.num_agents);
    let mut optimizers: Vec<tch::nn::Optimizer> = Vec::with_capacity(cfg.num_agents);
    for _ in 0..cfg.num_agents {
        let mut p = MultiDiscreteMlpPolicy::new(
            obs_dim as i64,
            action_dims.to_vec(),
            cfg.hidden_dim,
        );
        let opt = p.optimizer(cfg.lr)?;
        policies.push(p);
        optimizers.push(opt);
    }
    let device = policies[0].device();

    // First observation (every agent has the same global view in BB).
    let initial = env.reset(Some(cfg.seed));
    let mut last_obs = initial[0].clone();

    println!(
        "== thrust P3: scenario={} lambda_red={} seed={} num_agents={} ==",
        cfg.scenario, cfg.lambda_red, cfg.seed, cfg.num_agents
    );
    println!("obs_dim={} action_dims={:?} device={:?}", obs_dim, action_dims, device);

    let t_total = Instant::now();
    for it in 0..cfg.num_iterations {
        let t0 = Instant::now();
        let rollout = collect_rollout(
            &mut env,
            &policies,
            cfg.rollout_steps,
            obs_dim,
            cfg.num_agents,
            num_action_dims,
            device,
            &mut last_obs,
        );

        let mean_reward = rollout
            .rewards
            .iter()
            .map(|r| f64::try_from(&r.sum(Kind::Float)).unwrap_or(0.0))
            .sum::<f64>()
            / cfg.rollout_steps as f64;

        let stats = ppo_update(&cfg, &policies, &mut optimizers, &rollout);

        let dt = t0.elapsed();
        if it % cfg.num_iterations.max(10) / 10 == 0 || it == cfg.num_iterations - 1 {
            println!(
                "  iter {:4} | team_reward {:8.3} | red_loss {:.4} | total {:.3} | {:.1}s",
                it,
                mean_reward,
                stats.redundancy_loss,
                stats.total_loss,
                dt.as_secs_f64(),
            );
        }
    }
    let elapsed = t_total.elapsed();
    println!("done in {:.1}s ({} iters)", elapsed.as_secs_f64(), cfg.num_iterations);

    // Persist checkpoints.
    std::fs::create_dir_all(&cfg.output_dir)?;
    for (i, policy) in policies.iter().enumerate() {
        let path = cfg.output_dir.join(format!("agent_{i}.safetensors"));
        policy
            .var_store()
            .save(&path)
            .map_err(|e| anyhow::anyhow!("save policy {}: {}", i, e))?;
    }
    println!("saved {} policy checkpoints to {:?}", cfg.num_agents, cfg.output_dir);
    Ok(())
}
