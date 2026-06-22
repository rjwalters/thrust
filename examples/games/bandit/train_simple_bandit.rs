//! Train an actor-critic policy on `SimpleBandit` using the Burn 0.21
//! backend.
//!
//! The bandit trainer is the canonical end-to-end Burn example for the
//! thrust-rl crate. It demonstrates the rollout/loss/update loop using
//! the [`MlpBurnPolicy`] module, the move-through optimizer ownership
//! model, and the `EnvPool` vectorized env wrapper.
//!
//! # Backends
//!
//! - **Default**: CPU `NdArray` + `Autodiff` (no extra features).
//! - **Opt-in GPU**: `--features "training,wgpu"` swaps in `Autodiff<Wgpu>`
//!   (cross-platform GPU via Vulkan / Metal / DX12 / WebGPU). Used by issue
//!   #102 to validate Burn's GPU path end-to-end.
//!
//! Run:
//!
//! ```bash
//! cargo run --example train_simple_bandit --release
//! # or, on a GPU box:
//! cargo run --release --features "training,wgpu" --example train_simple_bandit
//! ```
//!
//! Expected: success rate > 0.95 within ~50k env steps on either
//! backend.

use anyhow::Result;
use burn::{
    backend::Autodiff,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, Tensor, TensorData},
};
use thrust_rl::{
    env::{Environment, pool::EnvPool, simple_bandit::SimpleBandit},
    policy::mlp::MlpBurnPolicy,
};

// Concrete backend stack — selected at compile time via Cargo features.
// `--features "training,wgpu"` swaps the CPU NdArray default for Burn's
// cross-platform GPU backend (Vulkan / Metal / DX12 / WebGPU).
#[cfg(not(feature = "wgpu"))]
type InnerBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "wgpu")]
type InnerBackend = burn::backend::Wgpu<f32, i32>;
type Backend = Autodiff<InnerBackend>;

#[cfg(not(feature = "wgpu"))]
const BACKEND_LABEL: &str = "NdArray<f32> + Autodiff (CPU)";
#[cfg(feature = "wgpu")]
const BACKEND_LABEL: &str = "Wgpu<f32, i32> + Autodiff (GPU: Vulkan/Metal/DX12/WebGPU)";

const NUM_ENVS: usize = 4;
const NUM_STEPS: usize = 100;
const TOTAL_TIMESTEPS: usize = 50_000;
const LEARNING_RATE: f32 = 1e-3;
const N_EPOCHS: usize = 10;
const CLIP_RANGE: f32 = 0.2;
const VF_COEF: f32 = 0.5;
const ENT_COEF: f32 = 0.1;
const HIDDEN_DIM: usize = 64;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    tracing::info!("SimpleBandit + Burn ({}) scout trainer", BACKEND_LABEL);
    let training_start = std::time::Instant::now();

    // ---- env setup ---------------------------------------------------------
    let probe = SimpleBandit::new();
    let obs_dim = probe.observation_space().shape[0];
    let action_dim = match probe.action_space().space_type {
        thrust_rl::env::SpaceType::Discrete(n) => n,
        _ => panic!("expected discrete action space"),
    };

    let mut env_pool = EnvPool::new(SimpleBandit::new, NUM_ENVS);
    let device = Default::default();

    // ---- policy / optimizer -----------------------------------------------
    // FRICTION: Burn's optimizer/module ownership model is "move in, move
    // out": `optim.step(...)` consumes the module and returns the updated
    // copy. There's no in-place equivalent. We keep `policy` in an Option
    // to swap it through `take().unwrap() -> optim.step(...) -> Some(...)`.
    let mut policy: Option<MlpBurnPolicy<Backend>> =
        Some(MlpBurnPolicy::new(obs_dim, action_dim, HIDDEN_DIM, &device));
    let mut optim = AdamConfig::new().init();

    // ---- rollout buffers (host-side, not on-device) -----------------------
    // The scout deliberately bypasses `crate::buffer::rollout::RolloutBuffer`
    // (it's tied to tch tensors). Sized for one rollout: NUM_STEPS *
    // NUM_ENVS transitions.
    let cap = NUM_STEPS * NUM_ENVS;
    let mut buf_obs = Vec::with_capacity(cap * obs_dim);
    let mut buf_actions = Vec::with_capacity(cap);
    let mut buf_log_probs = Vec::with_capacity(cap);
    let mut buf_values = Vec::with_capacity(cap);
    let mut buf_rewards = Vec::with_capacity(cap);

    let mut observations = env_pool.reset();
    let num_updates = TOTAL_TIMESTEPS / (NUM_STEPS * NUM_ENVS);

    let mut total_reward = 0.0_f64;
    let mut total_steps = 0_usize;

    for update in 0..num_updates {
        buf_obs.clear();
        buf_actions.clear();
        buf_log_probs.clear();
        buf_values.clear();
        buf_rewards.clear();

        // ---- rollout (no gradients needed) --------------------------------
        for _step in 0..NUM_STEPS {
            let obs_flat: Vec<f32> = observations.iter().flatten().copied().collect();
            let obs_t: Tensor<Backend, 2> =
                Tensor::from_data(TensorData::new(obs_flat.clone(), [NUM_ENVS, obs_dim]), &device);

            let (actions, log_probs, values) = policy.as_ref().unwrap().get_action_host(obs_t);

            let results = env_pool.step(&actions);

            for env_id in 0..NUM_ENVS {
                buf_obs.extend_from_slice(&observations[env_id]);
                buf_actions.push(actions[env_id]);
                buf_log_probs.push(log_probs[env_id]);
                buf_values.push(values[env_id]);
                buf_rewards.push(results[env_id].reward);

                total_reward += results[env_id].reward as f64;
                total_steps += 1;

                observations[env_id] = results[env_id].observation.clone();
                if results[env_id].terminated || results[env_id].truncated {
                    observations[env_id] = env_pool.reset_env(env_id)?;
                }
            }
        }

        // ---- advantages (gamma=0, gae_lambda=0 → A = r - V) ---------------
        // SimpleBandit is a contextual bandit; no temporal credit
        // assignment needed.
        let advantages: Vec<f32> =
            buf_rewards.iter().zip(buf_values.iter()).map(|(r, v)| r - v).collect();
        let returns: Vec<f32> = buf_rewards.clone();

        // Normalize advantages (standard PPO trick).
        let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let var =
            advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
        let std = (var + 1e-8).sqrt();
        let advantages: Vec<f32> = advantages.iter().map(|a| (a - mean) / std).collect();

        // ---- one PPO update (N_EPOCHS full-batch) -------------------------
        // FRICTION: no minibatching here — the bandit dataset is so small
        // it fits in one batch. Real trainers will need a Burn-side
        // Dataset/DataLoader story.
        let batch = advantages.len();
        let obs_b: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(buf_obs.clone(), [batch, obs_dim]), &device);
        let actions_b: Tensor<Backend, 1, Int> =
            Tensor::from_data(TensorData::new(buf_actions.clone(), [batch]), &device);
        let old_log_probs_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(buf_log_probs.clone(), [batch]), &device);
        let adv_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(advantages.clone(), [batch]), &device);
        let returns_b: Tensor<Backend, 1> =
            Tensor::from_data(TensorData::new(returns.clone(), [batch]), &device);

        let mut last_loss = 0.0_f32;
        let mut last_entropy = 0.0_f32;
        for _epoch in 0..N_EPOCHS {
            let p = policy.take().unwrap();
            let (new_log_probs, entropy, values_pred) =
                p.evaluate_actions(obs_b.clone(), actions_b.clone());

            // Surrogate clipped PPO loss.
            let ratio = (new_log_probs - old_log_probs_b.clone()).exp();
            let unclipped = ratio.clone() * adv_b.clone();
            let clipped = ratio.clamp(1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * adv_b.clone();
            let policy_loss = -unclipped.min_pair(clipped).mean();

            // Value MSE.
            let value_diff = values_pred - returns_b.clone();
            let value_loss = (value_diff.clone() * value_diff).mean();

            // Entropy bonus (encourages exploration).
            let entropy_mean = entropy.mean();

            let loss = policy_loss + value_loss.mul_scalar(VF_COEF)
                - entropy_mean.clone().mul_scalar(ENT_COEF);

            last_loss = loss.clone().into_scalar();
            last_entropy = entropy_mean.into_scalar();

            // FRICTION: gradient extraction is two-step:
            //   1. loss.backward() -> B::Gradients (one tensor per param)
            //   2. GradientsParams::from_grads(grads, &module) ties each gradient back to
            //      its parameter id by walking the module tree. Then optim.step consumes
            //      the module + grads together. The "from_grads + step" pair has no direct
            //      analog in tch's `optim.backward_step(&loss)`.
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &p);
            policy = Some(optim.step(LEARNING_RATE.into(), p, grads));
        }

        if update % 10 == 0 || update == num_updates - 1 {
            let success_rate = total_reward / total_steps as f64;
            tracing::info!(
                "Update {}/{} | steps: {} | success: {:.1}% | loss: {:.3} | entropy: {:.3}",
                update + 1,
                num_updates,
                total_steps,
                success_rate * 100.0,
                last_loss,
                last_entropy,
            );
        }
    }

    let final_success = total_reward / total_steps as f64;
    let elapsed = training_start.elapsed();
    tracing::info!(
        "Final success rate (burn path, {}): {:.1}% over {} steps",
        BACKEND_LABEL,
        final_success * 100.0,
        total_steps
    );
    tracing::info!(
        "Training wall-clock: {:.2}s ({:.0} env-steps/sec)",
        elapsed.as_secs_f64(),
        total_steps as f64 / elapsed.as_secs_f64()
    );

    // Sanity-check: also confirm the trained policy makes correct argmax
    // choices on the two contexts (eval mode, no autodiff).
    let trained = policy.unwrap().valid();
    eval_policy(&trained, &Default::default());

    Ok(())
}

/// Evaluate the trained policy on the two SimpleBandit contexts (0.0, 1.0)
/// and log the argmax action probabilities. Uses the inner (non-autodiff)
/// backend via `Module::valid()`.
fn eval_policy(policy: &MlpBurnPolicy<InnerBackend>, device: &burn::tensor::Device<InnerBackend>) {
    let obs: Tensor<InnerBackend, 2> =
        Tensor::from_data(TensorData::new(vec![0.0_f32, 1.0_f32], [2, 1]), device);
    let (logits, _) = policy.forward(obs);
    let probs_data: Vec<f32> = logits.clone().into_data().to_vec().expect("eval logits to_vec");
    tracing::info!(
        "Trained logits (state=0): {:?}  (state=1): {:?}",
        &probs_data[0..2],
        &probs_data[2..4],
    );
}
