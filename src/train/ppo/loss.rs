//! PPO loss math (Burn backend).
//!
//! After phase 5 of the Burn migration (#82), Burn is the only tensor
//! backend in the workspace; this module hosts the canonical PPO loss
//! surface used by [`crate::train::ppo::trainer::PPOTrainerBurn`].
//!
//! # API notes
//!
//! - Burn's `mean()` returns a rank-1 scalar tensor — there is no `Kind::Float`
//!   argument, and the resulting tensor still carries the backend parameter. We
//!   expose the scalar-extracting helper ([`scalar_f64`]) that the trainer
//!   needs at the loss boundary.
//! - Burn carries the const rank in the type, so the policy-loss / value-loss
//!   tensors carry their concrete rank (`Tensor<B, 1>` for per-row, scalar at
//!   the end).
//! - We extract scalars via [`scalar_f64`] which detaches implicitly via
//!   `.into_scalar()`; no explicit `no_grad` scope is needed because the
//!   autograd tape is not extended beyond the loss tensors we *return*.

use burn::{
    prelude::ToElement,
    tensor::{Tensor, backend::Backend},
};

/// Read a scalar tensor's value into an `f64`.
///
/// Burn's `into_scalar()` consumes the tensor and returns
/// `B::FloatElem`. We use Burn's [`ToElement`] trait to convert it to
/// `f64` regardless of whether the underlying float type is `f32` or
/// `f64`. Equivalent to tch's `f64::try_from(&tensor)` pattern, and
/// (critically) does not extend the autograd tape — the returned `f64`
/// is a host-side primitive.
pub fn scalar_f64<B: Backend>(tensor: Tensor<B, 1>) -> f64 {
    tensor.into_scalar().to_f64()
}

/// Compute PPO policy (surrogate) loss with clipping.
///
/// Returns `(policy_loss, clip_fraction, approx_kl)`. The two `f64`
/// values are host-side metrics (already detached from the autograd
/// graph) for trainer-side reporting and KL early stopping.
///
/// # Arguments
///
/// * `log_probs` - Log-probabilities of the actions under the *current* policy.
///   `[batch]`.
/// * `old_log_probs` - Log-probabilities under the *behaviour* policy.
///   `[batch]`.
/// * `advantages` - Normalized advantages `[batch]`.
/// * `clip_range` - PPO clipping parameter `ε`.
///
/// # Returns
///
/// 1. `policy_loss` — scalar tensor (rank 1, shape `[1]`) carrying the
///    grad-bearing surrogate loss.
/// 2. `clip_fraction` — fraction of samples where `|ratio − 1| > ε`. Diagnostic
///    only.
/// 3. `approx_kl` — `E[old_log_prob − new_log_prob]`, the standard biased KL
///    estimator used by SB3/CleanRL for early stopping.
pub fn compute_policy_loss<B: Backend>(
    log_probs: Tensor<B, 1>,
    old_log_probs: Tensor<B, 1>,
    advantages: Tensor<B, 1>,
    clip_range: f64,
) -> (Tensor<B, 1>, f64, f64) {
    // ratio = exp(new − old)
    let log_ratio = log_probs.clone() - old_log_probs.clone();
    let ratio = log_ratio.clone().exp();

    // Clipped surrogate objective (pessimistic min).
    let clipped_ratio = ratio.clone().clamp(1.0 - clip_range, 1.0 + clip_range);
    let surrogate_1 = advantages.clone() * ratio.clone();
    let surrogate_2 = advantages * clipped_ratio;
    let per_sample = surrogate_1.min_pair(surrogate_2);
    let policy_loss = per_sample.mean().neg();

    // Clip fraction: mean of (|ratio − 1| > ε).
    //
    // Burn 0.21's bool→float cast goes via `.float()`. The threshold
    // comparison gives a `Bool` tensor; we materialize it to a float,
    // take the mean, and pull the scalar to the host.
    let one = Tensor::<B, 1>::ones_like(&ratio);
    let abs_dev = (ratio - one).abs();
    let clipped_mask = abs_dev.greater_elem(clip_range as f32).float();
    let clip_fraction = scalar_f64(clipped_mask.mean());

    // Biased KL estimator: E[old − new].
    let approx_kl = scalar_f64((old_log_probs - log_probs).mean());

    (policy_loss, clip_fraction, approx_kl)
}

/// Compute the value-function loss with optional clipping.
///
/// Implements the *pessimistic* clipped value loss (the SB3/CleanRL
/// recipe). When `clip_range_vf` is non-finite the clipped branch is
/// skipped and we fall through to the plain MSE — semantically
/// identical to the tch path.
///
/// # Arguments
///
/// * `values` - Current value-function predictions `V(s_t)`. `[batch]`.
/// * `old_values` - Behaviour-policy value predictions. `[batch]`.
/// * `returns` - Bootstrap targets (advantages + old_values). `[batch]`.
/// * `clip_range_vf` - Value clipping parameter; `f64::INFINITY` to disable.
///
/// # Returns
///
/// 1. `value_loss` — scalar tensor carrying the grad-bearing value loss.
/// 2. `explained_var` — host-side `1 − Var(returns − values) / Var(returns)`.
pub fn compute_value_loss<B: Backend>(
    values: Tensor<B, 1>,
    old_values: Tensor<B, 1>,
    returns: Tensor<B, 1>,
    clip_range_vf: f64,
) -> (Tensor<B, 1>, f64) {
    let value_loss = if clip_range_vf.is_finite() {
        let clipped_dev =
            (values.clone() - old_values.clone()).clamp(-clip_range_vf, clip_range_vf);
        let values_clipped = old_values + clipped_dev;
        let vf_loss_1 = (values.clone() - returns.clone()).powf_scalar(2.0_f32);
        let vf_loss_2 = (values_clipped - returns.clone()).powf_scalar(2.0_f32);
        vf_loss_1.max_pair(vf_loss_2).mean()
    } else {
        (values.clone() - returns.clone()).powf_scalar(2.0_f32).mean()
    };

    // Explained variance is a pure host-side metric.
    let returns_vec: Vec<f32> = returns.clone().into_data().to_vec().unwrap_or_default();
    let values_vec: Vec<f32> = values.into_data().to_vec().unwrap_or_default();
    let explained_var = host_explained_variance(&returns_vec, &values_vec);

    (value_loss, explained_var)
}

/// Host-side explained-variance computation.
///
/// `1 − Var(returns − values) / Var(returns)`. Mirrors the tch path's
/// `var(false)` (biased estimator with denominator `n`) so the two
/// implementations agree numerically.
fn host_explained_variance(returns: &[f32], values: &[f32]) -> f64 {
    if returns.is_empty() {
        return 1.0;
    }
    let var_returns = host_variance_biased(returns);
    if var_returns == 0.0 {
        return 1.0;
    }
    let residual: Vec<f32> = returns.iter().zip(values).map(|(r, v)| r - v).collect();
    let var_residual = host_variance_biased(&residual);
    1.0 - var_residual / var_returns
}

fn host_variance_biased(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let sq_dev = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>();
    sq_dev / n
}

/// Entropy loss = `-mean(entropy)` (we minimize loss, so we negate the
/// entropy bonus before adding it into the total loss).
pub fn compute_entropy_loss<B: Backend>(entropy: Tensor<B, 1>) -> Tensor<B, 1> {
    entropy.mean().neg()
}

/// Generate randomized minibatch indices from a flat rollout buffer.
///
/// Backend-agnostic helper (uses `rand` only). Each minibatch contains
/// approximately `batch_size` samples; the last batch is short if
/// `buffer_size % batch_size != 0`.
///
/// This convenience overload draws shuffle randomness from the
/// thread-local `rand::rng()`. **Prefer
/// [`generate_minibatch_indices_with_rng`] when seedable
/// reproducibility matters** (e.g. PSRO / NFSP inner loops that pipe
/// a [`rand::rngs::StdRng`] seeded from the trainer config; see issue #109).
///
/// # Arguments
///
/// * `buffer_size` - Total number of samples in the buffer.
/// * `batch_size` - Desired size of each minibatch.
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains indices for one
/// minibatch.
pub fn generate_minibatch_indices(buffer_size: usize, batch_size: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    generate_minibatch_indices_with_rng(buffer_size, batch_size, &mut rng)
}

/// Generate randomized minibatch indices from a flat rollout buffer
/// using a caller-supplied RNG.
///
/// Identical to [`generate_minibatch_indices`] except the shuffle
/// randomness is drawn from `rng`. Pass a `StdRng` seeded from your
/// trainer config to make PPO / PSRO / NFSP inner loops
/// bit-reproducible (issue #109).
///
/// # Arguments
///
/// * `buffer_size` - Total number of samples in the buffer.
/// * `batch_size` - Desired size of each minibatch.
/// * `rng` - Caller-owned RNG used for the shuffle.
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains indices for one
/// minibatch.
pub fn generate_minibatch_indices_with_rng<R: rand::Rng + ?Sized>(
    buffer_size: usize,
    batch_size: usize,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    use rand::seq::SliceRandom;

    let mut indices: Vec<usize> = (0..buffer_size).collect();
    indices.shuffle(rng);

    indices.chunks(batch_size).map(|chunk| chunk.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Tensor, TensorData},
    };

    use super::*;

    type B = Autodiff<NdArray<f32>>;

    fn tensor1d(data: &[f32]) -> Tensor<B, 1> {
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
    }

    #[test]
    fn test_compute_policy_loss_shapes_and_ranges() {
        let log_probs = tensor1d(&[0.0, 0.5, -0.5]);
        let old_log_probs = tensor1d(&[0.0, 0.0, 0.0]);
        let advantages = tensor1d(&[1.0, -1.0, 0.5]);
        let clip_range = 0.2;

        let (loss, clip_frac, kl) =
            compute_policy_loss(log_probs, old_log_probs, advantages, clip_range);

        // Loss should be a rank-1 scalar tensor of shape [1].
        assert_eq!(loss.dims(), [1]);
        assert!((0.0..=1.0).contains(&clip_frac));
        // The biased KL estimator can go either direction; just sanity
        // check it's finite.
        assert!(kl.is_finite());
    }

    #[test]
    fn test_compute_value_loss_pessimistic_clip() {
        // Same construction as the tch test: clipped loss is tiny vs
        // unclipped, so the pessimistic max must reflect the unclipped
        // MSE.
        let old_values = tensor1d(&[0.0_f32, 0.0, 0.0]);
        let values = tensor1d(&[5.0_f32, 5.0, 5.0]);
        let returns = tensor1d(&[0.0_f32, 0.0, 0.0]);
        let clip_range_vf = 0.2;

        let (loss, _) = compute_value_loss(values, old_values, returns, clip_range_vf);
        let loss_val = scalar_f64(loss);

        let unclipped_mse = 25.0_f64;
        let eps = 1e-4_f64;
        assert!(
            loss_val >= unclipped_mse - eps,
            "expected pessimistic clipped value loss >= unclipped MSE ({}), got {}",
            unclipped_mse,
            loss_val
        );
    }

    #[test]
    fn test_compute_value_loss_infinite_clip_is_plain_mse() {
        let old_values = tensor1d(&[1.0_f32, 1.5, 0.8]);
        let values = tensor1d(&[1.0_f32, 2.0, 0.5]);
        let returns = tensor1d(&[1.2_f32, 2.1, 0.6]);

        let (loss_inf, _) = compute_value_loss(values, old_values, returns, f64::INFINITY);
        let loss_inf_val = scalar_f64(loss_inf);

        let expected = 0.02_f64;
        assert!(
            (loss_inf_val - expected).abs() < 1e-4,
            "infinite clip should yield plain MSE {}, got {}",
            expected,
            loss_inf_val
        );
    }

    #[test]
    fn test_compute_entropy_loss_negates_mean() {
        let entropy = tensor1d(&[0.5, 1.0, 0.1]);
        let loss = compute_entropy_loss(entropy);
        assert_eq!(loss.dims(), [1]);
        let loss_val = scalar_f64(loss);
        // mean = (0.5 + 1.0 + 0.1) / 3 = 0.5333...; negated = -0.5333...
        assert!(loss_val < 0.0);
        assert!((loss_val - (-0.5333333_f64)).abs() < 1e-4);
    }

    /// Issue #109 determinism guard: two RNGs constructed with the same
    /// seed must produce bit-identical minibatch index lists.
    ///
    /// This is the load-bearing test for the inner-loop seedable
    /// shuffle introduced in #109. Both
    /// `JointMultiAgentTrainer::update_with_active_agents` and
    /// `PPOTrainerBurn::train_step` ultimately consume this helper,
    /// so a stable contract here is sufficient to guarantee that
    /// same-seed PSRO / NFSP / single-agent PPO runs see the same
    /// minibatch order at every gradient step.
    #[test]
    fn test_generate_minibatch_indices_with_rng_is_deterministic() {
        use rand::{SeedableRng, rngs::StdRng};

        let buffer_size = 256;
        let batch_size = 32;
        let seed: u64 = 0xC0FFEE;

        let mut rng_a = StdRng::seed_from_u64(seed);
        let mut rng_b = StdRng::seed_from_u64(seed);

        let a = generate_minibatch_indices_with_rng(buffer_size, batch_size, &mut rng_a);
        let b = generate_minibatch_indices_with_rng(buffer_size, batch_size, &mut rng_b);

        assert_eq!(a, b, "same-seed RNG must yield bit-identical minibatch indices");

        // Also check that a *different* seed produces a *different*
        // index ordering — i.e. the seed actually controls the shuffle.
        let mut rng_c = StdRng::seed_from_u64(seed.wrapping_add(1));
        let c = generate_minibatch_indices_with_rng(buffer_size, batch_size, &mut rng_c);
        assert_ne!(a, c, "different seeds should produce different minibatch index orderings");
    }

    /// Multi-step determinism check: drawing several rounds of
    /// minibatch indices from the same RNG state on two parallel
    /// trainers must stay synchronized. This is the property that
    /// `update_with_active_agents` relies on across `n_epochs` epochs
    /// inside one PSRO/NFSP iteration.
    #[test]
    fn test_generate_minibatch_indices_with_rng_stays_synchronized() {
        use rand::{SeedableRng, rngs::StdRng};

        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        for epoch in 0..8 {
            let a = generate_minibatch_indices_with_rng(64, 8, &mut rng_a);
            let b = generate_minibatch_indices_with_rng(64, 8, &mut rng_b);
            assert_eq!(a, b, "epoch {epoch}: parallel RNGs diverged");
        }
    }
}
