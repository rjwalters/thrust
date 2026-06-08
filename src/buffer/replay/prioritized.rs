//! Prioritized Experience Replay (PER) buffer.
//!
//! This module adds a *prioritized* counterpart to
//! [`super::ReplayBuffer`]. Transitions are stored in the same flat
//! `Vec<f32>` layout (so the buffer stays WASM-compatible) but sampling
//! is biased toward transitions with large TD error — those are the ones
//! the Q-network has the most to learn from.
//!
//! # Algorithm (Schaul et al., 2015)
//!
//! Each transition `i` carries a scalar **priority** `pᵢ` that the
//! buffer maintains in a [`SumTree`]. The sampling probability is
//!
//! ```text
//! P(i) = pᵢ^α / Σⱼ pⱼ^α
//! ```
//!
//! where `α` controls how sharply we bias toward high-priority
//! transitions (`α = 0` recovers uniform sampling; `α = 1` is fully
//! proportional). Because biased sampling skews the Q-update's
//! expectation, each transition's loss is reweighted by an
//! **importance-sampling (IS) weight**:
//!
//! ```text
//! wᵢ = (1 / (N · P(i)))^β
//! ```
//!
//! with `β` annealed from a small value (e.g. `0.4`) to `1.0` over
//! training so the bias correction kicks in fully only once the
//! Q-function has stabilized. By convention `wᵢ` is normalized by
//! `max wⱼ` across the batch so the loss magnitude is comparable to the
//! uniform-replay case (Schaul §3.4).
//!
//! In this implementation the actual priority stored in the tree is
//! `pᵢ = (|TD_error_i| + ε)^α`, so the sum-tree's "weight" of leaf `i`
//! already equals `pᵢ^α` and `P(i) = treeᵢ / total`. The `α` field is
//! kept on the buffer purely so [`Self::push`] / [`Self::update_priorities`]
//! can fold it back into newly written priorities.
//!
//! # Storage
//!
//! For a buffer of capacity `C` and observation dimension `D`:
//! - `observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `actions`: `Vec<i64>` of length `C`
//! - `rewards`: `Vec<f32>` of length `C`
//! - `next_observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `dones`: `Vec<bool>` of length `C`
//! - `tree`: a [`SumTree`] with `C` leaves, where leaf `i` corresponds 1:1 to
//!   transition slot `i`. (We never permute slots, so leaf index and transition
//!   position are the same value — but the API distinguishes them for clarity
//!   and so a future "non-contiguous leaves" layout doesn't break callers.)
//!
//! # References
//!
//! - Schaul et al., *Prioritized Experience Replay* ([ICLR 2016](https://arxiv.org/abs/1511.05952)).

#[cfg(feature = "training-burn")]
use burn::tensor::{Int, Tensor as BurnTensor, TensorData, backend::Backend};
use rand::Rng;
#[cfg(feature = "training")]
use tch::{Device, Tensor};

use super::sum_tree::SumTree;

/// One minibatch sampled from a [`PrioritizedReplayBuffer`].
///
/// All fields are CPU-side primitive vectors; convert to `tch::Tensor`s
/// via [`PrioritizedBatch::to_tensors`] when handing them to the trainer.
#[derive(Debug, Clone)]
pub struct PrioritizedBatch {
    /// Flattened current observations, shape `[batch_size * obs_dim]`.
    pub observations: Vec<f32>,
    /// Actions taken, length `batch_size`.
    pub actions: Vec<i64>,
    /// Rewards received, length `batch_size`.
    pub rewards: Vec<f32>,
    /// Flattened next observations, shape `[batch_size * obs_dim]`.
    pub next_observations: Vec<f32>,
    /// Episode-end mask, length `batch_size`.
    pub dones: Vec<bool>,
    /// Per-sample importance-sampling weights, length `batch_size`,
    /// normalized so `max(weights) == 1.0`.
    pub is_weights: Vec<f32>,
    /// Sum-tree leaf indices for the sampled transitions, length
    /// `batch_size`. Pass these back into
    /// [`PrioritizedReplayBuffer::update_priorities`] alongside the new
    /// TD errors.
    pub indices: Vec<usize>,
    /// Length of one observation slice (so `to_tensors` can reshape).
    pub obs_dim: usize,
}

impl PrioritizedBatch {
    /// Number of transitions in the batch.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// `true` if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Stack the batch into `tch::Tensor`s on `device`.
    ///
    /// Returns `(obs, actions, rewards, next_obs, dones, is_weights)`
    /// with shapes:
    /// - `obs`: `[batch, obs_dim]`, `Kind::Float`
    /// - `actions`: `[batch]`, `Kind::Int64`
    /// - `rewards`: `[batch]`, `Kind::Float`
    /// - `next_obs`: `[batch, obs_dim]`, `Kind::Float`
    /// - `dones`: `[batch]`, `Kind::Float` (0.0 or 1.0)
    /// - `is_weights`: `[batch]`, `Kind::Float`
    #[cfg(feature = "training")]
    pub fn to_tensors(&self, device: Device) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let batch = self.len() as i64;
        let obs_dim = self.obs_dim as i64;

        let obs = Tensor::from_slice(&self.observations)
            .reshape([batch, obs_dim])
            .to_device(device);
        let next_obs = Tensor::from_slice(&self.next_observations)
            .reshape([batch, obs_dim])
            .to_device(device);
        let actions = Tensor::from_slice(&self.actions).to_device(device);
        let rewards = Tensor::from_slice(&self.rewards).to_device(device);
        let dones_f: Vec<f32> = self.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
        let dones = Tensor::from_slice(&dones_f).to_device(device);
        let is_weights = Tensor::from_slice(&self.is_weights).to_device(device);

        (obs, actions, rewards, next_obs, dones, is_weights)
    }

    /// Stack the batch into Burn tensors on `device`.
    ///
    /// Parallel to [`Self::to_tensors`] but emits a named
    /// [`PrioritizedBurnTensors`] struct so the DQN trainer can grab
    /// fields by name. Includes the importance-sampling weights that the
    /// uniform [`super::ReplayBatch`] does not carry.
    ///
    /// Shapes (all on `device`):
    /// - `observations`: `[batch, obs_dim]`, `f32`
    /// - `actions`: `[batch]`, `i64`
    /// - `rewards`: `[batch]`, `f32`
    /// - `next_observations`: `[batch, obs_dim]`, `f32`
    /// - `dones`: `[batch]`, `f32` (0.0 / 1.0)
    /// - `is_weights`: `[batch]`, `f32`
    ///
    /// Note: `indices` is *not* a tensor — leaf-index round-trips back
    /// into [`PrioritizedReplayBuffer::update_priorities`] as a host
    /// `&[usize]`, so it stays on `self`.
    #[cfg(feature = "training-burn")]
    pub fn to_burn_tensors<B: Backend>(&self, device: &B::Device) -> PrioritizedBurnTensors<B> {
        let batch = self.len();
        let obs_dim = self.obs_dim;

        // Direct rank-2 construction (vs. rank-1 + reshape) to keep the
        // empty-batch case panic-free; the reshape path trips an internal
        // shape assertion in cubecl-zspace when both dims are zero.
        let observations = BurnTensor::<B, 2>::from_data(
            TensorData::new(self.observations.clone(), [batch, obs_dim]),
            device,
        );
        let next_observations = BurnTensor::<B, 2>::from_data(
            TensorData::new(self.next_observations.clone(), [batch, obs_dim]),
            device,
        );
        let actions = BurnTensor::<B, 1, Int>::from_data(
            TensorData::new(self.actions.clone(), [batch]),
            device,
        );
        let rewards =
            BurnTensor::<B, 1>::from_data(TensorData::new(self.rewards.clone(), [batch]), device);
        let dones_f: Vec<f32> = self.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
        let dones = BurnTensor::<B, 1>::from_data(TensorData::new(dones_f, [batch]), device);
        let is_weights = BurnTensor::<B, 1>::from_data(
            TensorData::new(self.is_weights.clone(), [batch]),
            device,
        );

        PrioritizedBurnTensors {
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            is_weights,
        }
    }
}

/// Bundle of Burn tensors produced by [`PrioritizedBatch::to_burn_tensors`].
///
/// Adds the per-sample importance-sampling weights on top of the
/// `ReplayBurnTensors` field set. The trainer multiplies the per-row
/// TD-error magnitude by `is_weights` before reducing to a scalar loss
/// — this is the bias correction described in Schaul et al. §3.4.
#[cfg(feature = "training-burn")]
#[derive(Debug)]
pub struct PrioritizedBurnTensors<B: Backend> {
    /// Observations, shape `[batch, obs_dim]`, dtype `f32`.
    pub observations: BurnTensor<B, 2>,
    /// Discrete actions, shape `[batch]`, dtype `i64`.
    pub actions: BurnTensor<B, 1, Int>,
    /// Rewards, shape `[batch]`, dtype `f32`.
    pub rewards: BurnTensor<B, 1>,
    /// Bootstrap-state observations, shape `[batch, obs_dim]`, dtype `f32`.
    pub next_observations: BurnTensor<B, 2>,
    /// Episode-terminal mask (0.0 or 1.0), shape `[batch]`, dtype `f32`.
    pub dones: BurnTensor<B, 1>,
    /// Importance-sampling weights, shape `[batch]`, dtype `f32`,
    /// normalized so the max element is `1.0`.
    pub is_weights: BurnTensor<B, 1>,
}

/// Fixed-capacity prioritized replay buffer.
///
/// Stores `(obs, action, reward, next_obs, done)` transitions in a ring
/// buffer (same FIFO eviction as [`super::ReplayBuffer`]), plus a
/// per-transition priority in a [`SumTree`] for O(log N) proportional
/// sampling.
///
/// New transitions are pushed with the **maximum** existing priority
/// (or `1.0` for an empty buffer), guaranteeing each fresh transition is
/// sampled at least once before its priority can be updated by a TD
/// error.
#[derive(Debug, Clone)]
pub struct PrioritizedReplayBuffer {
    obs_dim: usize,
    capacity: usize,
    len: usize,
    write_idx: usize,

    observations: Vec<f32>,
    actions: Vec<i64>,
    rewards: Vec<f32>,
    next_observations: Vec<f32>,
    dones: Vec<bool>,

    tree: SumTree,
    alpha: f32,
    epsilon: f32,

    /// Tracks the largest priority seen so far. New transitions inherit
    /// this so they're guaranteed to be sampled at least once.
    /// Starts at `1.0` so the first transition has a non-zero priority.
    max_priority: f32,
}

impl PrioritizedReplayBuffer {
    /// Create a new prioritized replay buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions stored. Must be > 0.
    /// * `obs_dim` - Length of one observation vector. Must be > 0.
    /// * `alpha` - Priority exponent `α ∈ [0, 1]`. `0` recovers uniform
    ///   sampling; `1` is fully proportional. Typical value `0.6`.
    /// * `epsilon` - Tiny constant added to `|TD error|` before raising to `α`,
    ///   so transitions with zero TD error still have a tiny chance of being
    ///   resampled. Typical value `1e-6`.
    ///
    /// # Panics
    /// Panics if `capacity == 0`, `obs_dim == 0`, `alpha` is outside
    /// `[0, 1]`, or `epsilon < 0`.
    pub fn new(capacity: usize, obs_dim: usize, alpha: f32, epsilon: f32) -> Self {
        assert!(capacity > 0, "PrioritizedReplayBuffer capacity must be > 0");
        assert!(obs_dim > 0, "PrioritizedReplayBuffer obs_dim must be > 0");
        assert!(
            (0.0..=1.0).contains(&alpha),
            "PrioritizedReplayBuffer alpha must be in [0, 1], got {}",
            alpha
        );
        assert!(epsilon >= 0.0, "PrioritizedReplayBuffer epsilon must be >= 0, got {}", epsilon);

        Self {
            obs_dim,
            capacity,
            len: 0,
            write_idx: 0,
            observations: vec![0.0; capacity * obs_dim],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            next_observations: vec![0.0; capacity * obs_dim],
            dones: vec![false; capacity],
            tree: SumTree::new(capacity),
            alpha,
            epsilon,
            max_priority: 1.0,
        }
    }

    /// Push a single transition into the buffer with the current
    /// maximum priority (or `1.0` if the buffer is empty).
    ///
    /// When the buffer is at capacity, this overwrites the oldest
    /// transition (FIFO eviction); the corresponding leaf in the
    /// sum-tree is overwritten with the new priority.
    pub fn push(&mut self, obs: &[f32], action: i64, reward: f32, next_obs: &[f32], done: bool) {
        debug_assert_eq!(
            obs.len(),
            self.obs_dim,
            "PrioritizedReplayBuffer::push: obs.len() ({}) != obs_dim ({})",
            obs.len(),
            self.obs_dim
        );
        debug_assert_eq!(
            next_obs.len(),
            self.obs_dim,
            "PrioritizedReplayBuffer::push: next_obs.len() ({}) != obs_dim ({})",
            next_obs.len(),
            self.obs_dim
        );

        let start = self.write_idx * self.obs_dim;
        let end = start + self.obs_dim;
        self.observations[start..end].copy_from_slice(obs);
        self.next_observations[start..end].copy_from_slice(next_obs);
        self.actions[self.write_idx] = action;
        self.rewards[self.write_idx] = reward;
        self.dones[self.write_idx] = done;

        // Newly inserted transitions get max_priority so they're sampled
        // at least once before their priority is set by a TD error.
        self.tree.update(self.write_idx, self.max_priority);

        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Number of transitions currently in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the buffer holds no transitions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of transitions the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Observation dimension (length of one obs slice).
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Priority exponent `α`.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Priority floor `ε`.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Maximum leaf priority currently in the tree.
    ///
    /// Useful for diagnostics; the buffer also tracks this internally
    /// so `push` can hand fresh transitions a max-priority slot.
    pub fn max_priority(&self) -> f32 {
        self.max_priority
    }

    /// `true` if the buffer holds at least `min_size` transitions.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.len >= min_size
    }

    /// Sample a minibatch using stratified proportional sampling.
    ///
    /// Divides `[0, total_priority]` into `batch_size` equal segments
    /// and draws one uniform `p` from each, then walks the sum-tree to
    /// find the leaf whose cumulative-priority interval contains `p`.
    /// Stratification reduces variance vs. drawing `batch_size`
    /// independent uniforms from `[0, total]` (Schaul §3.3).
    ///
    /// Importance-sampling weights are computed as
    /// `wᵢ = (1 / (N · P(i)))^β` and then normalized by `max(wⱼ)` so the
    /// largest weight is exactly `1.0`.
    ///
    /// # Arguments
    /// * `batch_size` - Number of transitions to draw. Must be > 0.
    /// * `beta` - IS-weight exponent. Caller anneals from a small value (e.g.
    ///   `0.4`) to `1.0` over training.
    /// * `rng` - source of randomness.
    ///
    /// # Panics
    /// Panics if `self.is_empty()`, `batch_size == 0`, or the tree's
    /// total priority is zero (all leaves were explicitly set to zero —
    /// shouldn't happen in normal use because `push` always uses
    /// `max_priority ≥ ε^α > 0`).
    pub fn sample<R: Rng>(&self, batch_size: usize, beta: f32, rng: &mut R) -> PrioritizedBatch {
        assert!(!self.is_empty(), "PrioritizedReplayBuffer is empty; cannot sample");
        assert!(batch_size > 0, "batch_size must be > 0");

        let obs_dim = self.obs_dim;
        let total = self.tree.total();
        assert!(total > 0.0, "PrioritizedReplayBuffer::sample: tree total priority is zero");
        let n = self.len as f32;
        let segment = total / batch_size as f32;

        let mut observations = vec![0.0f32; batch_size * obs_dim];
        let mut next_observations = vec![0.0f32; batch_size * obs_dim];
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        let mut raw_weights = vec![0.0f32; batch_size];

        for k in 0..batch_size {
            let lo = segment * k as f32;
            let hi = segment * (k + 1) as f32;
            // `random_range(lo..hi)` would panic if `lo == hi` (zero-width
            // segment) — in that case just take the boundary.
            let p = if hi > lo {
                rng.random_range(lo..hi)
            } else {
                lo
            };
            let (leaf_idx, leaf_priority) = self.tree.find(p);

            // Defensive: if we somehow landed on a leaf past `self.len`
            // (only possible if the user manually zeroed all "valid"
            // leaves and left a stray non-zero leaf in the tail), clamp.
            // This can't happen in normal usage.
            let pos = leaf_idx.min(self.len - 1);

            let obs_slice = &mut observations[k * obs_dim..(k + 1) * obs_dim];
            let next_slice = &mut next_observations[k * obs_dim..(k + 1) * obs_dim];
            obs_slice.copy_from_slice(&self.observations[pos * obs_dim..(pos + 1) * obs_dim]);
            next_slice.copy_from_slice(&self.next_observations[pos * obs_dim..(pos + 1) * obs_dim]);
            actions.push(self.actions[pos]);
            rewards.push(self.rewards[pos]);
            dones.push(self.dones[pos]);
            indices.push(leaf_idx);

            // P(i) = leaf_priority / total → w_i = (1 / (N * P(i)))^β
            //                                   = (total / (N * leaf_priority))^β
            let prob = leaf_priority.max(1e-12) / total;
            raw_weights[k] = (1.0 / (n * prob)).powf(beta);
        }

        // Normalize so max(weights) == 1.0.
        let max_w = raw_weights.iter().copied().fold(0.0f32, f32::max);
        if max_w > 0.0 {
            for w in raw_weights.iter_mut() {
                *w /= max_w;
            }
        }

        PrioritizedBatch {
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            is_weights: raw_weights,
            indices,
            obs_dim,
        }
    }

    /// Update the priorities of a set of leaf indices with new TD errors.
    ///
    /// For each `(idx, td_err)` pair, the stored priority becomes
    /// `(|td_err| + ε)^α`. The `max_priority` tracker is bumped so
    /// future [`Self::push`] calls inherit at least this value.
    ///
    /// # Panics
    /// Panics if `indices` and `td_errors` have different lengths or if
    /// any index is out of range.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        assert_eq!(
            indices.len(),
            td_errors.len(),
            "update_priorities: indices ({}) and td_errors ({}) length mismatch",
            indices.len(),
            td_errors.len(),
        );

        for (&idx, &td_err) in indices.iter().zip(td_errors.iter()) {
            let magnitude = td_err.abs();
            // (|δ| + ε)^α
            let priority = (magnitude + self.epsilon).powf(self.alpha);
            // Guard against the (impossible-in-practice) case where
            // (magnitude + epsilon)^alpha = NaN due to a NaN td_err.
            let priority = if priority.is_finite() && priority >= 0.0 {
                priority
            } else {
                0.0
            };
            self.tree.update(idx, priority);
            if priority > self.max_priority {
                self.max_priority = priority;
            }
        }
    }

    /// Borrow the underlying sum-tree (useful for tests / diagnostics).
    pub fn tree(&self) -> &SumTree {
        &self.tree
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    fn push_synth(buf: &mut PrioritizedReplayBuffer, n: usize) {
        for i in 0..n {
            let obs = [i as f32, i as f32 + 0.1];
            let next_obs = [(i + 1) as f32, (i + 1) as f32 + 0.1];
            buf.push(&obs, (i % 2) as i64, i as f32, &next_obs, i % 4 == 3);
        }
    }

    #[test]
    fn test_push_and_basic_accessors() {
        let mut buf = PrioritizedReplayBuffer::new(8, 2, 0.6, 1e-6);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.obs_dim(), 2);
        assert!((buf.alpha() - 0.6).abs() < 1e-6);
        assert!((buf.epsilon() - 1e-6).abs() < 1e-9);

        push_synth(&mut buf, 5);
        assert_eq!(buf.len(), 5);
        assert!(!buf.is_empty());
        assert!(buf.is_ready(3));
        assert!(!buf.is_ready(6));
    }

    #[test]
    fn test_push_uses_max_priority_for_new_transitions() {
        // After update_priorities raises max, subsequent push should
        // use the bumped value (so freshly-inserted transitions are
        // sampled at least once even if older ones have huge priorities).
        let mut buf = PrioritizedReplayBuffer::new(4, 2, 0.6, 1e-6);
        push_synth(&mut buf, 2);
        // Bump leaf 0 to a huge priority via an enormous TD error.
        buf.update_priorities(&[0], &[100.0]);
        let bumped_max = buf.max_priority();
        assert!(
            bumped_max > 1.0,
            "max_priority should rise after big TD error, got {}",
            bumped_max
        );

        // Push another transition; its priority should equal bumped_max.
        buf.push(&[9.0, 9.0], 0, 0.0, &[10.0, 10.0], false);
        let new_leaf = buf.tree().leaf_priority(2);
        assert!(
            (new_leaf - bumped_max).abs() < 1e-4,
            "new leaf priority {} != max_priority {}",
            new_leaf,
            bumped_max,
        );
    }

    #[test]
    fn test_sum_tree_round_trip() {
        // Push N transitions with varying priorities, sample many times,
        // and verify the sampling frequency is proportional to priority
        // within a tolerance.
        let n: usize = 4;
        let mut buf = PrioritizedReplayBuffer::new(n, 1, 1.0, 0.0);
        for i in 0..n {
            buf.push(&[i as f32], i as i64, 0.0, &[(i + 1) as f32], false);
        }
        // Set priorities to [1, 2, 3, 4] (with α = 1, ε = 0 so the
        // stored priority is exactly |td|).
        buf.update_priorities(&[0, 1, 2, 3], &[1.0, 2.0, 3.0, 4.0]);

        let total: f32 = 1.0 + 2.0 + 3.0 + 4.0;
        let expected: [f32; 4] = [1.0 / total, 2.0 / total, 3.0 / total, 4.0 / total];

        let mut counts = vec![0usize; n];
        let mut rng = StdRng::seed_from_u64(42);
        let trials = 20_000;
        // Use batch_size=1 so each sample is independent (otherwise the
        // stratified sampler enforces one draw per segment which biases
        // the per-index histogram for small N).
        for _ in 0..trials {
            let batch = buf.sample(1, 0.4, &mut rng);
            counts[batch.actions[0] as usize] += 1;
        }

        for i in 0..n {
            let observed = counts[i] as f32 / trials as f32;
            let diff = (observed - expected[i]).abs();
            assert!(
                diff < 0.02,
                "leaf {} sampling freq off: expected {:.4}, observed {:.4} (diff {:.4})",
                i,
                expected[i],
                observed,
                diff,
            );
        }
    }

    #[test]
    fn test_priority_update_zero_stops_sampling() {
        // Push 4 transitions, then set one priority to ~0 via
        // update_priorities (the actual stored priority is
        // (0 + ε)^α = ε^α, which is what we set ε = 0 for here).
        let mut buf = PrioritizedReplayBuffer::new(4, 1, 1.0, 0.0);
        for i in 0..4 {
            buf.push(&[i as f32], i as i64, 0.0, &[(i + 1) as f32], false);
        }
        // Default priorities are all max (1.0 each). Zero out leaf 2.
        buf.update_priorities(&[0, 1, 2, 3], &[1.0, 1.0, 0.0, 1.0]);

        let mut rng = StdRng::seed_from_u64(7);
        let mut seen_two = 0usize;
        for _ in 0..100 {
            let batch = buf.sample(1, 0.4, &mut rng);
            if batch.actions[0] == 2 {
                seen_two += 1;
            }
        }
        assert_eq!(seen_two, 0, "zero-priority transition should never be sampled");
    }

    #[test]
    fn test_is_weight_normalization_to_max_one() {
        let mut buf = PrioritizedReplayBuffer::new(8, 1, 0.6, 1e-6);
        for i in 0..8 {
            buf.push(&[i as f32], i as i64, 0.0, &[(i + 1) as f32], false);
        }
        // Vary priorities so IS weights aren't all equal.
        buf.update_priorities(&[0, 1, 2, 3, 4, 5, 6, 7], &[0.1, 1.0, 2.0, 0.5, 5.0, 0.3, 0.7, 1.5]);

        let mut rng = StdRng::seed_from_u64(11);
        let batch = buf.sample(4, 0.4, &mut rng);
        let max_w = batch.is_weights.iter().copied().fold(0.0f32, f32::max);
        assert!(
            (max_w - 1.0).abs() < 1e-6,
            "max IS weight must be exactly 1.0 after normalization, got {}",
            max_w
        );
        for w in &batch.is_weights {
            assert!(*w > 0.0 && *w <= 1.0 + 1e-6, "IS weight out of (0, 1] range: {}", w);
        }
    }

    #[test]
    fn test_sample_indices_are_leaf_indices_round_trip_priorities() {
        // After sampling, calling update_priorities with the returned
        // indices should change exactly those leaves and leave others
        // alone.
        let mut buf = PrioritizedReplayBuffer::new(4, 1, 1.0, 0.0);
        for i in 0..4 {
            buf.push(&[i as f32], i as i64, 0.0, &[(i + 1) as f32], false);
        }
        // Equal priorities.
        buf.update_priorities(&[0, 1, 2, 3], &[1.0, 1.0, 1.0, 1.0]);

        let mut rng = StdRng::seed_from_u64(99);
        let batch = buf.sample(4, 0.4, &mut rng);

        // Update each sampled index with a known TD error.
        let new_errors: Vec<f32> = (0..batch.indices.len()).map(|k| 0.5 + k as f32).collect();
        buf.update_priorities(&batch.indices, &new_errors);

        // Group by index in case the sampler drew duplicates.
        let mut last_update: HashMap<usize, f32> = HashMap::new();
        for (i, &idx) in batch.indices.iter().enumerate() {
            last_update.insert(idx, (new_errors[i].abs() + 0.0_f32).powf(1.0));
        }
        for (&idx, &expected) in &last_update {
            let stored = buf.tree().leaf_priority(idx);
            assert!(
                (stored - expected).abs() < 1e-5,
                "leaf {} priority {} != expected {}",
                idx,
                stored,
                expected,
            );
        }
    }

    #[test]
    fn test_capacity_one_works() {
        // Edge case: capacity == 1 means SumTree has a single node.
        let mut buf = PrioritizedReplayBuffer::new(1, 2, 0.6, 1e-6);
        buf.push(&[1.0, 2.0], 0, 0.5, &[3.0, 4.0], false);
        assert_eq!(buf.len(), 1);

        let mut rng = StdRng::seed_from_u64(0);
        let batch = buf.sample(1, 0.4, &mut rng);
        assert_eq!(batch.actions, vec![0]);
        assert_eq!(batch.rewards, vec![0.5]);
        assert_eq!(batch.indices, vec![0]);
        assert!((batch.is_weights[0] - 1.0).abs() < 1e-6);
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_to_tensors_shapes() {
        let mut buf = PrioritizedReplayBuffer::new(8, 4, 0.6, 1e-6);
        for i in 0..6 {
            buf.push(&[i as f32; 4], (i % 2) as i64, i as f32, &[i as f32 + 1.0; 4], i == 5);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let batch = buf.sample(3, 0.4, &mut rng);
        let (obs, actions, rewards, next_obs, dones, is_weights) = batch.to_tensors(Device::Cpu);
        assert_eq!(obs.size(), vec![3, 4]);
        assert_eq!(next_obs.size(), vec![3, 4]);
        assert_eq!(actions.size(), vec![3]);
        assert_eq!(rewards.size(), vec![3]);
        assert_eq!(dones.size(), vec![3]);
        assert_eq!(is_weights.size(), vec![3]);
    }

    #[cfg(feature = "training-burn")]
    mod burn_tests {
        use burn::backend::NdArray;

        use super::*;

        type B = NdArray<f32>;

        #[test]
        fn test_to_burn_tensors_shapes_and_roundtrip() {
            let mut buf = PrioritizedReplayBuffer::new(8, 4, 0.6, 1e-6);
            for i in 0..6 {
                buf.push(&[i as f32; 4], (i % 2) as i64, i as f32, &[i as f32 + 1.0; 4], i == 5);
            }
            let mut rng = StdRng::seed_from_u64(1);
            let batch = buf.sample(3, 0.4, &mut rng);
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);

            assert_eq!(t.observations.dims(), [3, 4]);
            assert_eq!(t.next_observations.dims(), [3, 4]);
            assert_eq!(t.actions.dims(), [3]);
            assert_eq!(t.rewards.dims(), [3]);
            assert_eq!(t.dones.dims(), [3]);
            assert_eq!(t.is_weights.dims(), [3]);

            let obs_flat: Vec<f32> = t.observations.into_data().to_vec().unwrap();
            assert_eq!(obs_flat, batch.observations);
            let next_flat: Vec<f32> = t.next_observations.into_data().to_vec().unwrap();
            assert_eq!(next_flat, batch.next_observations);
            let acts: Vec<i64> = t.actions.into_data().to_vec().unwrap();
            assert_eq!(acts, batch.actions);
            let rews: Vec<f32> = t.rewards.into_data().to_vec().unwrap();
            assert_eq!(rews, batch.rewards);
            let isw: Vec<f32> = t.is_weights.into_data().to_vec().unwrap();
            assert_eq!(isw, batch.is_weights);
            let dones_f: Vec<f32> = t.dones.into_data().to_vec().unwrap();
            let expected_dones: Vec<f32> =
                batch.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
            assert_eq!(dones_f, expected_dones);
        }

        #[test]
        fn test_to_burn_tensors_empty_batch_does_not_panic() {
            let batch = PrioritizedBatch {
                observations: vec![],
                actions: vec![],
                rewards: vec![],
                next_observations: vec![],
                dones: vec![],
                is_weights: vec![],
                indices: vec![],
                obs_dim: 4,
            };
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);
            assert_eq!(t.observations.dims(), [0, 4]);
            assert_eq!(t.next_observations.dims(), [0, 4]);
            assert_eq!(t.actions.dims(), [0]);
            assert_eq!(t.rewards.dims(), [0]);
            assert_eq!(t.dones.dims(), [0]);
            assert_eq!(t.is_weights.dims(), [0]);
        }
    }

    #[test]
    fn test_ring_wraparound_evicts_oldest() {
        // Capacity 4, push 6 → first 2 must be gone, latest 4 must be
        // there. Sample many times; we should only see actions 2..6.
        let mut buf = PrioritizedReplayBuffer::new(4, 1, 1.0, 0.0);
        for i in 0..6 {
            buf.push(&[i as f32], i as i64, 0.0, &[(i + 100) as f32], false);
        }
        assert_eq!(buf.len(), 4);

        let mut rng = StdRng::seed_from_u64(0);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let batch = buf.sample(1, 0.4, &mut rng);
            seen.insert(batch.actions[0]);
        }
        assert!(!seen.contains(&0));
        assert!(!seen.contains(&1));
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _ = PrioritizedReplayBuffer::new(0, 2, 0.6, 1e-6);
    }

    #[test]
    #[should_panic(expected = "alpha must be in")]
    fn test_bad_alpha_panics() {
        let _ = PrioritizedReplayBuffer::new(4, 2, 1.5, 1e-6);
    }

    #[test]
    #[should_panic(expected = "epsilon must be")]
    fn test_negative_epsilon_panics() {
        let _ = PrioritizedReplayBuffer::new(4, 2, 0.6, -1.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_update_length_mismatch_panics() {
        let mut buf = PrioritizedReplayBuffer::new(4, 1, 0.6, 1e-6);
        buf.push(&[0.0], 0, 0.0, &[1.0], false);
        buf.update_priorities(&[0, 1], &[1.0]);
    }
}
