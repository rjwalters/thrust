//! Recurrent rollout buffer and full-sequence (Strategy A) sampler.
//!
//! Phase 2 of the recurrent-policy epic (#262). Adds a **new**
//! [`RecurrentRolloutBuffer`] alongside the feedforward
//! [`RolloutBuffer`](super::storage::RolloutBuffer); the feedforward buffer
//! and all its consumers are left byte-for-byte untouched (see the design
//! note `docs/RECURRENT_POLICY_DESIGN.md`, Q1). The feedforward buffer's
//! whole contract flattens time and env into a rank-2 batch, structurally
//! erasing the temporal order recurrence needs — so recurrence gets its own
//! type that preserves the `[num_steps, num_envs]` grid and materializes it
//! as **rank-3** `[N_env, T, obs_dim]` sequences.
//!
//! # Composition over duplication
//!
//! [`RecurrentRolloutBuffer`] embeds a private
//! [`RolloutBuffer`](super::storage::RolloutBuffer) for the nine
//! `[num_steps, num_envs]` transition arrays (observations, actions,
//! rewards, values, log-probs, terminated, truncated, advantages, returns)
//! and adds two `[num_steps, num_envs, hidden_dim]` arrays, `hidden` and
//! `cell`, for the rollout-time recurrent state. GAE is delegated straight
//! through to
//! [`compute_advantages`](super::gae::compute_advantages) /
//! [`compute_advantages_partial`](super::gae::compute_advantages_partial),
//! so advantage/return math is **byte-identical** to the feedforward path.
//!
//! # The GAE / state-reset asymmetry (do not homogenize)
//!
//! GAE bootstraps on `terminated` **only** — a truncation is a time-limit
//! cut, not an MDP terminal, so the value target should still bootstrap from
//! the next state. The hidden-state reset mask
//! ([`RecurrentRolloutBatch::episode_starts`]) uses `terminated || truncated`
//! — a truncated step still ends the episode, so its recurrent state must
//! reset. Both consumers read the same stored `terminated`/`truncated`
//! grids but combine them differently; do not "fix" one to match the other
//! (design note Q2).
//!
//! # Warm-start (Strategy A), not Strategy B
//!
//! The stored `(h, c)` serve rollout-time warm-starting only: at the start
//! of the next rollout iteration, an env that did **not** end its episode
//! (`terminated || truncated`) on the last step carries its final recurrent
//! state into step 0; an env that did is seeded with zeros. See
//! [`RecurrentRolloutBuffer::seed_warm_start`]. The `to_sequence_batch`
//! training forward always passes `initial_state: None` (zeros) because
//! episode boundaries inside the sequence are handled by `episode_starts`
//! masking in
//! [`LstmBurnPolicy::evaluate_sequences`](crate::policy::lstm::LstmBurnPolicy::evaluate_sequences).
//! Strategy B (fixed-length subsequences with stored boundary states) is
//! deferred; the `(h, c)` storage here is the hook it would reuse.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use super::storage::RolloutBuffer;

/// Rollout buffer that preserves temporal order for a recurrent policy.
///
/// Stores the same nine `[num_steps, num_envs]` transition arrays as
/// [`RolloutBuffer`](super::storage::RolloutBuffer) (via composition) plus
/// per-step recurrent state `hidden` / `cell`, each shaped
/// `[num_steps, num_envs, hidden_dim]`.
#[derive(Debug, Clone)]
pub struct RecurrentRolloutBuffer {
    /// Feedforward transition storage + GAE. Kept private so the recurrent
    /// buffer's own accessors are the only supported surface.
    inner: RolloutBuffer,

    /// Recurrent hidden state entering each step
    /// `[num_steps, num_envs, hidden_dim]`.
    hidden: Vec<Vec<Vec<f32>>>,

    /// Recurrent cell state entering each step
    /// `[num_steps, num_envs, hidden_dim]`.
    cell: Vec<Vec<Vec<f32>>>,

    /// Width of the recurrent `(h, c)` state.
    hidden_dim: usize,
}

impl RecurrentRolloutBuffer {
    /// Create a new recurrent rollout buffer.
    ///
    /// # Arguments
    /// * `num_steps` - Number of timesteps per rollout
    /// * `num_envs` - Number of parallel environments
    /// * `obs_dim` - Dimensionality of observations
    /// * `hidden_dim` - Width of the LSTM `(h, c)` state
    pub fn new(num_steps: usize, num_envs: usize, obs_dim: usize, hidden_dim: usize) -> Self {
        let inner = RolloutBuffer::new(num_steps, num_envs, obs_dim);
        let hidden = vec![vec![vec![0.0; hidden_dim]; num_envs]; num_steps];
        let cell = vec![vec![vec![0.0; hidden_dim]; num_envs]; num_steps];
        Self { inner, hidden, cell, hidden_dim }
    }

    /// Add a transition to the buffer.
    ///
    /// Delegates verbatim to
    /// [`RolloutBuffer::add`](super::storage::RolloutBuffer::add); see that
    /// method for argument semantics. The recurrent `(h, c)` for this step
    /// is recorded separately via [`Self::add_recurrent_state`].
    // Each argument is a distinct transition field; bundling them into a
    // struct would add boilerplate at every call site without improving
    // clarity (mirrors the feedforward `RolloutBuffer::add`).
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        step: usize,
        env_id: usize,
        observation: &[f32],
        action: i64,
        reward: f32,
        value: f32,
        log_prob: f32,
        terminated: bool,
        truncated: bool,
    ) {
        self.inner.add(
            step,
            env_id,
            observation,
            action,
            reward,
            value,
            log_prob,
            terminated,
            truncated,
        );
    }

    /// Record the recurrent state `(h, c)` **entering** step `step` for
    /// env `env_id`.
    ///
    /// The rollout loop calls this *before* the env step, so the stored
    /// state is the one that entered the step — used for warm-start
    /// verification (via [`Self::seed_warm_start`]) and reserved as the
    /// hook a future Strategy B trainer would reuse. The training forward
    /// itself does not read these back (it recomputes states from a zeroed
    /// `initial_state`).
    ///
    /// # Panics
    /// Panics (debug builds) if `step`/`env_id` are out of range or if
    /// `h`/`c` do not have length `hidden_dim`.
    pub fn add_recurrent_state(&mut self, step: usize, env_id: usize, h: &[f32], c: &[f32]) {
        debug_assert!(step < self.hidden.len(), "step {} out of range", step);
        debug_assert!(env_id < self.hidden[step].len(), "env_id {} out of range", env_id);
        debug_assert_eq!(h.len(), self.hidden_dim, "hidden state dimension mismatch");
        debug_assert_eq!(c.len(), self.hidden_dim, "cell state dimension mismatch");
        self.hidden[step][env_id].copy_from_slice(h);
        self.cell[step][env_id].copy_from_slice(c);
    }

    /// Seed the step-0 recurrent state for the **next** rollout iteration,
    /// in place.
    ///
    /// For each env, if it did **not** end its episode
    /// (`terminated || truncated`) on step `last_step`, its final recurrent
    /// state `final_hidden[env]` / `final_cell[env]` (the state exiting the
    /// last collected step) is carried into `hidden[0][env]` /
    /// `cell[0][env]`. If it **did** end its episode, step 0 is seeded with
    /// zeros — the episode already ended, so no memory should survive into
    /// the fresh one. This is the warm-start half of the GAE/state-reset
    /// asymmetry: the carry decision follows `terminated || truncated`, not
    /// `terminated` alone.
    ///
    /// `final_hidden` / `final_cell` are indexed `[env][hidden_dim]` and
    /// must have `num_envs` rows.
    ///
    /// # Panics
    /// Panics if `last_step >= num_steps`, if `final_hidden` / `final_cell`
    /// do not have `num_envs` rows, or (debug builds) if any row is not
    /// `hidden_dim` wide.
    pub fn seed_warm_start(
        &mut self,
        last_step: usize,
        final_hidden: &[Vec<f32>],
        final_cell: &[Vec<f32>],
    ) {
        let (num_steps, num_envs, _) = self.inner.shape();
        assert!(
            last_step < num_steps,
            "last_step ({}) must be < num_steps ({})",
            last_step,
            num_steps
        );
        assert_eq!(final_hidden.len(), num_envs, "final_hidden must have num_envs rows");
        assert_eq!(final_cell.len(), num_envs, "final_cell must have num_envs rows");

        let terminated = self.inner.terminated();
        let truncated = self.inner.truncated();
        for env in 0..num_envs {
            let ended = terminated[last_step][env] || truncated[last_step][env];
            if ended {
                // Episode ended on the last step — start the next iteration
                // from a zeroed state.
                self.hidden[0][env].iter_mut().for_each(|x| *x = 0.0);
                self.cell[0][env].iter_mut().for_each(|x| *x = 0.0);
            } else {
                debug_assert_eq!(final_hidden[env].len(), self.hidden_dim, "hidden row width");
                debug_assert_eq!(final_cell[env].len(), self.hidden_dim, "cell row width");
                self.hidden[0][env].copy_from_slice(&final_hidden[env]);
                self.cell[0][env].copy_from_slice(&final_cell[env]);
            }
        }
    }

    /// Reset the buffer's advantages/returns for a new rollout.
    ///
    /// Delegates to
    /// [`RolloutBuffer::reset`](super::storage::RolloutBuffer::reset).
    /// The recurrent `(h, c)` arrays are intentionally left in place — they
    /// are overwritten by [`Self::add_recurrent_state`] during the next
    /// collection and [`Self::seed_warm_start`] seeds step 0 explicitly.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Compute GAE over the full `[num_steps, num_envs]` capacity.
    ///
    /// Delegates unchanged to
    /// [`compute_advantages`](super::gae::compute_advantages) — the
    /// `[step][env]` advantage/return grid is byte-identical to the
    /// feedforward path (GAE bootstraps on `terminated` only).
    pub fn compute_advantages(&mut self, last_values: &[f32], gamma: f32, gae_lambda: f32) {
        self.inner.compute_advantages(last_values, gamma, gae_lambda);
    }

    /// Compute GAE over the first `valid_steps` rows.
    ///
    /// Delegates unchanged to
    /// [`compute_advantages_partial`](super::gae::compute_advantages_partial).
    pub fn compute_advantages_partial(
        &mut self,
        valid_steps: usize,
        last_values: &[f32],
        gamma: f32,
        gae_lambda: f32,
    ) {
        self.inner
            .compute_advantages_partial(valid_steps, last_values, gamma, gae_lambda);
    }

    /// Buffer shape `(num_steps, num_envs, obs_dim)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }

    /// Width of the recurrent `(h, c)` state.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    // ---- Delegating getters (read-only views into the inner buffer) ----

    /// Per-step observations, indexed `[step][env]` then by obs dimension.
    pub fn observations(&self) -> &[Vec<Vec<f32>>] {
        self.inner.observations()
    }
    /// Per-step discrete actions, indexed `[step][env]`.
    pub fn actions(&self) -> &[Vec<i64>] {
        self.inner.actions()
    }
    /// Per-step value estimates, indexed `[step][env]`.
    pub fn values(&self) -> &[Vec<f32>] {
        self.inner.values()
    }
    /// Per-step behavior-policy log-probs, indexed `[step][env]`.
    pub fn log_probs(&self) -> &[Vec<f32>] {
        self.inner.log_probs()
    }
    /// Per-step terminal flags, indexed `[step][env]`.
    pub fn terminated(&self) -> &[Vec<bool>] {
        self.inner.terminated()
    }
    /// Per-step truncation flags, indexed `[step][env]`.
    pub fn truncated(&self) -> &[Vec<bool>] {
        self.inner.truncated()
    }
    /// Per-step GAE advantages, indexed `[step][env]`.
    pub fn advantages(&self) -> &[Vec<f32>] {
        self.inner.advantages()
    }
    /// Per-step value-function targets, indexed `[step][env]`.
    pub fn returns(&self) -> &[Vec<f32>] {
        self.inner.returns()
    }

    /// Recurrent hidden state entering each step, indexed
    /// `[step][env]` then by hidden dimension.
    pub fn hidden(&self) -> &[Vec<Vec<f32>>] {
        &self.hidden
    }
    /// Recurrent cell state entering each step, indexed `[step][env]`
    /// then by hidden dimension.
    pub fn cell(&self) -> &[Vec<Vec<f32>>] {
        &self.cell
    }

    /// Materialize the full buffer as a rank-3 sequence batch (`T =
    /// num_steps`).
    ///
    /// Convenience wrapper over [`Self::to_sequence_batch_partial`] with
    /// `valid_steps == num_steps`.
    pub fn to_sequence_batch<B: Backend>(&self, device: &B::Device) -> RecurrentRolloutBatch<B> {
        let num_steps = self.inner.shape().0;
        self.to_sequence_batch_partial::<B>(num_steps, device)
    }

    /// Materialize the first `valid_steps` rows as a rank-3 sequence batch
    /// (`T = valid_steps`).
    ///
    /// Selects **all** envs. See [`RecurrentRolloutBatch`] for the exact
    /// field shapes. Env-major layout: each env-trajectory is one
    /// contiguous row of length `T`.
    ///
    /// # Panics
    /// Panics if `valid_steps > num_steps`.
    pub fn to_sequence_batch_partial<B: Backend>(
        &self,
        valid_steps: usize,
        device: &B::Device,
    ) -> RecurrentRolloutBatch<B> {
        let num_envs = self.inner.shape().1;
        let env_ids: Vec<usize> = (0..num_envs).collect();
        self.sequence_batch_for_envs::<B>(&env_ids, valid_steps, device)
    }

    /// Build an env-major minibatch iterator over full trajectories
    /// (`T = num_steps`).
    ///
    /// Shuffles **environment** indices (not global timesteps) and chunks
    /// them by `envs_per_minibatch` — the recurrent analogue of the
    /// feedforward `batch_size`, counting whole env-trajectories rather
    /// than loose timesteps. Over one epoch every env index appears in
    /// exactly one minibatch.
    ///
    /// When `shuffle` is `false` the env order is `0..num_envs` (useful for
    /// deterministic tests / reproducible evaluation).
    pub fn to_minibatches<'a, B: Backend>(
        &'a self,
        envs_per_minibatch: usize,
        shuffle: bool,
        device: &B::Device,
    ) -> RecurrentMinibatchIterator<'a, B> {
        let num_steps = self.inner.shape().0;
        RecurrentMinibatchIterator::new(self, envs_per_minibatch, num_steps, shuffle, device)
    }

    /// Core materialization: build a [`RecurrentRolloutBatch`] from the
    /// selected `env_ids` over the first `valid_steps` rows.
    ///
    /// Rows are laid out in the order of `env_ids` (env-major); each field
    /// is `[env_ids.len(), valid_steps, ..]`.
    fn sequence_batch_for_envs<B: Backend>(
        &self,
        env_ids: &[usize],
        valid_steps: usize,
        device: &B::Device,
    ) -> RecurrentRolloutBatch<B> {
        let (num_steps, _num_envs, obs_dim) = self.inner.shape();
        assert!(
            valid_steps <= num_steps,
            "valid_steps ({}) must not exceed num_steps ({})",
            valid_steps,
            num_steps
        );

        let n_env = env_ids.len();
        let t = valid_steps;

        let observations = self.inner.observations();
        let actions_grid = self.inner.actions();
        let values_grid = self.inner.values();
        let log_probs_grid = self.inner.log_probs();
        let terminated_grid = self.inner.terminated();
        let truncated_grid = self.inner.truncated();
        let advantages_grid = self.inner.advantages();
        let returns_grid = self.inner.returns();

        let mut obs_flat = Vec::with_capacity(n_env * t * obs_dim);
        let mut actions_flat = Vec::with_capacity(n_env * t);
        let mut starts_flat = Vec::with_capacity(n_env * t);
        let mut log_probs_flat = Vec::with_capacity(n_env * t);
        let mut values_flat = Vec::with_capacity(n_env * t);
        let mut advantages_flat = Vec::with_capacity(n_env * t);
        let mut returns_flat = Vec::with_capacity(n_env * t);

        // Env-major, step-minor: one env-trajectory per contiguous block.
        for &env in env_ids {
            for step in 0..t {
                obs_flat.extend_from_slice(&observations[step][env]);
                actions_flat.push(actions_grid[step][env]);
                // Hidden-state reset mask: `terminated || truncated`
                // (NOT the terminated-only GAE flag).
                let ended = terminated_grid[step][env] || truncated_grid[step][env];
                starts_flat.push(if ended { 1.0_f32 } else { 0.0_f32 });
                log_probs_flat.push(log_probs_grid[step][env]);
                values_flat.push(values_grid[step][env]);
                advantages_flat.push(advantages_grid[step][env]);
                returns_flat.push(returns_grid[step][env]);
            }
        }

        let obs_seq =
            Tensor::<B, 3>::from_data(TensorData::new(obs_flat, [n_env, t, obs_dim]), device);
        let actions =
            Tensor::<B, 2, Int>::from_data(TensorData::new(actions_flat, [n_env, t]), device);
        let episode_starts =
            Tensor::<B, 2>::from_data(TensorData::new(starts_flat, [n_env, t]), device);
        let old_log_probs =
            Tensor::<B, 2>::from_data(TensorData::new(log_probs_flat, [n_env, t]), device);
        let old_values =
            Tensor::<B, 2>::from_data(TensorData::new(values_flat, [n_env, t]), device);
        let advantages =
            Tensor::<B, 2>::from_data(TensorData::new(advantages_flat, [n_env, t]), device);
        let returns = Tensor::<B, 2>::from_data(TensorData::new(returns_flat, [n_env, t]), device);

        RecurrentRolloutBatch {
            obs_seq,
            actions,
            episode_starts,
            old_log_probs,
            old_values,
            advantages,
            returns,
        }
    }
}

/// A rank-3 batch of recurrent rollout data, ready to feed
/// [`LstmBurnPolicy::evaluate_sequences`](crate::policy::lstm::LstmBurnPolicy::evaluate_sequences)
/// with no shape adapters.
///
/// Every field is per-`(env, step)`: `obs_seq` is rank-3
/// `[N_env, T, obs_dim]`, the rest are rank-2 `[N_env, T]`. The recurrent
/// PPO surrogate needs these per-step quantities, not a flattened rank-1
/// batch. `initial_state` is intentionally absent — the training forward
/// always starts from a zeroed `(h, c)` and relies on `episode_starts` for
/// in-sequence resets (design note Q2, Strategy A).
#[derive(Debug)]
pub struct RecurrentRolloutBatch<B: Backend> {
    /// Observations, `[N_env, T, obs_dim]` — feeds `obs_seq`.
    pub obs_seq: Tensor<B, 3>,
    /// Discrete actions, `[N_env, T]` — feeds `actions`.
    pub actions: Tensor<B, 2, Int>,
    /// Episode-boundary (state-reset) mask, `[N_env, T]`: `1.0` where
    /// `terminated || truncated`, else `0.0`. Feeds `episode_starts` — the
    /// hidden-state reset mask, **not** the terminated-only GAE flag.
    pub episode_starts: Tensor<B, 2>,
    /// Behavior-policy log-probs `[N_env, T]` for the PPO ratio.
    pub old_log_probs: Tensor<B, 2>,
    /// Behavior-policy value estimates `V(s_t)`, `[N_env, T]`.
    pub old_values: Tensor<B, 2>,
    /// GAE advantages `[N_env, T]` (terminated-only bootstrap, from GAE).
    pub advantages: Tensor<B, 2>,
    /// Value-function targets `[N_env, T]` (advantages + values).
    pub returns: Tensor<B, 2>,
}

impl<B: Backend> RecurrentRolloutBatch<B> {
    /// Number of env-trajectories (`N_env`) in the batch.
    pub fn num_envs(&self) -> usize {
        self.obs_seq.dims()[0]
    }

    /// Sequence length (`T`) of each trajectory.
    pub fn seq_len(&self) -> usize {
        self.obs_seq.dims()[1]
    }
}

/// Env-major minibatch iterator over whole env-trajectories
/// (full-sequence, Strategy A).
///
/// Yields one [`RecurrentRolloutBatch`] per chunk of `envs_per_minibatch`
/// shuffled env indices. Unlike the feedforward sampler, which shuffles
/// loose timesteps, this shuffles only the env dimension so each
/// trajectory stays temporally intact — episode boundaries are handled by
/// `episode_starts` masking inside the forward, never by cutting
/// sequences. Over one full pass every env appears in exactly one
/// minibatch.
pub struct RecurrentMinibatchIterator<'a, B: Backend> {
    buffer: &'a RecurrentRolloutBuffer,
    device: B::Device,
    /// Shuffled env-id chunks, one per minibatch.
    chunks: Vec<Vec<usize>>,
    valid_steps: usize,
    current: usize,
}

impl<'a, B: Backend> RecurrentMinibatchIterator<'a, B> {
    /// Create a new env-major minibatch iterator.
    ///
    /// * `buffer` - Source recurrent rollout buffer
    /// * `envs_per_minibatch` - Whole env-trajectories per minibatch
    /// * `valid_steps` - Sequence length `T` (rows to materialize)
    /// * `shuffle` - Shuffle env indices (else natural `0..num_envs` order)
    /// * `device` - Device the batch tensors are built on
    pub fn new(
        buffer: &'a RecurrentRolloutBuffer,
        envs_per_minibatch: usize,
        valid_steps: usize,
        shuffle: bool,
        device: &B::Device,
    ) -> Self {
        let num_envs = buffer.shape().1;
        let mut env_ids: Vec<usize> = (0..num_envs).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            env_ids.shuffle(&mut rand::rng());
        }
        // `envs_per_minibatch == 0` would make `chunks` panic; clamp to at
        // least one whole trajectory per minibatch.
        let chunk_len = envs_per_minibatch.max(1);
        let chunks: Vec<Vec<usize>> =
            env_ids.chunks(chunk_len).map(|chunk| chunk.to_vec()).collect();

        Self { buffer, device: device.clone(), chunks, valid_steps, current: 0 }
    }
}

impl<B: Backend> Iterator for RecurrentMinibatchIterator<'_, B> {
    type Item = RecurrentRolloutBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.chunks.len() {
            return None;
        }
        let env_ids = &self.chunks[self.current];
        self.current += 1;
        Some(
            self.buffer
                .sequence_batch_for_envs::<B>(env_ids, self.valid_steps, &self.device),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;
    use crate::buffer::rollout::storage::RolloutBuffer;

    type B = NdArray<f32>;

    fn device() -> <B as burn::tensor::backend::BackendTypes>::Device {
        crate::utils::cuda::default_burn_device::<B>()
    }

    /// Fill a recurrent buffer with deterministic per-`(step, env)` data so
    /// shape/value assertions are exact. Observations encode `(step, env,
    /// dim)` so the env-major flatten order can be verified.
    fn fill_buffer(num_steps: usize, num_envs: usize, obs_dim: usize) -> RecurrentRolloutBuffer {
        let hidden_dim = 3;
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        for step in 0..num_steps {
            for env in 0..num_envs {
                let obs: Vec<f32> =
                    (0..obs_dim).map(|d| (step * 100 + env * 10 + d) as f32).collect();
                buf.add(
                    step,
                    env,
                    &obs,
                    (step + env) as i64,
                    step as f32,          // reward
                    (env as f32) * 0.5,   // value
                    -(step as f32) * 0.1, // log_prob
                    false,
                    false,
                );
            }
        }
        buf
    }

    /// `to_sequence_batch` produces rank-3 `[N_env, T, obs_dim]` obs and
    /// rank-2 `[N_env, T]` everything else, for a 4-env × 8-step buffer.
    #[test]
    fn test_to_sequence_batch_shapes() {
        let buf = fill_buffer(8, 4, 5);
        let dev = device();
        let batch = buf.to_sequence_batch::<B>(&dev);

        assert_eq!(batch.obs_seq.dims(), [4, 8, 5]);
        assert_eq!(batch.actions.dims(), [4, 8]);
        assert_eq!(batch.episode_starts.dims(), [4, 8]);
        assert_eq!(batch.old_log_probs.dims(), [4, 8]);
        assert_eq!(batch.old_values.dims(), [4, 8]);
        assert_eq!(batch.advantages.dims(), [4, 8]);
        assert_eq!(batch.returns.dims(), [4, 8]);
        assert_eq!(batch.num_envs(), 4);
        assert_eq!(batch.seq_len(), 8);
    }

    /// Env-major flatten order: `obs_seq[env, step, dim]` must equal the
    /// value stored at `[step][env][dim]`.
    #[test]
    fn test_to_sequence_batch_env_major_layout() {
        let (num_steps, num_envs, obs_dim) = (3, 2, 4);
        let buf = fill_buffer(num_steps, num_envs, obs_dim);
        let dev = device();
        let batch = buf.to_sequence_batch::<B>(&dev);

        let obs: Vec<f32> = batch.obs_seq.into_data().to_vec().unwrap();
        for env in 0..num_envs {
            for step in 0..num_steps {
                for d in 0..obs_dim {
                    let idx = (env * num_steps + step) * obs_dim + d;
                    let expected = (step * 100 + env * 10 + d) as f32;
                    assert_eq!(obs[idx], expected, "env {} step {} dim {}", env, step, d);
                }
            }
        }
    }

    /// `episode_starts[step][env] == 1.0` iff `terminated || truncated`,
    /// across truncated-only, terminated-only, both, and neither.
    #[test]
    fn test_episode_starts_flag_correctness() {
        let (num_steps, num_envs, obs_dim) = (4, 1, 2);
        let hidden_dim = 2;
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        // step 0: neither, step 1: terminated only, step 2: truncated only,
        // step 3: both.
        let flags = [(false, false), (true, false), (false, true), (true, true)];
        for (step, &(term, trunc)) in flags.iter().enumerate() {
            buf.add(step, 0, &[0.0, 0.0], 0, 0.0, 0.0, 0.0, term, trunc);
        }

        let dev = device();
        let batch = buf.to_sequence_batch::<B>(&dev);
        let starts: Vec<f32> = batch.episode_starts.into_data().to_vec().unwrap();
        // Single env, so flat order is just step order.
        assert_eq!(starts, vec![0.0, 1.0, 1.0, 1.0]);
    }

    /// GAE delegation is byte-identical to the feedforward buffer: build a
    /// recurrent and a feedforward buffer with identical rewards/values/
    /// terminated/truncated, run `compute_advantages_partial` on both, and
    /// assert advantages/returns agree element-wise.
    #[test]
    fn test_gae_input_parity_with_feedforward() {
        let (num_steps, num_envs, obs_dim) = (6, 3, 2);
        let hidden_dim = 4;
        let mut rec = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        let mut ff = RolloutBuffer::new(num_steps, num_envs, obs_dim);

        for step in 0..num_steps {
            for env in 0..num_envs {
                let obs = [step as f32, env as f32];
                let action = (step + env) as i64;
                let reward = ((step * 2 + env) as f32).sin();
                let value = ((step + env) as f32) * 0.3;
                let log_prob = -0.1 * (step as f32);
                // Terminate env 0 at step 2, truncate env 1 at step 4. GAE
                // must react to `terminated` only; both buffers see the
                // same flags so results must still match.
                let term = env == 0 && step == 2;
                let trunc = env == 1 && step == 4;
                rec.add(step, env, &obs, action, reward, value, log_prob, term, trunc);
                ff.add(step, env, &obs, action, reward, value, log_prob, term, trunc);
            }
        }

        let last_values = vec![0.7_f32, -0.2, 0.4];
        let (gamma, lam, valid) = (0.99_f32, 0.95_f32, num_steps);
        rec.compute_advantages_partial(valid, &last_values, gamma, lam);
        ff.compute_advantages_partial(valid, &last_values, gamma, lam);

        for step in 0..num_steps {
            for env in 0..num_envs {
                assert!(
                    (rec.advantages()[step][env] - ff.advantages()[step][env]).abs() < 1e-6,
                    "advantage mismatch at [{}][{}]",
                    step,
                    env
                );
                assert!(
                    (rec.returns()[step][env] - ff.returns()[step][env]).abs() < 1e-6,
                    "return mismatch at [{}][{}]",
                    step,
                    env
                );
            }
        }
    }

    /// `add_recurrent_state` round-trips `(h, c)` for every `(step, env)`.
    #[test]
    fn test_add_recurrent_state_round_trip() {
        let (num_steps, num_envs, obs_dim, hidden_dim) = (3, 2, 2, 4);
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        for step in 0..num_steps {
            for env in 0..num_envs {
                let h: Vec<f32> =
                    (0..hidden_dim).map(|k| (step * 1000 + env * 100 + k) as f32).collect();
                let c: Vec<f32> =
                    (0..hidden_dim).map(|k| -((step * 1000 + env * 100 + k) as f32)).collect();
                buf.add_recurrent_state(step, env, &h, &c);
            }
        }
        for step in 0..num_steps {
            for env in 0..num_envs {
                for k in 0..hidden_dim {
                    let expected = (step * 1000 + env * 100 + k) as f32;
                    assert_eq!(buf.hidden()[step][env][k], expected);
                    assert_eq!(buf.cell()[step][env][k], -expected);
                }
            }
        }
    }

    /// Warm-start: an env ended (`terminated || truncated`) on the last
    /// step gets a zeroed step-0 state; an env that did neither carries its
    /// non-zero final state.
    #[test]
    fn test_seed_warm_start_zeros_for_ended_envs() {
        let (num_steps, num_envs, obs_dim, hidden_dim) = (4, 3, 2, 3);
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        let last_step = num_steps - 1;
        // env 0: terminated, env 1: truncated, env 2: neither.
        buf.add(last_step, 0, &[0.0, 0.0], 0, 0.0, 0.0, 0.0, true, false);
        buf.add(last_step, 1, &[0.0, 0.0], 0, 0.0, 0.0, 0.0, false, true);
        buf.add(last_step, 2, &[0.0, 0.0], 0, 0.0, 0.0, 0.0, false, false);

        let final_hidden = vec![vec![1.0, 2.0, 3.0]; num_envs];
        let final_cell = vec![vec![-1.0, -2.0, -3.0]; num_envs];
        buf.seed_warm_start(last_step, &final_hidden, &final_cell);

        // Ended envs -> zeros.
        assert_eq!(buf.hidden()[0][0], vec![0.0, 0.0, 0.0]);
        assert_eq!(buf.cell()[0][0], vec![0.0, 0.0, 0.0]);
        assert_eq!(buf.hidden()[0][1], vec![0.0, 0.0, 0.0]);
        assert_eq!(buf.cell()[0][1], vec![0.0, 0.0, 0.0]);
        // Live env -> carries the final state.
        assert_eq!(buf.hidden()[0][2], vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.cell()[0][2], vec![-1.0, -2.0, -3.0]);
    }

    /// Env-major sampler: over one epoch every env index appears exactly
    /// once across all minibatches (no dup, full coverage), with the last
    /// chunk shorter when `num_envs % envs_per_minibatch != 0`.
    #[test]
    fn test_env_major_sampler_coverage() {
        let (num_steps, num_envs, obs_dim) = (5, 7, 2);
        let buf = fill_buffer(num_steps, num_envs, obs_dim);
        let dev = device();
        let envs_per_minibatch = 3;

        let mut seen = std::collections::HashSet::new();
        let mut total = 0usize;
        let mut n_batches = 0usize;
        for batch in buf.to_minibatches::<B>(envs_per_minibatch, true, &dev) {
            n_batches += 1;
            assert!(batch.num_envs() <= envs_per_minibatch);
            assert_eq!(batch.seq_len(), num_steps);
            // Recover which envs are in this batch from the obs' step-0
            // dim-1 encoding: obs[env,0,1] = 0*100 + env*10 + 1.
            let obs: Vec<f32> = batch.obs_seq.clone().into_data().to_vec().unwrap();
            for e in 0..batch.num_envs() {
                // Step 0, dim 1: obs[env,0,1] = env*10 + 1.
                let idx = (e * num_steps) * obs_dim + 1;
                let env = ((obs[idx] as usize) - 1) / 10;
                assert!(seen.insert(env), "env {} appeared twice", env);
                total += 1;
            }
        }
        assert_eq!(total, num_envs, "every env covered exactly once");
        assert_eq!(seen.len(), num_envs);
        // ceil(7 / 3) == 3 minibatches.
        assert_eq!(n_batches, 3);
    }

    /// Sampler minibatch shapes: `obs_seq` is `[envs_per_minibatch, T,
    /// obs_dim]` and `episode_starts` is `[envs_per_minibatch, T]` for a
    /// full chunk.
    #[test]
    fn test_sampler_minibatch_shape() {
        let (num_steps, num_envs, obs_dim) = (6, 4, 3);
        let buf = fill_buffer(num_steps, num_envs, obs_dim);
        let dev = device();
        // 4 envs / 2 per minibatch => exactly 2 full chunks.
        let batches: Vec<_> = buf.to_minibatches::<B>(2, false, &dev).collect();
        assert_eq!(batches.len(), 2);
        for batch in &batches {
            assert_eq!(batch.obs_seq.dims(), [2, num_steps, obs_dim]);
            assert_eq!(batch.episode_starts.dims(), [2, num_steps]);
            assert_eq!(batch.actions.dims(), [2, num_steps]);
        }
        // Unshuffled order: first chunk = envs {0,1}, second = {2,3}.
        let obs0: Vec<f32> = batches[0].obs_seq.clone().into_data().to_vec().unwrap();
        // obs[env=0, step=0, dim=0] == 0.
        assert_eq!(obs0[0], 0.0);
    }

    /// End-to-end: the batch fields feed
    /// `LstmBurnPolicy::evaluate_sequences` with no shape adapters, and the
    /// three outputs come back `[N_env, T]`.
    #[test]
    fn test_evaluate_sequences_integration() {
        use crate::policy::lstm::{LstmBurnConfig, LstmBurnPolicy};
        type AB = burn::backend::Autodiff<NdArray<f32>>;

        let (num_steps, num_envs, obs_dim, action_dim) = (5, 3, 4, 2);
        let dev = crate::utils::cuda::default_burn_device::<AB>();

        let hidden_dim = 8;
        let mut buf = RecurrentRolloutBuffer::new(num_steps, num_envs, obs_dim, hidden_dim);
        for step in 0..num_steps {
            for env in 0..num_envs {
                let obs: Vec<f32> = (0..obs_dim).map(|d| 0.1 * (step + env + d) as f32).collect();
                let term = env == 0 && step == 3;
                buf.add(step, env, &obs, (env % action_dim) as i64, 0.0, 0.0, 0.0, term, false);
            }
        }

        let batch = buf.to_sequence_batch::<AB>(&dev);
        let cfg = LstmBurnConfig { hidden_dim, ..Default::default() }.with_seed(11);
        let policy = LstmBurnPolicy::<AB>::with_config(obs_dim, action_dim, cfg, &dev);

        // No adapters: pass the batch fields straight through. `None`
        // initial state — Strategy A resets internally via episode_starts.
        let (log_probs, entropy, values) =
            policy.evaluate_sequences(batch.obs_seq, batch.actions, None, batch.episode_starts);
        assert_eq!(log_probs.dims(), [num_envs, num_steps]);
        assert_eq!(entropy.dims(), [num_envs, num_steps]);
        assert_eq!(values.dims(), [num_envs, num_steps]);
    }
}
