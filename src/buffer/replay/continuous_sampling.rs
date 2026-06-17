//! Uniform random sampling and tensor conversion for the continuous-action
//! replay buffer.
//!
//! `sample` draws `batch_size` independent uniform indices from the filled
//! portion of a [`super::continuous_storage::ContinuousReplayBuffer`] and
//! copies the transitions into flat CPU-side vectors.
//! [`ContinuousReplayBatch::to_burn_tensors`] then stacks those vectors into
//! Burn tensors on the requested device so the trainer can run actor/critic
//! forward passes.
//!
//! This mirrors [`super::sampling`] but with continuous `f32` actions of
//! width `action_dim` (shape `[batch, action_dim]`) rather than discrete
//! `i64` actions.

use burn::tensor::{Tensor as BurnTensor, TensorData, backend::Backend};
use rand::Rng;

use super::continuous_storage::ContinuousReplayBuffer;

/// One minibatch sampled from the continuous replay buffer.
///
/// All fields are CPU-side primitive vectors; convert to Burn tensors via
/// [`ContinuousReplayBatch::to_burn_tensors`] when handing them to the
/// trainer.
#[derive(Debug, Clone)]
pub struct ContinuousReplayBatch {
    /// Flattened current observations, shape `[batch_size * obs_dim]`.
    pub observations: Vec<f32>,
    /// Flattened actions taken, shape `[batch_size * action_dim]`.
    pub actions: Vec<f32>,
    /// Rewards received, length `batch_size`.
    pub rewards: Vec<f32>,
    /// Flattened next observations, shape `[batch_size * obs_dim]`.
    pub next_observations: Vec<f32>,
    /// Episode-end mask, length `batch_size`. `true` means the transition
    /// terminated the episode (so the TD target drops the bootstrap term).
    pub dones: Vec<bool>,
    /// Length of one observation slice (so `to_burn_tensors` can reshape).
    pub obs_dim: usize,
    /// Length of one action slice (so `to_burn_tensors` can reshape).
    pub action_dim: usize,
}

impl ContinuousReplayBatch {
    /// Number of transitions in the batch.
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// `true` if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }

    /// Stack the batch into Burn tensors on `device`.
    ///
    /// Returns a named [`ContinuousReplayBurnTensors`] struct so trainers can
    /// pattern-match named fields rather than positional tuple elements.
    ///
    /// Shapes (all on `device`):
    /// - `observations`: `[batch, obs_dim]`, `f32`
    /// - `actions`: `[batch, action_dim]`, `f32`
    /// - `rewards`: `[batch]`, `f32`
    /// - `next_observations`: `[batch, obs_dim]`, `f32`
    /// - `dones`: `[batch]`, `f32` (0.0 / 1.0; the TD-target formula `(1 -
    ///   done)` consumes it directly as a float mask).
    pub fn to_burn_tensors<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ContinuousReplayBurnTensors<B> {
        let batch = self.len();
        let obs_dim = self.obs_dim;
        let action_dim = self.action_dim;

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
        let actions = BurnTensor::<B, 2>::from_data(
            TensorData::new(self.actions.clone(), [batch, action_dim]),
            device,
        );
        let rewards =
            BurnTensor::<B, 1>::from_data(TensorData::new(self.rewards.clone(), [batch]), device);
        let dones_f: Vec<f32> = self.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
        let dones = BurnTensor::<B, 1>::from_data(TensorData::new(dones_f, [batch]), device);

        ContinuousReplayBurnTensors { observations, actions, rewards, next_observations, dones }
    }
}

/// Bundle of Burn tensors produced by
/// [`ContinuousReplayBatch::to_burn_tensors`].
///
/// Fields are in the order off-policy continuous-control trainers consume
/// them: state and action first, then the reward signal, then the bootstrap
/// state and terminal mask used to build the TD target. Fields are named so
/// downstream patches that add fields stay source-compatible.
#[derive(Debug)]
pub struct ContinuousReplayBurnTensors<B: Backend> {
    /// Observations, shape `[batch, obs_dim]`, dtype `f32`.
    pub observations: BurnTensor<B, 2>,
    /// Continuous actions, shape `[batch, action_dim]`, dtype `f32`.
    pub actions: BurnTensor<B, 2>,
    /// Rewards, shape `[batch]`, dtype `f32`.
    pub rewards: BurnTensor<B, 1>,
    /// Bootstrap-state observations, shape `[batch, obs_dim]`, dtype `f32`.
    pub next_observations: BurnTensor<B, 2>,
    /// Episode-terminal mask (0.0 or 1.0), shape `[batch]`, dtype `f32`.
    /// Kept as `f32` so the TD-target formula `(1 - done)` works directly.
    pub dones: BurnTensor<B, 1>,
}

/// Sample `batch_size` transitions uniformly with replacement from the
/// filled portion of `buffer`.
///
/// Uniform with-replacement matches the canonical off-policy recipe and is
/// what makes the sampler O(1) per draw. For small batches relative to
/// buffer size the difference vs without-replacement is negligible.
///
/// # Panics
/// Panics if `buffer.is_empty()` or `batch_size == 0`.
pub fn sample<R: Rng>(
    buffer: &ContinuousReplayBuffer,
    batch_size: usize,
    rng: &mut R,
) -> ContinuousReplayBatch {
    assert!(!buffer.is_empty(), "ContinuousReplayBuffer is empty; cannot sample");
    assert!(batch_size > 0, "batch_size must be > 0");

    let obs_dim = buffer.obs_dim();
    let action_dim = buffer.action_dim();
    let len = buffer.len();

    let mut observations = vec![0.0f32; batch_size * obs_dim];
    let mut next_observations = vec![0.0f32; batch_size * obs_dim];
    let mut actions = vec![0.0f32; batch_size * action_dim];
    let mut rewards = Vec::with_capacity(batch_size);
    let mut dones = Vec::with_capacity(batch_size);

    for k in 0..batch_size {
        let idx = rng.random_range(0..len);
        let obs_slice = &mut observations[k * obs_dim..(k + 1) * obs_dim];
        let next_slice = &mut next_observations[k * obs_dim..(k + 1) * obs_dim];
        let action_slice = &mut actions[k * action_dim..(k + 1) * action_dim];
        let (r, d) = buffer.read_into(idx, obs_slice, action_slice, next_slice);
        rewards.push(r);
        dones.push(d);
    }

    ContinuousReplayBatch {
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        obs_dim,
        action_dim,
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_sample_returns_correct_count() {
        let mut buf = ContinuousReplayBuffer::new(16, 3, 2);
        for i in 0..10 {
            buf.push(
                &[i as f32, i as f32 + 0.1, i as f32 + 0.2],
                &[i as f32, -(i as f32)],
                i as f32,
                &[(i + 1) as f32, (i + 1) as f32 + 0.1, (i + 1) as f32 + 0.2],
                false,
            );
        }
        let mut rng = StdRng::seed_from_u64(42);
        let batch = sample(&buf, 5, &mut rng);
        assert_eq!(batch.len(), 5);
        assert_eq!(batch.rewards.len(), 5);
        assert_eq!(batch.dones.len(), 5);
        assert_eq!(batch.observations.len(), 5 * 3);
        assert_eq!(batch.next_observations.len(), 5 * 3);
        assert_eq!(batch.actions.len(), 5 * 2);
        assert_eq!(batch.obs_dim, 3);
        assert_eq!(batch.action_dim, 2);
    }

    #[test]
    fn test_sampled_values_match_pushed_values() {
        // Push a single transition; every sample must return it.
        let mut buf = ContinuousReplayBuffer::new(8, 2, 3);
        buf.push(&[7.0, 8.0], &[0.1, 0.2, 0.3], 42.0, &[9.0, 10.0], true);

        let mut rng = StdRng::seed_from_u64(0);
        let batch = sample(&buf, 4, &mut rng);
        for k in 0..4 {
            assert_eq!(batch.rewards[k], 42.0);
            assert!(batch.dones[k]);
            assert_eq!(&batch.observations[k * 2..(k + 1) * 2], &[7.0, 8.0]);
            assert_eq!(&batch.next_observations[k * 2..(k + 1) * 2], &[9.0, 10.0]);
            assert_eq!(&batch.actions[k * 3..(k + 1) * 3], &[0.1, 0.2, 0.3]);
        }
    }

    mod burn_tests {
        use burn::backend::NdArray;

        use super::*;

        type B = NdArray<f32>;

        #[test]
        fn test_to_burn_tensors_shapes_and_roundtrip() {
            let mut buf = ContinuousReplayBuffer::new(8, 4, 2);
            for i in 0..6 {
                buf.push(
                    &[i as f32; 4],
                    &[i as f32, i as f32 + 0.5],
                    i as f32,
                    &[i as f32 + 1.0; 4],
                    i == 5,
                );
            }
            let mut rng = StdRng::seed_from_u64(1);
            let batch = sample(&buf, 3, &mut rng);
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);

            // Shapes.
            assert_eq!(t.observations.dims(), [3, 4]);
            assert_eq!(t.next_observations.dims(), [3, 4]);
            assert_eq!(t.actions.dims(), [3, 2]);
            assert_eq!(t.rewards.dims(), [3]);
            assert_eq!(t.dones.dims(), [3]);

            // Round-trip: copying out the host data should match the
            // CPU-side `Vec`s the batch was built from.
            let obs_flat: Vec<f32> = t.observations.into_data().to_vec().unwrap();
            assert_eq!(obs_flat, batch.observations);
            let next_flat: Vec<f32> = t.next_observations.into_data().to_vec().unwrap();
            assert_eq!(next_flat, batch.next_observations);
            let acts: Vec<f32> = t.actions.into_data().to_vec().unwrap();
            assert_eq!(acts, batch.actions);
            let rews: Vec<f32> = t.rewards.into_data().to_vec().unwrap();
            assert_eq!(rews, batch.rewards);
            let dones_f: Vec<f32> = t.dones.into_data().to_vec().unwrap();
            let expected_dones: Vec<f32> =
                batch.dones.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
            assert_eq!(dones_f, expected_dones);
        }

        #[test]
        fn test_to_burn_tensors_empty_batch_does_not_panic() {
            // A 0-row batch must produce well-formed zero-element tensors.
            // The buffer-side `sample` API doesn't accept batch_size == 0,
            // so build the batch by hand to exercise the edge case.
            let batch = ContinuousReplayBatch {
                observations: vec![],
                actions: vec![],
                rewards: vec![],
                next_observations: vec![],
                dones: vec![],
                obs_dim: 4,
                action_dim: 2,
            };
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);
            assert_eq!(t.observations.dims(), [0, 4]);
            assert_eq!(t.next_observations.dims(), [0, 4]);
            assert_eq!(t.actions.dims(), [0, 2]);
            assert_eq!(t.rewards.dims(), [0]);
            assert_eq!(t.dones.dims(), [0]);
        }
    }

    #[test]
    #[should_panic(expected = "ContinuousReplayBuffer is empty")]
    fn test_sample_empty_panics() {
        let buf = ContinuousReplayBuffer::new(4, 2, 1);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample(&buf, 2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn test_zero_batch_size_panics() {
        let mut buf = ContinuousReplayBuffer::new(4, 2, 1);
        buf.push(&[0.0, 0.0], &[0.0], 0.0, &[0.0, 0.0], false);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample(&buf, 0, &mut rng);
    }
}
