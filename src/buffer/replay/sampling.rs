//! Uniform random sampling and tensor conversion for the replay buffer.
//!
//! `sample` draws `batch_size` independent uniform indices from the
//! filled portion of a [`super::storage::ReplayBuffer`] and copies the
//! transitions into flat CPU-side vectors. `ReplayBatch::to_tensors`
//! then stacks those vectors into `tch::Tensor`s on the requested
//! device so the trainer can run a single Q-network forward pass.
//!
//! For the in-flight Burn migration (#65), `ReplayBatch::to_burn_tensors`
//! produces the equivalent bundle as Burn tensors. The two methods are
//! parallel surfaces, gated by `feature = "training"` and
//! `feature = "training-burn"` respectively; nothing in the storage
//! layout itself depends on the choice of tensor backend.

use rand::Rng;
#[cfg(feature = "training")]
use tch::{Device, Tensor};

#[cfg(feature = "training-burn")]
use burn::tensor::{Int, Tensor as BurnTensor, TensorData, backend::Backend};

use super::storage::ReplayBuffer;

/// One minibatch sampled from the replay buffer.
///
/// All fields are CPU-side primitive vectors; convert to `tch::Tensor`s
/// via [`ReplayBatch::to_tensors`] when handing them to the trainer.
#[derive(Debug, Clone)]
pub struct ReplayBatch {
    /// Flattened current observations, shape `[batch_size * obs_dim]`.
    pub observations: Vec<f32>,
    /// Actions taken, length `batch_size`.
    pub actions: Vec<i64>,
    /// Rewards received, length `batch_size`.
    pub rewards: Vec<f32>,
    /// Flattened next observations, shape `[batch_size * obs_dim]`.
    pub next_observations: Vec<f32>,
    /// Episode-end mask, length `batch_size`. `true` means the transition
    /// terminated the episode (so the TD target drops the bootstrap term).
    pub dones: Vec<bool>,
    /// Length of one observation slice (so `to_tensors` can reshape).
    pub obs_dim: usize,
}

impl ReplayBatch {
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
    /// Returns `(obs, actions, rewards, next_obs, dones)` with shapes:
    /// - `obs`: `[batch, obs_dim]`, `Kind::Float`
    /// - `actions`: `[batch]`, `Kind::Int64`
    /// - `rewards`: `[batch]`, `Kind::Float`
    /// - `next_obs`: `[batch, obs_dim]`, `Kind::Float`
    /// - `dones`: `[batch]`, `Kind::Float` (0.0 or 1.0 — Kind::Float is what
    ///   the TD-target formula `(1 - done)` needs).
    #[cfg(feature = "training")]
    pub fn to_tensors(&self, device: Device) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
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

        (obs, actions, rewards, next_obs, dones)
    }

    /// Stack the batch into Burn tensors on `device`.
    ///
    /// Parallel to [`Self::to_tensors`] but emits a named
    /// [`ReplayBurnTensors`] struct so trainers can pattern-match named
    /// fields rather than positional tuple elements. Mirrors the
    /// `RolloutTchTensors` convention introduced in #86.
    ///
    /// Shapes (all on `device`):
    /// - `observations`: `[batch, obs_dim]`, `f32`
    /// - `actions`: `[batch]`, `i64` (Burn `Int` kind)
    /// - `rewards`: `[batch]`, `f32`
    /// - `next_observations`: `[batch, obs_dim]`, `f32`
    /// - `dones`: `[batch]`, `f32` (0.0 / 1.0 — same convention as the
    ///   tch path so the `(1 - done)` TD-target formula carries over).
    #[cfg(feature = "training-burn")]
    pub fn to_burn_tensors<B: Backend>(&self, device: &B::Device) -> ReplayBurnTensors<B> {
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

        ReplayBurnTensors { observations, actions, rewards, next_observations, dones }
    }
}

/// Bundle of Burn tensors produced by [`ReplayBatch::to_burn_tensors`].
///
/// Fields are in the order DQN trainers consume them: state and action
/// first, then the reward signal, then the bootstrap state and terminal
/// mask used to build the TD target. Parallel to the tch path's positional
/// 5-tuple but named so callers can grab fields by name (which keeps
/// downstream patches that add fields source-compatible).
#[cfg(feature = "training-burn")]
#[derive(Debug)]
pub struct ReplayBurnTensors<B: Backend> {
    /// Observations, shape `[batch, obs_dim]`, dtype `f32`.
    pub observations: BurnTensor<B, 2>,
    /// Discrete actions, shape `[batch]`, dtype `i64`.
    pub actions: BurnTensor<B, 1, Int>,
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
/// Uniform with-replacement matches the canonical DQN recipe and is
/// what makes the sampler O(1) per draw. For small batches relative to
/// buffer size the difference vs without-replacement is negligible.
///
/// # Panics
/// Panics if `buffer.is_empty()` or `batch_size == 0`.
pub fn sample<R: Rng>(buffer: &ReplayBuffer, batch_size: usize, rng: &mut R) -> ReplayBatch {
    assert!(!buffer.is_empty(), "ReplayBuffer is empty; cannot sample");
    assert!(batch_size > 0, "batch_size must be > 0");

    let obs_dim = buffer.obs_dim();
    let len = buffer.len();

    let mut observations = vec![0.0f32; batch_size * obs_dim];
    let mut next_observations = vec![0.0f32; batch_size * obs_dim];
    let mut actions = Vec::with_capacity(batch_size);
    let mut rewards = Vec::with_capacity(batch_size);
    let mut dones = Vec::with_capacity(batch_size);

    for k in 0..batch_size {
        let idx = rng.random_range(0..len);
        let obs_slice = &mut observations[k * obs_dim..(k + 1) * obs_dim];
        let next_slice = &mut next_observations[k * obs_dim..(k + 1) * obs_dim];
        let (a, r, d) = buffer.read_into(idx, obs_slice, next_slice);
        actions.push(a);
        rewards.push(r);
        dones.push(d);
    }

    ReplayBatch { observations, actions, rewards, next_observations, dones, obs_dim }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};
    #[cfg(feature = "training")]
    use tch::Kind;

    use super::*;

    #[test]
    fn test_sample_returns_correct_count() {
        let mut buf = ReplayBuffer::new(16, 3);
        for i in 0..10 {
            buf.push(
                &[i as f32, i as f32 + 0.1, i as f32 + 0.2],
                (i % 2) as i64,
                i as f32,
                &[(i + 1) as f32, (i + 1) as f32 + 0.1, (i + 1) as f32 + 0.2],
                false,
            );
        }
        let mut rng = StdRng::seed_from_u64(42);
        let batch = sample(&buf, 5, &mut rng);
        assert_eq!(batch.len(), 5);
        assert_eq!(batch.actions.len(), 5);
        assert_eq!(batch.rewards.len(), 5);
        assert_eq!(batch.dones.len(), 5);
        assert_eq!(batch.observations.len(), 5 * 3);
        assert_eq!(batch.next_observations.len(), 5 * 3);
        assert_eq!(batch.obs_dim, 3);
    }

    #[test]
    fn test_sampled_values_match_pushed_values() {
        // Push a single transition; every sample must return it.
        let mut buf = ReplayBuffer::new(8, 2);
        buf.push(&[7.0, 8.0], 1, 42.0, &[9.0, 10.0], true);

        let mut rng = StdRng::seed_from_u64(0);
        let batch = sample(&buf, 4, &mut rng);
        for k in 0..4 {
            assert_eq!(batch.actions[k], 1);
            assert_eq!(batch.rewards[k], 42.0);
            assert!(batch.dones[k]);
            assert_eq!(&batch.observations[k * 2..(k + 1) * 2], &[7.0, 8.0]);
            assert_eq!(&batch.next_observations[k * 2..(k + 1) * 2], &[9.0, 10.0]);
        }
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_to_tensors_shapes() {
        let mut buf = ReplayBuffer::new(8, 4);
        for i in 0..6 {
            buf.push(&[i as f32; 4], (i % 2) as i64, i as f32, &[i as f32 + 1.0; 4], i == 5);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let batch = sample(&buf, 3, &mut rng);
        let (obs, actions, rewards, next_obs, dones) = batch.to_tensors(Device::Cpu);
        assert_eq!(obs.size(), vec![3, 4]);
        assert_eq!(next_obs.size(), vec![3, 4]);
        assert_eq!(actions.size(), vec![3]);
        assert_eq!(rewards.size(), vec![3]);
        assert_eq!(dones.size(), vec![3]);
        assert_eq!(actions.kind(), Kind::Int64);
        assert_eq!(rewards.kind(), Kind::Float);
        // dones tensor is float-valued so it can be used in (1 - done)
        assert_eq!(dones.kind(), Kind::Float);
    }

    #[cfg(feature = "training-burn")]
    mod burn_tests {
        use burn::backend::NdArray;

        use super::*;

        type B = NdArray<f32>;

        #[test]
        fn test_to_burn_tensors_shapes_and_roundtrip() {
            let mut buf = ReplayBuffer::new(8, 4);
            for i in 0..6 {
                buf.push(&[i as f32; 4], (i % 2) as i64, i as f32, &[i as f32 + 1.0; 4], i == 5);
            }
            let mut rng = StdRng::seed_from_u64(1);
            let batch = sample(&buf, 3, &mut rng);
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);

            // Shapes.
            assert_eq!(t.observations.dims(), [3, 4]);
            assert_eq!(t.next_observations.dims(), [3, 4]);
            assert_eq!(t.actions.dims(), [3]);
            assert_eq!(t.rewards.dims(), [3]);
            assert_eq!(t.dones.dims(), [3]);

            // Round-trip: copying out the host data should match the
            // CPU-side `Vec`s the batch was built from. Observations are
            // reshaped to `[3, 4]` but the underlying buffer order is the
            // same row-major flatten.
            let obs_flat: Vec<f32> = t.observations.into_data().to_vec().unwrap();
            assert_eq!(obs_flat, batch.observations);
            let next_flat: Vec<f32> = t.next_observations.into_data().to_vec().unwrap();
            assert_eq!(next_flat, batch.next_observations);
            let acts: Vec<i64> = t.actions.into_data().to_vec().unwrap();
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
            let batch = ReplayBatch {
                observations: vec![],
                actions: vec![],
                rewards: vec![],
                next_observations: vec![],
                dones: vec![],
                obs_dim: 4,
            };
            let device = crate::utils::cuda::default_burn_device::<B>();
            let t = batch.to_burn_tensors::<B>(&device);
            assert_eq!(t.observations.dims(), [0, 4]);
            assert_eq!(t.next_observations.dims(), [0, 4]);
            assert_eq!(t.actions.dims(), [0]);
            assert_eq!(t.rewards.dims(), [0]);
            assert_eq!(t.dones.dims(), [0]);
        }
    }

    #[test]
    #[should_panic(expected = "ReplayBuffer is empty")]
    fn test_sample_empty_panics() {
        let buf = ReplayBuffer::new(4, 2);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample(&buf, 2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn test_zero_batch_size_panics() {
        let mut buf = ReplayBuffer::new(4, 2);
        buf.push(&[0.0, 0.0], 0, 0.0, &[0.0, 0.0], false);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = sample(&buf, 0, &mut rng);
    }
}
