//! Uniform random sampling and tensor conversion for the replay buffer.
//!
//! `sample` draws `batch_size` independent uniform indices from the
//! filled portion of a [`super::storage::ReplayBuffer`] and copies the
//! transitions into flat CPU-side vectors. `ReplayBatch::to_tensors`
//! then stacks those vectors into `tch::Tensor`s on the requested
//! device so the trainer can run a single Q-network forward pass.

use rand::Rng;
use tch::{Device, Tensor};

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
