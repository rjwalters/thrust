//! Fixed dataset of expert demonstrations for Behavioral Cloning.
//!
//! [`Demonstrations`] is a small, flat supervised dataset of discrete-action
//! demonstrations: a contiguous block of observations paired with the expert
//! action taken at each. It is intentionally NOT a reuse of
//! [`crate::buffer::rollout::RolloutBuffer`] or
//! [`crate::buffer::replay::ReplayBuffer`], which carry rewards, dones,
//! advantages, and priorities that supervised imitation has no use for (BC
//! epic #161, decision 2).
//!
//! Seeded minibatching reuses PPO's
//! [`generate_minibatch_indices_with_rng`](crate::train::ppo::loss::generate_minibatch_indices_with_rng)
//! so the BC trainer (#167) can produce bit-reproducible shuffles from a
//! [`BcConfig`](crate::train::bc::BcConfig) seed.
//!
//! This is the dataset half of PR A of the BC decomposition (#164). A
//! `from_file` constructor can be layered on later without disturbing the
//! in-memory API.

use anyhow::{Result, anyhow};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// A fixed dataset of expert `(observation, action)` pairs for supervised
/// imitation learning.
///
/// Observations are stored as a single flat row-major `Vec<f32>` of shape
/// `[len, obs_dim]`; the discrete expert action for example `i` lives at
/// `actions[i]`. The dataset is immutable once constructed and serves
/// host-side minibatches by index via [`Demonstrations::batch`].
#[derive(Debug, Clone)]
pub struct Demonstrations {
    /// Flat row-major observation buffer of shape `[len, obs_dim]`.
    obs: Vec<f32>,
    /// Discrete expert action per example, length `len`.
    actions: Vec<i64>,
    /// Dimensionality of a single observation.
    obs_dim: usize,
    /// Number of `(observation, action)` examples.
    len: usize,
}

impl Demonstrations {
    /// Build a dataset from a flat observation buffer and a parallel action
    /// vector.
    ///
    /// `obs` must be row-major of length `len * obs_dim` and `actions` must
    /// have length `len`, where `len` is inferred from `actions`. Returns an
    /// `Err` if the lengths are inconsistent or `obs_dim` is zero.
    ///
    /// # Arguments
    ///
    /// * `obs` - Flat row-major observations of shape `[len, obs_dim]`.
    /// * `actions` - Expert action index per example.
    /// * `obs_dim` - Dimensionality of a single observation.
    pub fn new(obs: Vec<f32>, actions: Vec<i64>, obs_dim: usize) -> Result<Self> {
        if obs_dim == 0 {
            return Err(anyhow!("obs_dim must be positive"));
        }
        let len = actions.len();
        if obs.len() != len * obs_dim {
            return Err(anyhow!(
                "obs length {} does not match actions.len() ({}) * obs_dim ({}) = {}",
                obs.len(),
                len,
                obs_dim,
                len * obs_dim
            ));
        }
        Ok(Self { obs, actions, obs_dim, len })
    }

    /// Number of `(observation, action)` examples in the dataset.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the dataset contains no examples.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Dimensionality of a single observation.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Gather the given example indices into a minibatch of tensors.
    ///
    /// Returns `(obs, actions)` where `obs` has shape `[k, obs_dim]` and
    /// `actions` has shape `[k]`, with `k = indices.len()`. Gathering is done
    /// host-side; the resulting tensors are plain inputs/labels with no
    /// gradient tracking concerns.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds (`>= len`).
    ///
    /// # Arguments
    ///
    /// * `indices` - Example indices to gather into the minibatch.
    /// * `device` - Backend device the tensors are allocated on.
    pub fn batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        let k = indices.len();
        let mut obs_data = Vec::with_capacity(k * self.obs_dim);
        let mut action_data = Vec::with_capacity(k);

        for &i in indices {
            assert!(i < self.len, "index {} out of bounds for dataset of len {}", i, self.len);
            let start = i * self.obs_dim;
            obs_data.extend_from_slice(&self.obs[start..start + self.obs_dim]);
            action_data.push(self.actions[i]);
        }

        let obs = Tensor::<B, 2>::from_data(TensorData::new(obs_data, [k, self.obs_dim]), device);
        let actions = Tensor::<B, 1, Int>::from_data(TensorData::new(action_data, [k]), device);
        (obs, actions)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;
    use crate::train::ppo::loss::generate_minibatch_indices_with_rng;

    type B = Autodiff<NdArray<f32>>;

    fn sample_dataset() -> Demonstrations {
        // 4 examples, obs_dim = 2.
        let obs = vec![
            0.0, 0.1, // ex 0
            1.0, 1.1, // ex 1
            2.0, 2.1, // ex 2
            3.0, 3.1, // ex 3
        ];
        let actions = vec![0i64, 1, 0, 1];
        Demonstrations::new(obs, actions, 2).unwrap()
    }

    #[test]
    fn test_new_validates_dimensions() {
        // Valid: 4 actions * obs_dim 2 == 8 floats.
        assert!(Demonstrations::new(vec![0.0; 8], vec![0i64; 4], 2).is_ok());

        // obs too short.
        assert!(Demonstrations::new(vec![0.0; 6], vec![0i64; 4], 2).is_err());

        // obs too long.
        assert!(Demonstrations::new(vec![0.0; 10], vec![0i64; 4], 2).is_err());

        // obs_dim zero rejected.
        assert!(Demonstrations::new(vec![], vec![], 0).is_err());
    }

    #[test]
    fn test_len_and_is_empty() {
        let ds = sample_dataset();
        assert_eq!(ds.len(), 4);
        assert_eq!(ds.obs_dim(), 2);
        assert!(!ds.is_empty());

        let empty = Demonstrations::new(vec![], vec![], 2).unwrap();
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_batch_shapes_and_contents() {
        let ds = sample_dataset();
        let device = Default::default();

        let indices = [2usize, 1];
        let (obs, actions) = ds.batch::<B>(&indices, &device);

        assert_eq!(obs.dims(), [2, 2]);
        assert_eq!(actions.dims(), [2]);

        let obs_vals: Vec<f32> = obs.into_data().to_vec().unwrap();
        // Row 0 -> example 2, row 1 -> example 1.
        assert_eq!(obs_vals, vec![2.0, 2.1, 1.0, 1.1]);

        let action_vals: Vec<i64> = actions.into_data().to_vec().unwrap();
        // actions[2] == 0, actions[1] == 1.
        assert_eq!(action_vals, vec![0, 1]);
    }

    #[test]
    fn test_seeded_minibatch_shuffle_is_reproducible() {
        use rand::{SeedableRng, rngs::StdRng};

        let ds = sample_dataset();

        let mut rng_a = StdRng::seed_from_u64(7);
        let batches_a = generate_minibatch_indices_with_rng(ds.len(), 2, &mut rng_a);

        let mut rng_b = StdRng::seed_from_u64(7);
        let batches_b = generate_minibatch_indices_with_rng(ds.len(), 2, &mut rng_b);

        // Same seed -> identical minibatch partition.
        assert_eq!(batches_a, batches_b);

        // Every example index appears exactly once across the partition.
        let mut seen: Vec<usize> = batches_a.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }
}
