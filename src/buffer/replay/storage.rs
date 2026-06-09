//! Replay buffer storage for off-policy training
//!
//! This module implements a fixed-capacity ring buffer over flat `Vec`s
//! suitable for vanilla DQN-style replay. Transitions are stored as
//! plain CPU-side primitive vectors rather than Burn tensors so the
//! buffer stays WASM-compatible if it's ever exposed there, and so the
//! same code path can serve `--no-default-features` builds.
//!
//! # Layout
//!
//! For a buffer of capacity `C` and observation dimension `D`:
//! - `observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `actions`: `Vec<i64>` of length `C`
//! - `rewards`: `Vec<f32>` of length `C`
//! - `next_observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `dones`: `Vec<bool>` of length `C`
//!
//! Position `i` in the buffer is the slice
//! `observations[i*D .. (i+1)*D]` paired with `actions[i]`, `rewards[i]`,
//! `next_observations[i*D .. (i+1)*D]`, and `dones[i]`.
//!
//! # Ring semantics
//!
//! `push` writes at `write_idx`, increments `write_idx` modulo `capacity`,
//! and grows `len` up to (but not past) `capacity`. Once the buffer is
//! full, additional pushes overwrite the oldest transition (FIFO eviction).

/// Fixed-capacity FIFO replay buffer for off-policy training.
///
/// Stores `(obs, action, reward, next_obs, done)` transitions and supports
/// uniform random sampling via `sampling::sample`. The buffer is allocated
/// once at construction (`vec![0.0; capacity * obs_dim]` etc.); no further
/// allocations occur during `push`.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    obs_dim: usize,
    capacity: usize,
    len: usize,
    write_idx: usize,

    observations: Vec<f32>,
    actions: Vec<i64>,
    rewards: Vec<f32>,
    next_observations: Vec<f32>,
    dones: Vec<bool>,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity and observation
    /// dimensionality.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions stored. Must be > 0.
    /// * `obs_dim` - Length of one observation vector. Must be > 0.
    ///
    /// # Panics
    /// Panics if `capacity == 0` or `obs_dim == 0`.
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        assert!(capacity > 0, "ReplayBuffer capacity must be > 0");
        assert!(obs_dim > 0, "ReplayBuffer obs_dim must be > 0");

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
        }
    }

    /// Push a single transition into the buffer.
    ///
    /// When the buffer is at capacity, this overwrites the oldest
    /// transition (FIFO eviction).
    ///
    /// # Arguments
    /// * `obs` - Current observation; must have length `obs_dim`.
    /// * `action` - Discrete action taken.
    /// * `reward` - Reward received.
    /// * `next_obs` - Resulting observation; must have length `obs_dim`.
    /// * `done` - `true` if the episode terminated (used to mask the bootstrap
    ///   term in the TD target).
    ///
    /// # Panics
    /// Panics in debug builds if `obs.len() != obs_dim` or
    /// `next_obs.len() != obs_dim`.
    pub fn push(&mut self, obs: &[f32], action: i64, reward: f32, next_obs: &[f32], done: bool) {
        debug_assert_eq!(
            obs.len(),
            self.obs_dim,
            "ReplayBuffer::push: obs.len() ({}) != obs_dim ({})",
            obs.len(),
            self.obs_dim
        );
        debug_assert_eq!(
            next_obs.len(),
            self.obs_dim,
            "ReplayBuffer::push: next_obs.len() ({}) != obs_dim ({})",
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

    /// `true` if the buffer holds at least `min_size` transitions.
    ///
    /// Used by the trainer to decide whether to start sampling /
    /// performing gradient updates: vanilla DQN normally collects a
    /// few thousand transitions before the first update.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.len >= min_size
    }

    /// Read the transition at logical index `i` (0..len) into the provided
    /// scratch slices. Used by [`super::sampling`] to build batches.
    ///
    /// # Panics
    /// Panics if `i >= len`, or if `obs_out`/`next_obs_out` aren't sized
    /// `obs_dim`.
    pub(super) fn read_into(
        &self,
        i: usize,
        obs_out: &mut [f32],
        next_obs_out: &mut [f32],
    ) -> (i64, f32, bool) {
        assert!(i < self.len, "ReplayBuffer::read_into: i ({}) >= len ({})", i, self.len);
        assert_eq!(obs_out.len(), self.obs_dim);
        assert_eq!(next_obs_out.len(), self.obs_dim);

        let start = i * self.obs_dim;
        let end = start + self.obs_dim;
        obs_out.copy_from_slice(&self.observations[start..end]);
        next_obs_out.copy_from_slice(&self.next_observations[start..end]);
        (self.actions[i], self.rewards[i], self.dones[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut buf = ReplayBuffer::new(8, 3);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.obs_dim(), 3);

        buf.push(&[1.0, 2.0, 3.0], 1, 0.5, &[4.0, 5.0, 6.0], false);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        for i in 0..5 {
            buf.push(&[i as f32; 3], i as i64, i as f32, &[(i + 1) as f32; 3], false);
        }
        assert_eq!(buf.len(), 6);
    }

    #[test]
    fn test_roundtrip_values() {
        // Push known values, then read them back through read_into.
        let mut buf = ReplayBuffer::new(4, 2);
        buf.push(&[1.0, 2.0], 0, 10.0, &[3.0, 4.0], false);
        buf.push(&[5.0, 6.0], 1, 20.0, &[7.0, 8.0], true);

        let mut obs = [0.0f32; 2];
        let mut next_obs = [0.0f32; 2];
        let (a, r, d) = buf.read_into(0, &mut obs, &mut next_obs);
        assert_eq!(obs, [1.0, 2.0]);
        assert_eq!(next_obs, [3.0, 4.0]);
        assert_eq!(a, 0);
        assert_eq!(r, 10.0);
        assert!(!d);

        let (a, r, d) = buf.read_into(1, &mut obs, &mut next_obs);
        assert_eq!(obs, [5.0, 6.0]);
        assert_eq!(next_obs, [7.0, 8.0]);
        assert_eq!(a, 1);
        assert_eq!(r, 20.0);
        assert!(d);
    }

    #[test]
    fn test_ring_wraparound_evicts_oldest() {
        // Capacity 4: push 6 → first 2 should be gone, slots 0..4 should
        // hold transitions 2,3,4,5 (in write order: 4,5,2,3 since write_idx
        // wrapped).
        let mut buf = ReplayBuffer::new(4, 1);
        for i in 0..6 {
            buf.push(&[i as f32], i as i64, i as f32, &[(i + 100) as f32], false);
        }
        assert_eq!(buf.len(), 4, "len must saturate at capacity");

        // Collect all transitions in slot order.
        let mut found_actions = std::collections::HashSet::new();
        let mut obs_scratch = [0.0f32; 1];
        let mut next_scratch = [0.0f32; 1];
        for i in 0..buf.len() {
            let (a, _, _) = buf.read_into(i, &mut obs_scratch, &mut next_scratch);
            found_actions.insert(a);
        }

        // After 6 pushes into capacity 4, transitions 0 and 1 must be gone
        // and transitions 2,3,4,5 must remain.
        assert!(!found_actions.contains(&0), "oldest transition 0 not evicted");
        assert!(!found_actions.contains(&1), "oldest transition 1 not evicted");
        assert!(found_actions.contains(&2));
        assert!(found_actions.contains(&3));
        assert!(found_actions.contains(&4));
        assert!(found_actions.contains(&5));
    }

    #[test]
    fn test_is_ready() {
        let mut buf = ReplayBuffer::new(8, 2);
        assert!(!buf.is_ready(1));
        for _ in 0..4 {
            buf.push(&[0.0, 0.0], 0, 0.0, &[0.0, 0.0], false);
        }
        assert!(buf.is_ready(4));
        assert!(!buf.is_ready(5));
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _ = ReplayBuffer::new(0, 4);
    }

    #[test]
    #[should_panic(expected = "obs_dim must be > 0")]
    fn test_zero_obs_dim_panics() {
        let _ = ReplayBuffer::new(4, 0);
    }
}
