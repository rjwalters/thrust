//! Continuous-action replay buffer storage for off-policy training.
//!
//! This module mirrors [`super::storage::ReplayBuffer`] but stores
//! **continuous** `Vec<f32>` actions of width `action_dim` (e.g. for SAC /
//! TD3 / DDPG) instead of a single discrete `i64` action. Like the discrete
//! buffer, transitions are stored as plain CPU-side primitive vectors rather
//! than Burn tensors so the buffer stays WASM-compatible if it's ever exposed
//! there, and so the same code path can serve `--no-default-features` builds.
//!
//! # Layout
//!
//! For a buffer of capacity `C`, observation dimension `D`, and action
//! dimension `A`:
//! - `observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `actions`: `Vec<f32>` of length `C * A` (flattened)
//! - `rewards`: `Vec<f32>` of length `C`
//! - `next_observations`: `Vec<f32>` of length `C * D` (flattened)
//! - `dones`: `Vec<bool>` of length `C`
//!
//! Position `i` in the buffer is the slice
//! `observations[i*D .. (i+1)*D]` paired with `actions[i*A .. (i+1)*A]`,
//! `rewards[i]`, `next_observations[i*D .. (i+1)*D]`, and `dones[i]`.
//!
//! # Ring semantics
//!
//! `push` writes at `write_idx`, increments `write_idx` modulo `capacity`,
//! and grows `len` up to (but not past) `capacity`. Once the buffer is
//! full, additional pushes overwrite the oldest transition (FIFO eviction).

/// Fixed-capacity FIFO replay buffer for off-policy continuous-action
/// training.
///
/// Stores `(obs, action, reward, next_obs, done)` transitions with
/// continuous `Vec<f32>` actions of width `action_dim`, and supports uniform
/// random sampling via [`super::continuous_sampling::sample`]. The buffer is
/// allocated once at construction (`vec![0.0; capacity * obs_dim]` etc.); no
/// further allocations occur during `push`.
///
/// This is the continuous-action sibling of
/// [`ReplayBuffer`](super::storage::ReplayBuffer); the discrete DQN buffer is
/// left untouched.
#[derive(Debug, Clone)]
pub struct ContinuousReplayBuffer {
    obs_dim: usize,
    action_dim: usize,
    capacity: usize,
    len: usize,
    write_idx: usize,

    observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    next_observations: Vec<f32>,
    dones: Vec<bool>,
}

impl ContinuousReplayBuffer {
    /// Create a new continuous replay buffer with the given capacity,
    /// observation dimensionality, and action dimensionality.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions stored. Must be > 0.
    /// * `obs_dim` - Length of one observation vector. Must be > 0.
    /// * `action_dim` - Length of one action vector. Must be > 0.
    ///
    /// # Panics
    /// Panics if `capacity == 0`, `obs_dim == 0`, or `action_dim == 0`.
    pub fn new(capacity: usize, obs_dim: usize, action_dim: usize) -> Self {
        assert!(capacity > 0, "ContinuousReplayBuffer capacity must be > 0");
        assert!(obs_dim > 0, "ContinuousReplayBuffer obs_dim must be > 0");
        assert!(action_dim > 0, "ContinuousReplayBuffer action_dim must be > 0");

        Self {
            obs_dim,
            action_dim,
            capacity,
            len: 0,
            write_idx: 0,
            observations: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * action_dim],
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
    /// * `action` - Continuous action taken; must have length `action_dim`.
    /// * `reward` - Reward received.
    /// * `next_obs` - Resulting observation; must have length `obs_dim`.
    /// * `done` - `true` if the episode terminated (used to mask the bootstrap
    ///   term in the TD target).
    ///
    /// # Panics
    /// Panics in debug builds if `obs.len() != obs_dim`,
    /// `next_obs.len() != obs_dim`, or `action.len() != action_dim`.
    pub fn push(&mut self, obs: &[f32], action: &[f32], reward: f32, next_obs: &[f32], done: bool) {
        debug_assert_eq!(
            obs.len(),
            self.obs_dim,
            "ContinuousReplayBuffer::push: obs.len() ({}) != obs_dim ({})",
            obs.len(),
            self.obs_dim
        );
        debug_assert_eq!(
            next_obs.len(),
            self.obs_dim,
            "ContinuousReplayBuffer::push: next_obs.len() ({}) != obs_dim ({})",
            next_obs.len(),
            self.obs_dim
        );
        debug_assert_eq!(
            action.len(),
            self.action_dim,
            "ContinuousReplayBuffer::push: action.len() ({}) != action_dim ({})",
            action.len(),
            self.action_dim
        );

        let obs_start = self.write_idx * self.obs_dim;
        let obs_end = obs_start + self.obs_dim;
        self.observations[obs_start..obs_end].copy_from_slice(obs);
        self.next_observations[obs_start..obs_end].copy_from_slice(next_obs);

        let act_start = self.write_idx * self.action_dim;
        let act_end = act_start + self.action_dim;
        self.actions[act_start..act_end].copy_from_slice(action);

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

    /// Action dimension (length of one action slice).
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// `true` if the buffer holds at least `min_size` transitions.
    ///
    /// Used by the trainer to decide whether to start sampling /
    /// performing gradient updates: SAC normally collects a few thousand
    /// transitions before the first update.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.len >= min_size
    }

    /// Read the transition at logical index `i` (0..len) into the provided
    /// scratch slices. Used by [`super::continuous_sampling`] to build
    /// batches.
    ///
    /// # Panics
    /// Panics if `i >= len`, if `obs_out`/`next_obs_out` aren't sized
    /// `obs_dim`, or if `action_out` isn't sized `action_dim`.
    pub(super) fn read_into(
        &self,
        i: usize,
        obs_out: &mut [f32],
        action_out: &mut [f32],
        next_obs_out: &mut [f32],
    ) -> (f32, bool) {
        assert!(
            i < self.len,
            "ContinuousReplayBuffer::read_into: i ({}) >= len ({})",
            i,
            self.len
        );
        assert_eq!(obs_out.len(), self.obs_dim);
        assert_eq!(next_obs_out.len(), self.obs_dim);
        assert_eq!(action_out.len(), self.action_dim);

        let obs_start = i * self.obs_dim;
        let obs_end = obs_start + self.obs_dim;
        obs_out.copy_from_slice(&self.observations[obs_start..obs_end]);
        next_obs_out.copy_from_slice(&self.next_observations[obs_start..obs_end]);

        let act_start = i * self.action_dim;
        let act_end = act_start + self.action_dim;
        action_out.copy_from_slice(&self.actions[act_start..act_end]);

        (self.rewards[i], self.dones[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut buf = ContinuousReplayBuffer::new(8, 3, 2);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.obs_dim(), 3);
        assert_eq!(buf.action_dim(), 2);

        buf.push(&[1.0, 2.0, 3.0], &[0.1, 0.2], 0.5, &[4.0, 5.0, 6.0], false);
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        for i in 0..5 {
            buf.push(
                &[i as f32; 3],
                &[i as f32, i as f32 + 0.5],
                i as f32,
                &[(i + 1) as f32; 3],
                false,
            );
        }
        assert_eq!(buf.len(), 6);
    }

    #[test]
    fn test_roundtrip_values() {
        // Push known values, then read them back through read_into.
        let mut buf = ContinuousReplayBuffer::new(4, 2, 3);
        buf.push(&[1.0, 2.0], &[0.1, 0.2, 0.3], 10.0, &[3.0, 4.0], false);
        buf.push(&[5.0, 6.0], &[0.4, 0.5, 0.6], 20.0, &[7.0, 8.0], true);

        let mut obs = [0.0f32; 2];
        let mut action = [0.0f32; 3];
        let mut next_obs = [0.0f32; 2];
        let (r, d) = buf.read_into(0, &mut obs, &mut action, &mut next_obs);
        assert_eq!(obs, [1.0, 2.0]);
        assert_eq!(action, [0.1, 0.2, 0.3]);
        assert_eq!(next_obs, [3.0, 4.0]);
        assert_eq!(r, 10.0);
        assert!(!d);

        let (r, d) = buf.read_into(1, &mut obs, &mut action, &mut next_obs);
        assert_eq!(obs, [5.0, 6.0]);
        assert_eq!(action, [0.4, 0.5, 0.6]);
        assert_eq!(next_obs, [7.0, 8.0]);
        assert_eq!(r, 20.0);
        assert!(d);
    }

    #[test]
    fn test_ring_wraparound_evicts_oldest() {
        // Capacity 4: push 6 → first 2 should be gone, slots should hold
        // transitions 2,3,4,5. We tag each transition by its reward value.
        let mut buf = ContinuousReplayBuffer::new(4, 1, 2);
        for i in 0..6 {
            buf.push(&[i as f32], &[i as f32, -(i as f32)], i as f32, &[(i + 100) as f32], false);
        }
        assert_eq!(buf.len(), 4, "len must saturate at capacity");

        // Collect all transitions in slot order, keyed by reward (== i).
        let mut found_rewards = std::collections::HashSet::new();
        let mut obs_scratch = [0.0f32; 1];
        let mut action_scratch = [0.0f32; 2];
        let mut next_scratch = [0.0f32; 1];
        for i in 0..buf.len() {
            let (r, _) = buf.read_into(i, &mut obs_scratch, &mut action_scratch, &mut next_scratch);
            // Action width carries the same tag, sanity-check it survived.
            assert_eq!(action_scratch, [r, -r]);
            found_rewards.insert(r as i64);
        }

        // After 6 pushes into capacity 4, transitions 0 and 1 must be gone
        // and transitions 2,3,4,5 must remain.
        assert!(!found_rewards.contains(&0), "oldest transition 0 not evicted");
        assert!(!found_rewards.contains(&1), "oldest transition 1 not evicted");
        assert!(found_rewards.contains(&2));
        assert!(found_rewards.contains(&3));
        assert!(found_rewards.contains(&4));
        assert!(found_rewards.contains(&5));
    }

    #[test]
    fn test_is_ready() {
        let mut buf = ContinuousReplayBuffer::new(8, 2, 1);
        assert!(!buf.is_ready(1));
        for _ in 0..4 {
            buf.push(&[0.0, 0.0], &[0.0], 0.0, &[0.0, 0.0], false);
        }
        assert!(buf.is_ready(4));
        assert!(!buf.is_ready(5));
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _ = ContinuousReplayBuffer::new(0, 4, 2);
    }

    #[test]
    #[should_panic(expected = "obs_dim must be > 0")]
    fn test_zero_obs_dim_panics() {
        let _ = ContinuousReplayBuffer::new(4, 0, 2);
    }

    #[test]
    #[should_panic(expected = "action_dim must be > 0")]
    fn test_zero_action_dim_panics() {
        let _ = ContinuousReplayBuffer::new(4, 2, 0);
    }
}
