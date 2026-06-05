//! Sum-tree for prioritized experience replay.
//!
//! A sum-tree is a binary heap stored in a flat array where each internal
//! node holds the sum of its two children. The root holds the total
//! priority, which makes it cheap to sample a leaf proportional to its
//! priority via cumulative-sum traversal:
//!
//! 1. Draw `p ∈ [0, total]` uniformly.
//! 2. Walk from the root: if `p ≤ left`, descend left; else subtract `left`
//!    from `p` and descend right.
//! 3. The leaf reached is the chosen index.
//!
//! Both `update` and `find` are O(log N) where N is the number of leaves.
//!
//! # Layout
//!
//! For a tree with `capacity` leaves, the underlying `Vec<f32>` has length
//! `2 * capacity - 1`:
//!
//! - Indices `0 .. capacity - 1` are internal nodes (the root is at 0).
//! - Indices `capacity - 1 .. 2 * capacity - 1` are the leaves.
//!
//! For an internal node at index `i`, its children sit at `2*i + 1` (left)
//! and `2*i + 2` (right). For a leaf at index `i`, its parent is at
//! `(i - 1) / 2`.
//!
//! # References
//!
//! - Schaul et al., *Prioritized Experience Replay* ([ICLR 2016](https://arxiv.org/abs/1511.05952)).

/// Sum-tree over `capacity` leaves backed by a flat `Vec<f32>`.
///
/// Used by [`super::PrioritizedReplayBuffer`] to draw transitions with
/// probability proportional to their priority. The tree itself doesn't
/// know about transitions — the buffer maintains the leaf-to-transition
/// mapping separately.
#[derive(Debug, Clone)]
pub struct SumTree {
    /// Number of leaves the tree can hold.
    capacity: usize,
    /// Flat node storage of length `2 * capacity - 1`. The root is at
    /// index 0 and the leaves start at index `capacity - 1`.
    nodes: Vec<f32>,
}

impl SumTree {
    /// Create a new sum-tree with all priorities initialized to zero.
    ///
    /// # Arguments
    /// * `capacity` - Number of leaves. Must be > 0.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SumTree capacity must be > 0");
        Self { capacity, nodes: vec![0.0; 2 * capacity - 1] }
    }

    /// Total priority (the root of the tree).
    #[inline]
    pub fn total(&self) -> f32 {
        self.nodes[0]
    }

    /// Number of leaves.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Index of the first leaf in the underlying `nodes` array.
    #[inline]
    fn leaf_offset(&self) -> usize {
        self.capacity - 1
    }

    /// Maximum priority across all leaves.
    ///
    /// Used by [`super::PrioritizedReplayBuffer::push`] to assign a
    /// freshly inserted transition a priority that guarantees it gets
    /// sampled at least once.
    pub fn max_leaf(&self) -> f32 {
        let offset = self.leaf_offset();
        let mut m: f32 = 0.0;
        for &v in &self.nodes[offset..offset + self.capacity] {
            if v > m {
                m = v;
            }
        }
        m
    }

    /// Read the priority of leaf `leaf_idx ∈ [0, capacity)`.
    ///
    /// # Panics
    /// Panics if `leaf_idx >= capacity`.
    pub fn leaf_priority(&self, leaf_idx: usize) -> f32 {
        assert!(
            leaf_idx < self.capacity,
            "SumTree::leaf_priority: leaf_idx ({}) >= capacity ({})",
            leaf_idx,
            self.capacity
        );
        self.nodes[self.leaf_offset() + leaf_idx]
    }

    /// Set the priority of leaf `leaf_idx ∈ [0, capacity)` to `priority`,
    /// propagating the delta up to the root.
    ///
    /// # Panics
    /// Panics if `leaf_idx >= capacity` or `priority` is not finite or is
    /// negative.
    pub fn update(&mut self, leaf_idx: usize, priority: f32) {
        assert!(
            leaf_idx < self.capacity,
            "SumTree::update: leaf_idx ({}) >= capacity ({})",
            leaf_idx,
            self.capacity
        );
        assert!(
            priority.is_finite() && priority >= 0.0,
            "SumTree::update: priority must be finite and non-negative, got {}",
            priority
        );

        let node_idx = self.leaf_offset() + leaf_idx;
        let delta = priority - self.nodes[node_idx];
        self.nodes[node_idx] = priority;

        // Walk up to the root, adding `delta` along the way.
        let mut idx = node_idx;
        while idx > 0 {
            idx = (idx - 1) / 2;
            self.nodes[idx] += delta;
        }
    }

    /// Find the leaf whose cumulative-priority interval contains `p`.
    ///
    /// Returns `(leaf_idx, priority)`. The walk is deterministic: at each
    /// internal node we descend left if `p ≤ left_child`, else we subtract
    /// the left child's value from `p` and descend right.
    ///
    /// The caller is responsible for ensuring `p ∈ [0, total()]`; values
    /// slightly above `total()` are clamped by the walk so that the
    /// rightmost non-zero leaf is returned.
    ///
    /// # Panics
    /// Panics if `total() == 0` (cannot sample from a zero-priority tree).
    pub fn find(&self, mut p: f32) -> (usize, f32) {
        let total = self.total();
        assert!(total > 0.0, "SumTree::find: total priority is zero; nothing to sample from");

        // Clamp `p` into the valid range. Tiny over-runs from float
        // arithmetic on the boundary can otherwise direct the walk into
        // an empty (zero-priority) right subtree.
        if p < 0.0 {
            p = 0.0;
        }
        if p > total {
            p = total;
        }

        let mut idx = 0usize;
        let leaf_start = self.leaf_offset();
        while idx < leaf_start {
            let left = 2 * idx + 1;
            let right = left + 1;
            let left_val = self.nodes[left];
            if p <= left_val {
                idx = left;
            } else {
                p -= left_val;
                idx = right;
            }
        }
        let leaf_idx = idx - leaf_start;
        (leaf_idx, self.nodes[idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_zero_total() {
        let tree = SumTree::new(8);
        assert_eq!(tree.capacity(), 8);
        assert_eq!(tree.total(), 0.0);
        assert_eq!(tree.max_leaf(), 0.0);
    }

    #[test]
    fn test_update_propagates_to_root() {
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);
        assert!((tree.total() - 10.0).abs() < 1e-6);
        assert!((tree.max_leaf() - 4.0).abs() < 1e-6);
        assert!((tree.leaf_priority(0) - 1.0).abs() < 1e-6);
        assert!((tree.leaf_priority(3) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_in_place_changes_total() {
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);
        // Overwrite leaf 1 from 2.0 → 5.0. Total should rise by 3.
        tree.update(1, 5.0);
        assert!((tree.total() - 13.0).abs() < 1e-6);
        assert!((tree.leaf_priority(1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_walks_to_correct_leaf() {
        // Leaves: [1, 2, 3, 4]. Cumulative intervals (left-inclusive):
        //   leaf 0: (0, 1]
        //   leaf 1: (1, 3]
        //   leaf 2: (3, 6]
        //   leaf 3: (6, 10]
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert_eq!(tree.find(0.5).0, 0);
        assert_eq!(tree.find(1.0).0, 0);
        assert_eq!(tree.find(1.5).0, 1);
        assert_eq!(tree.find(3.0).0, 1);
        assert_eq!(tree.find(3.1).0, 2);
        assert_eq!(tree.find(6.0).0, 2);
        assert_eq!(tree.find(6.5).0, 3);
        assert_eq!(tree.find(10.0).0, 3);
    }

    #[test]
    fn test_find_skips_zero_priority_leaves() {
        // Leaves: [0, 5, 0, 3]. Anywhere in (0, 5] must land on leaf 1;
        // (5, 8] must land on leaf 3. Leaves 0 and 2 should never be hit.
        let mut tree = SumTree::new(4);
        tree.update(1, 5.0);
        tree.update(3, 3.0);

        let total = tree.total();
        assert!((total - 8.0).abs() < 1e-6);

        for p in [0.1, 1.0, 4.9, 5.0] {
            assert_eq!(tree.find(p).0, 1, "p={} should hit leaf 1", p);
        }
        for p in [5.1, 6.0, 8.0] {
            assert_eq!(tree.find(p).0, 3, "p={} should hit leaf 3", p);
        }
    }

    #[test]
    fn test_find_clamps_overshoot() {
        // If the caller passes p slightly above total, the walk should
        // still land on the rightmost non-zero leaf rather than wandering
        // into a zero-priority leaf at the end.
        let mut tree = SumTree::new(4);
        tree.update(0, 1.0);
        tree.update(1, 1.0);
        let (leaf, _) = tree.find(2.0 + 1e-3);
        assert!(leaf <= 1, "expected leaf ≤ 1 (rightmost non-zero), got {}", leaf);
    }

    #[test]
    fn test_max_leaf_tracks_updates() {
        let mut tree = SumTree::new(4);
        assert_eq!(tree.max_leaf(), 0.0);
        tree.update(0, 1.0);
        tree.update(1, 2.5);
        tree.update(2, 0.5);
        assert!((tree.max_leaf() - 2.5).abs() < 1e-6);
        tree.update(1, 0.1);
        assert!((tree.max_leaf() - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _ = SumTree::new(0);
    }

    #[test]
    #[should_panic(expected = "leaf_idx")]
    fn test_update_out_of_range_panics() {
        let mut tree = SumTree::new(4);
        tree.update(4, 1.0);
    }

    #[test]
    #[should_panic(expected = "priority must be finite")]
    fn test_update_nan_panics() {
        let mut tree = SumTree::new(4);
        tree.update(0, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "priority must be finite")]
    fn test_update_negative_panics() {
        let mut tree = SumTree::new(4);
        tree.update(0, -1.0);
    }

    #[test]
    #[should_panic(expected = "total priority is zero")]
    fn test_find_zero_total_panics() {
        let tree = SumTree::new(4);
        let _ = tree.find(0.5);
    }
}
