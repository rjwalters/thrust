//! GridWorld sparse-reward discrete navigation environment.
//!
//! [`GridWorld`] is a small, pure-Rust, zero-external-dependency
//! FrozenLake-style navigation task (issue #182, part of the
//! more-environments epic #180). It fills the gap in Thrust's
//! discrete-action catalog (`CartPole`, `Snake`, `Pong`, ...) for a
//! **sparse-reward navigation** problem: the agent must reach a goal on a
//! grid under a per-step penalty, with absorbing terminal states
//! (goal/hazard) that are distinct from step-cap truncation.
//!
//! The agent occupies one cell of a fixed `4x4` grid and moves between
//! cells. The default layout mirrors the classic `FrozenLake-v1` `4x4`
//! map (without slip): `S` start, `F` frozen/empty, `H` hole/hazard, `G`
//! goal.
//!
//! ```text
//! S F F F
//! F H F H
//! F F F H
//! H F F G
//! ```
//!
//! - **Action:** `i64`, four discrete moves: `0=Up`, `1=Right`, `2=Down`,
//!   `3=Left`. Out-of-range / invalid actions are treated as a no-op (the agent
//!   stays in place).
//! - **Observation:** one-hot over cells, a `Vec<f32>` of length `GRID_ROWS *
//!   GRID_COLS` (= 16). The entry at the agent's flattened index (`row *
//!   GRID_COLS + col`) is `1.0`; every other entry is `0.0`. This keeps the
//!   Q-network input fixed-width and discrete.
//! - **Reward:** reaching the goal yields [`GOAL_REWARD`] (`+1.0`); falling
//!   into a hole yields [`HOLE_REWARD`] (`-1.0`); any other move (including
//!   bumping into a wall) yields [`STEP_PENALTY`] (`-0.01`). The goal reward
//!   dominates so an optimal shortest path is the highest-return policy.
//! - **Termination:** the goal and hole cells are absorbing — `terminated =
//!   true`. Otherwise the episode runs until the step cap.
//! - **Truncation:** if not already terminated, `truncated = true` once `steps`
//!   reaches `max_steps` (default [`DEFAULT_MAX_STEPS`] = 100). Truncation is
//!   *not* termination.
//!
//! # Determinism contract
//!
//! The default layout has no stochastic dynamics: [`Environment::step`]
//! is fully deterministic given the current cell and action.
//! [`Environment::reset`] always returns the agent to the fixed start
//! cell. A seeded [`StdRng`] is held to match the surface of the other
//! envs (and is reserved for an optional later slip/random-layout mode),
//! but it is not consulted on the default layout. Consequently
//! [`Environment::restore_state`] followed by [`Environment::step`]
//! reproduces every subsequent [`StepResult`] bit-for-bit, and two envs
//! constructed with the same seed produce identical episodes. The
//! [`GridWorldState`] snapshot captures the agent position and step
//! counter (`agent_row`, `agent_col`, `steps`) but not the RNG.

use rand::{SeedableRng, rngs::StdRng};

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Number of rows in the grid.
pub const GRID_ROWS: usize = 4;

/// Number of columns in the grid.
pub const GRID_COLS: usize = 4;

/// Number of discrete actions (`Up`, `Right`, `Down`, `Left`).
pub const NUM_ACTIONS: usize = 4;

/// Per-step penalty for any non-terminal move (encourages short paths).
pub const STEP_PENALTY: f32 = -0.01;

/// Reward for reaching the goal cell.
pub const GOAL_REWARD: f32 = 1.0;

/// Reward for falling into a hole cell.
pub const HOLE_REWARD: f32 = -1.0;

/// Default episode length cap before truncation.
pub const DEFAULT_MAX_STEPS: usize = 100;

/// Default seed used by [`GridWorld::new`].
pub const DEFAULT_SEED: u64 = 0;

/// The kind of a grid cell in the default layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// Start cell (where the agent spawns on reset).
    Start,
    /// Empty / frozen cell the agent can stand on.
    Empty,
    /// Hole / hazard cell: absorbing, negative reward.
    Hole,
    /// Goal cell: absorbing, positive reward.
    Goal,
}

/// Default `4x4` FrozenLake-style layout (no slip).
///
/// ```text
/// S F F F
/// F H F H
/// F F F H
/// H F F G
/// ```
const DEFAULT_LAYOUT: [[Cell; GRID_COLS]; GRID_ROWS] = {
    use Cell::*;
    [
        [Start, Empty, Empty, Empty],
        [Empty, Hole, Empty, Hole],
        [Empty, Empty, Empty, Hole],
        [Hole, Empty, Empty, Goal],
    ]
};

/// Flattened index of the start cell in the default layout.
const START_ROW: usize = 0;
const START_COL: usize = 0;

/// Snapshot of [`GridWorld`]'s simulation state.
///
/// The default layout's dynamics are fully deterministic (no RNG), so
/// [`Environment::restore_state`] followed by [`Environment::step`]
/// reproduces every subsequent [`StepResult`] bit-for-bit. The snapshot
/// does *not* capture the RNG; restoring then calling
/// [`Environment::reset`] simply returns the agent to the start cell.
#[derive(Debug, Clone)]
pub struct GridWorldState {
    /// Agent row in `0..GRID_ROWS`.
    pub agent_row: usize,
    /// Agent column in `0..GRID_COLS`.
    pub agent_col: usize,
    /// Step counter for the current episode.
    pub steps: usize,
}

/// Sparse-reward discrete navigation task on a fixed `4x4` grid.
///
/// See the [module docs](self) for the full layout / action /
/// observation / reward / termination specification and the determinism
/// contract.
#[derive(Debug, Clone)]
pub struct GridWorld {
    /// Agent row in `0..GRID_ROWS`.
    agent_row: usize,

    /// Agent column in `0..GRID_COLS`.
    agent_col: usize,

    /// Step counter for the current episode.
    steps: usize,

    /// Maximum number of steps before truncation.
    max_steps: usize,

    /// Seeded RNG. Unused on the default layout; held to match the other
    /// envs' surface and reserved for a future slip/random-layout mode.
    #[allow(dead_code)]
    rng: StdRng,
}

impl GridWorld {
    /// Create a new env with the default seed, layout, and episode length.
    pub fn new() -> Self {
        Self::with_seed(DEFAULT_SEED)
    }

    /// Create a new env with a custom seed (default layout / episode
    /// length).
    ///
    /// On the default layout the seed has no observable effect; two envs
    /// constructed with any seeds produce identical episodes. The seed is
    /// reserved for a future slip/random-layout mode.
    pub fn with_seed(seed: u64) -> Self {
        Self::with_seed_and_max_steps(seed, DEFAULT_MAX_STEPS)
    }

    /// Create a new env with a custom seed and episode length.
    pub fn with_seed_and_max_steps(seed: u64, max_steps: usize) -> Self {
        let mut env = Self {
            agent_row: START_ROW,
            agent_col: START_COL,
            steps: 0,
            max_steps,
            rng: StdRng::seed_from_u64(seed),
        };
        env.reset();
        env
    }

    /// Current agent row (for inspection / tests).
    pub fn agent_row(&self) -> usize {
        self.agent_row
    }

    /// Current agent column (for inspection / tests).
    pub fn agent_col(&self) -> usize {
        self.agent_col
    }

    /// Current agent flattened cell index (`row * GRID_COLS + col`).
    pub fn agent_index(&self) -> usize {
        self.agent_row * GRID_COLS + self.agent_col
    }

    /// Look up the kind of the cell at `(row, col)`.
    fn cell_at(row: usize, col: usize) -> Cell {
        DEFAULT_LAYOUT[row][col]
    }

    /// Decode an action into a `(drow, dcol)` movement delta. Out-of-range
    /// or invalid actions decode to `(0, 0)` (a no-op / stay).
    fn action_delta(action: i64) -> (i64, i64) {
        match action {
            0 => (-1, 0), // Up
            1 => (0, 1),  // Right
            2 => (1, 0),  // Down
            3 => (0, -1), // Left
            _ => (0, 0),  // invalid / out-of-range -> stay
        }
    }
}

impl Default for GridWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for GridWorld {
    /// Discrete navigation action. One of `0=Up`, `1=Right`, `2=Down`,
    /// `3=Left`; any other value is a no-op (the agent stays put).
    type Action = i64;

    /// Snapshot type. The default layout is deterministic (no RNG in
    /// step), so restore + step reproduces subsequent results exactly.
    /// The RNG is not captured; see [`GridWorldState`].
    type State = GridWorldState;

    fn reset(&mut self) {
        self.agent_row = START_ROW;
        self.agent_col = START_COL;
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        // One-hot over cells: 1.0 at the agent's flattened index.
        let mut obs = vec![0.0; GRID_ROWS * GRID_COLS];
        obs[self.agent_index()] = 1.0;
        obs
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Decode the move and clamp into bounds: moving into a wall keeps
        // the agent in place.
        let (drow, dcol) = Self::action_delta(action);
        let new_row = self.agent_row as i64 + drow;
        let new_col = self.agent_col as i64 + dcol;
        if new_row >= 0 && new_row < GRID_ROWS as i64 && new_col >= 0 && new_col < GRID_COLS as i64
        {
            self.agent_row = new_row as usize;
            self.agent_col = new_col as usize;
        }

        self.steps += 1;

        let (reward, terminated) = match Self::cell_at(self.agent_row, self.agent_col) {
            Cell::Goal => (GOAL_REWARD, true),
            Cell::Hole => (HOLE_REWARD, true),
            Cell::Start | Cell::Empty => (STEP_PENALTY, false),
        };

        // Truncation only applies when the episode did not already end at
        // an absorbing cell.
        let truncated = !terminated && self.steps >= self.max_steps;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        // One-hot floats over the 16 cells. Box-shaped even though the
        // underlying state is discrete (mirrors CartPole reporting a Box
        // obs with Discrete actions).
        SpaceInfo { shape: vec![GRID_ROWS * GRID_COLS], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![NUM_ACTIONS], space_type: SpaceType::Discrete(NUM_ACTIONS) }
    }

    fn render(&self) -> Vec<u8> {
        Vec::new()
    }

    fn close(&mut self) {}

    fn clone_state(&self) -> GridWorldState {
        GridWorldState { agent_row: self.agent_row, agent_col: self.agent_col, steps: self.steps }
    }

    fn restore_state(&mut self, state: &GridWorldState) {
        self.agent_row = state.agent_row;
        self.agent_col = state.agent_col;
        self.steps = state.steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Action constants for readability.
    const UP: i64 = 0;
    const RIGHT: i64 = 1;
    const DOWN: i64 = 2;
    const LEFT: i64 = 3;

    #[test]
    fn observation_is_one_hot_of_length_sixteen() {
        let mut env = GridWorld::new();
        env.reset();

        let obs = env.get_observation();
        assert_eq!(obs.len(), GRID_ROWS * GRID_COLS, "observation must be 16-dimensional");

        // Exactly one entry is 1.0; the rest are 0.0.
        let ones = obs.iter().filter(|&&v| v == 1.0).count();
        let zeros = obs.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(ones, 1, "exactly one cell should be hot");
        assert_eq!(zeros, GRID_ROWS * GRID_COLS - 1, "every other cell should be zero");

        // The hot index is the agent's flattened index (start = 0).
        assert_eq!(obs[0], 1.0, "agent starts at cell index 0");
        assert_eq!(env.agent_index(), 0);
    }

    #[test]
    fn reset_places_agent_at_start() {
        let mut env = GridWorld::new();
        // Move away from start, then reset.
        env.step(RIGHT);
        env.step(DOWN);
        env.reset();
        assert_eq!(env.agent_row(), 0);
        assert_eq!(env.agent_col(), 0);
        assert_eq!(env.agent_index(), 0);
    }

    #[test]
    fn each_action_moves_in_the_right_direction() {
        // Start from an interior empty cell (2, 1) so all four moves are
        // valid and land on non-terminal cells.
        let interior = GridWorldState { agent_row: 2, agent_col: 1, steps: 0 };

        let mut env = GridWorld::new();

        env.restore_state(&interior);
        env.step(UP);
        assert_eq!((env.agent_row(), env.agent_col()), (1, 1), "Up decrements row");

        env.restore_state(&interior);
        env.step(DOWN);
        assert_eq!((env.agent_row(), env.agent_col()), (3, 1), "Down increments row");

        env.restore_state(&interior);
        env.step(LEFT);
        assert_eq!((env.agent_row(), env.agent_col()), (2, 0), "Left decrements col");

        env.restore_state(&interior);
        env.step(RIGHT);
        assert_eq!((env.agent_row(), env.agent_col()), (2, 2), "Right increments col");
    }

    #[test]
    fn walls_clamp_the_agent_in_place() {
        let mut env = GridWorld::new();

        // Top-left corner: Up and Left both bump walls and stay put.
        env.restore_state(&GridWorldState { agent_row: 0, agent_col: 0, steps: 0 });
        env.step(UP);
        assert_eq!((env.agent_row(), env.agent_col()), (0, 0), "Up at top edge stays put");
        env.step(LEFT);
        assert_eq!((env.agent_row(), env.agent_col()), (0, 0), "Left at left edge stays put");

        // Bottom-right corner of an empty cell: Down and Right bump walls.
        env.restore_state(&GridWorldState { agent_row: 0, agent_col: 3, steps: 0 });
        env.step(RIGHT);
        assert_eq!((env.agent_row(), env.agent_col()), (0, 3), "Right at right edge stays put");
        env.restore_state(&GridWorldState { agent_row: 3, agent_col: 1, steps: 0 });
        env.step(DOWN);
        assert_eq!((env.agent_row(), env.agent_col()), (3, 1), "Down at bottom edge stays put");
    }

    #[test]
    fn normal_move_yields_step_penalty_and_no_termination() {
        let mut env = GridWorld::new();
        env.reset();
        // Right from start lands on an empty cell (0, 1).
        let r = env.step(RIGHT);
        assert_eq!(r.reward, STEP_PENALTY, "normal move incurs the step penalty");
        assert!(!r.terminated, "empty cell is not terminal");
        assert!(!r.truncated, "no truncation early");
    }

    #[test]
    fn wall_bump_yields_step_penalty_not_termination() {
        let mut env = GridWorld::new();
        env.reset();
        // Up at the top edge: stay at start, still a normal (penalised) step.
        let r = env.step(UP);
        assert_eq!(r.reward, STEP_PENALTY);
        assert!(!r.terminated);
        assert_eq!((env.agent_row(), env.agent_col()), (0, 0));
    }

    #[test]
    fn reaching_goal_terminates_with_goal_reward() {
        let mut env = GridWorld::new();
        // Sit just left of the goal at (3, 2); move Right onto goal (3, 3).
        env.restore_state(&GridWorldState { agent_row: 3, agent_col: 2, steps: 0 });
        let r = env.step(RIGHT);
        assert_eq!((env.agent_row(), env.agent_col()), (3, 3));
        assert_eq!(r.reward, GOAL_REWARD, "goal yields the goal reward");
        assert!(r.terminated, "goal is absorbing / terminal");
        assert!(!r.truncated, "termination is not truncation");
    }

    #[test]
    fn falling_into_hole_terminates_with_hole_reward() {
        let mut env = GridWorld::new();
        // Sit at (0, 1); move Down into the hole at (1, 1).
        env.restore_state(&GridWorldState { agent_row: 0, agent_col: 1, steps: 0 });
        let r = env.step(DOWN);
        assert_eq!((env.agent_row(), env.agent_col()), (1, 1));
        assert_eq!(r.reward, HOLE_REWARD, "hole yields the hole reward");
        assert!(r.terminated, "hole is absorbing / terminal");
        assert!(!r.truncated, "termination is not truncation");
    }

    #[test]
    fn truncates_after_max_steps_without_termination() {
        // Small cap so we can hit it quickly. Bounce against the top wall
        // (Up at start) so we never reach goal or hole.
        let mut env = GridWorld::with_seed_and_max_steps(0, 5);
        env.reset();

        for i in 0..4 {
            let r = env.step(UP);
            assert!(!r.truncated, "should not truncate before max_steps (step {i})");
            assert!(!r.terminated, "wall bump never terminates");
        }

        let r = env.step(UP);
        assert!(r.truncated, "episode should truncate at max_steps");
        assert!(!r.terminated, "truncation is not termination");
    }

    #[test]
    fn truncation_does_not_fire_when_terminating_on_the_cap_step() {
        // max_steps = 1: a single step that lands on the goal must report
        // terminated (not truncated), proving termination takes priority.
        let mut env = GridWorld::with_seed_and_max_steps(0, 1);
        env.restore_state(&GridWorldState { agent_row: 3, agent_col: 2, steps: 0 });
        let r = env.step(RIGHT);
        assert!(r.terminated, "goal reached on the cap step terminates");
        assert!(!r.truncated, "truncation must not also fire when terminated");
    }

    #[test]
    fn out_of_range_and_invalid_actions_are_noops() {
        let mut env = GridWorld::new();
        env.restore_state(&GridWorldState { agent_row: 2, agent_col: 1, steps: 0 });

        for &bad in &[-1_i64, 4, 99, i64::MIN, i64::MAX] {
            env.restore_state(&GridWorldState { agent_row: 2, agent_col: 1, steps: 0 });
            let r = env.step(bad);
            assert_eq!(
                (env.agent_row(), env.agent_col()),
                (2, 1),
                "invalid action {bad} should be a no-op (stay)"
            );
            assert_eq!(r.reward, STEP_PENALTY, "a no-op still counts as a penalised step");
            assert!(!r.terminated);
        }
    }

    #[test]
    fn seeded_reset_is_reproducible() {
        let mut a = GridWorld::with_seed(42);
        let mut b = GridWorld::with_seed(7);
        a.reset();
        b.reset();
        // The default layout is deterministic, so any seeds agree.
        assert_eq!(a.get_observation(), b.get_observation(), "default layout is seed-independent");

        // Determinism holds across a full rollout.
        for action in [RIGHT, DOWN, RIGHT, DOWN, RIGHT] {
            let ra = a.step(action);
            let rb = b.step(action);
            assert_eq!(ra.observation, rb.observation);
            assert_eq!(ra.reward, rb.reward);
            assert_eq!(ra.terminated, rb.terminated);
            assert_eq!(ra.truncated, rb.truncated);
            if ra.terminated || ra.truncated {
                break;
            }
        }
    }

    #[test]
    fn clone_restore_round_trips_next_step() {
        let mut env = GridWorld::new();
        env.reset();
        // Advance a few steps so the snapshot is non-trivial.
        env.step(RIGHT);
        env.step(DOWN); // (1, 1) is a hole; restore to a safe interior instead.

        env.restore_state(&GridWorldState { agent_row: 2, agent_col: 1, steps: 3 });
        let snapshot = env.clone_state();
        let result_a = env.step(RIGHT);

        env.restore_state(&snapshot);
        let result_b = env.step(RIGHT);

        assert_eq!(result_a.observation, result_b.observation, "obs must reproduce bit-for-bit");
        assert_eq!(result_a.reward, result_b.reward, "reward must reproduce bit-for-bit");
        assert_eq!(result_a.terminated, result_b.terminated);
        assert_eq!(result_a.truncated, result_b.truncated);
    }

    #[test]
    fn action_space_is_discrete_four() {
        let env = GridWorld::new();
        let space = env.action_space();
        assert_eq!(space.shape, vec![NUM_ACTIONS]);
        assert!(matches!(space.space_type, SpaceType::Discrete(NUM_ACTIONS)));
    }

    #[test]
    fn observation_space_is_box_sixteen() {
        let env = GridWorld::new();
        let space = env.observation_space();
        assert_eq!(space.shape, vec![GRID_ROWS * GRID_COLS]);
        assert!(matches!(space.space_type, SpaceType::Box));
    }

    #[test]
    fn goal_reward_dominates_so_optimal_path_has_highest_return() {
        // The shortest start->goal path costs a handful of step penalties
        // and earns +1.0; its return must stay positive and exceed any
        // hole outcome.
        let max_path_cost = STEP_PENALTY * (DEFAULT_MAX_STEPS as f32);
        assert!(GOAL_REWARD + max_path_cost > HOLE_REWARD, "goal must dominate the hole");
        // The Manhattan distance from start to goal bounds the shortest
        // path; that path's return must stay positive so the goal path
        // beats any non-goal outcome.
        let mut env = GridWorld::new();
        env.restore_state(&GridWorldState { agent_row: 3, agent_col: 3, steps: 0 });
        let manhattan = (env.agent_row() - START_ROW) + (env.agent_col() - START_COL);
        let short_path_return = GOAL_REWARD + STEP_PENALTY * (manhattan as f32);
        assert!(short_path_return > 0.0, "a short goal path has positive return");
    }
}
