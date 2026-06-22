//! Snake environment implementation
//!
//! This module implements the SnakeEnv struct and the Environment trait
//! for both single-agent and multi-agent snake games.

use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{
    snake::Snake,
    types::{Cell, Direction, GameState, Position},
};
use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Default seed used when constructing a [`SnakeEnv`] without an explicit seed.
const DEFAULT_SEED: u64 = 0;

/// Snapshot of a [`SnakeEnv`]'s simulation state.
///
/// Captures every visible field of the env (grid, snake positions, food
/// position, score, step count, done flag) **and** the env's RNG. Because the
/// RNG is part of the snapshot, `restore_state` followed by `step` replays the
/// exact same random stream — so a food respawn after restore spawns food at
/// the same location as the original trajectory, making round-trips
/// bit-identical.
#[derive(Debug, Clone)]
pub struct SnakeEnvState {
    /// Grid width
    pub width: i32,
    /// Grid height
    pub height: i32,
    /// All snakes
    pub snakes: Vec<super::snake::Snake>,
    /// Number of agents
    pub num_agents: usize,
    /// Food position
    pub food: Position,
    /// Episode counter
    pub episode: usize,
    /// Step counter
    pub steps: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Done flag
    pub done: bool,
    /// RNG state used for food respawn (captured so restores are deterministic)
    pub rng: StdRng,
}

/// Multi-agent Snake environment.
///
/// # Snapshot semantics
///
/// `clone_state` / `restore_state` capture every visible env field (grid,
/// snake bodies, food location, score, step count, done flag) **and** the
/// env's owned RNG. Because the RNG is part of the snapshot, restoring and
/// re-stepping replays the identical random stream: a food respawn after
/// `restore_state` lands in the same cell as the original trajectory, so
/// round-trips are fully reproducible (including trajectories that eat food).
#[derive(Debug, Clone)]
pub struct SnakeEnv {
    /// Grid width
    pub width: i32,
    /// Grid height
    pub height: i32,
    /// All snakes
    pub snakes: Vec<Snake>,
    /// Number of agents
    pub num_agents: usize,
    /// Food position
    pub food: Position,
    /// Episode counter
    pub episode: usize,
    /// Step counter
    pub steps: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Done flag
    pub done: bool,
    /// Owned RNG for food spawning. Captured by `clone_state` so that
    /// `restore_state` + `step` reproduces the exact same trajectory.
    pub rng: StdRng,
}

impl SnakeEnv {
    /// Create new snake environment with specified number of agents
    pub fn new_multi(width: i32, height: i32, num_agents: usize) -> Self {
        Self::new_multi_with_seed(width, height, num_agents, DEFAULT_SEED)
    }

    /// Create new snake environment with a specific RNG seed.
    ///
    /// The seed controls food placement. Two envs constructed with the same
    /// seed and stepped with the same actions produce identical trajectories.
    pub fn new_multi_with_seed(width: i32, height: i32, num_agents: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Create snakes in different corners
        let mut snakes = Vec::new();
        let positions = [
            (width / 4, height / 4, Direction::Right),     // Top-left
            (3 * width / 4, height / 4, Direction::Left),  // Top-right
            (width / 4, 3 * height / 4, Direction::Right), // Bottom-left
            (3 * width / 4, 3 * height / 4, Direction::Left), // Bottom-right
        ];

        for i in 0..num_agents.min(4) {
            let (x, y, dir) = positions[i];
            let start_pos = Position::new(x, y);
            snakes.push(Snake::new(i, start_pos, dir));
        }

        // Generate initial food
        let food_pos = Position::new(rng.random_range(0..width), rng.random_range(0..height));

        Self {
            width,
            height,
            snakes,
            num_agents,
            food: food_pos,
            episode: 0,
            steps: 0,
            max_steps: 400,
            done: false,
            rng,
        }
    }

    /// Create new single-agent snake environment (for backward compatibility)
    pub fn new(width: i32, height: i32) -> Self {
        Self::new_multi(width, height, 1)
    }

    /// Toroidal (wraparound) distance between two positions
    fn toroidal_distance(&self, a: &Position, b: &Position) -> f32 {
        let dx = (a.x - b.x).unsigned_abs() as f32;
        let dy = (a.y - b.y).unsigned_abs() as f32;
        let dx = dx.min(self.width as f32 - dx);
        let dy = dy.min(self.height as f32 - dy);
        (dx * dx + dy * dy).sqrt()
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) {
        // Reset all snakes
        self.snakes.clear();
        let positions = [
            (self.width / 4, self.height / 4, Direction::Right),
            (3 * self.width / 4, self.height / 4, Direction::Left),
            (self.width / 4, 3 * self.height / 4, Direction::Right),
            (3 * self.width / 4, 3 * self.height / 4, Direction::Left),
        ];

        for i in 0..self.num_agents.min(4) {
            let (x, y, dir) = positions[i];
            let start_pos = Position::new(x, y);
            self.snakes.push(Snake::new(i, start_pos, dir));
        }

        // Generate new food
        self.food = Position::new(
            self.rng.random_range(0..self.width),
            self.rng.random_range(0..self.height),
        );

        self.episode += 1;
        self.steps = 0;
        self.done = false;
    }

    /// Execute multi-agent step with actions for all snakes
    /// Returns a single StepResult with summed rewards (for backward
    /// compatibility) Use step_multi_agents for per-agent rewards
    pub fn step_multi(&mut self, actions: &[i64]) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.get_observation(),
                reward: 0.0,
                terminated: true,
                truncated: false,
                info: StepInfo::default(),
            };
        }

        // Store previous distances to food for reward shaping (toroidal)
        let mut prev_distances: Vec<f32> = Vec::new();
        for snake in &self.snakes {
            if snake.is_alive() {
                prev_distances.push(self.toroidal_distance(&snake.head, &self.food));
            } else {
                prev_distances.push(f32::MAX); // Dead snakes don't get distance rewards
            }
        }

        // Apply actions and move all snakes (with wraparound boundaries)
        for (i, &action) in actions.iter().enumerate() {
            if i < self.snakes.len() && self.snakes[i].is_alive() {
                let new_direction = Direction::from_action(action);
                self.snakes[i].change_direction(new_direction);
                self.snakes[i].move_forward_wrap(self.width, self.height);
            }
        }

        self.steps += 1;

        let mut total_reward = 0.0;
        let mut any_alive = false;

        // Check collisions for each snake
        for i in 0..self.snakes.len() {
            if !self.snakes[i].is_alive() {
                continue;
            }

            // Note: no wall collision — game uses toroidal wraparound boundaries

            // Check self collision
            if self.snakes[i].collides_with_self() {
                self.snakes[i].alive = false;
                total_reward -= 0.5; // Reduced death penalty
                continue;
            }

            // Check collision with other snakes
            for j in 0..self.snakes.len() {
                if i == j || !self.snakes[j].is_alive() {
                    continue;
                }
                // Check if snake i's head collides with snake j's body
                if self.snakes[j].get_all_positions().contains(&self.snakes[i].head) {
                    self.snakes[i].alive = false;
                    total_reward -= 0.5; // Reduced death penalty
                    break;
                }
            }

            if !self.snakes[i].is_alive() {
                continue;
            }

            // Check food collection
            if self.snakes[i].eats_food(&self.food) {
                total_reward += 1.0;
                self.snakes[i].grow();

                // Generate new food
                loop {
                    let x = self.rng.random_range(0..self.width);
                    let y = self.rng.random_range(0..self.height);
                    let new_food = Position::new(x, y);

                    // Make sure food doesn't spawn on any snake
                    let mut on_snake = false;
                    for snake in &self.snakes {
                        if snake.get_all_positions().contains(&new_food) {
                            on_snake = true;
                            break;
                        }
                    }

                    if !on_snake {
                        self.food = new_food;
                        break;
                    }
                }
            }

            if self.snakes[i].is_alive() {
                any_alive = true;
                // Survival reward per step (0.01 per snake per step)
                // This encourages staying alive and exploring
                total_reward += 0.01;

                // Additional reward for longer snakes
                if self.snakes[i].body.len() > 3 {
                    let length_bonus = 0.1 * ((self.snakes[i].body.len() - 3) as f32);
                    total_reward += length_bonus;
                }

                // Distance-based reward shaping: reward getting closer to food (toroidal)
                if i < prev_distances.len() {
                    let current_distance = self.toroidal_distance(&self.snakes[i].head, &self.food);
                    let distance_delta = prev_distances[i] - current_distance;
                    let distance_reward =
                        distance_delta / (self.width.max(self.height) as f32) * 2.0;
                    total_reward += distance_reward;
                }
            }
        }

        // Check if all snakes are dead
        let terminated = !any_alive;
        if terminated {
            self.done = true;
        }

        // Check step limit
        let truncated = self.steps >= self.max_steps;
        if truncated {
            self.done = true;
        }

        StepResult {
            observation: self.get_observation(),
            reward: total_reward,
            terminated,
            truncated,
            info: StepInfo::default(),
        }
    }

    /// Execute multi-agent step with per-agent rewards
    /// Returns individual rewards for each snake
    pub fn step_multi_agents(&mut self, actions: &[i64]) -> (Vec<f32>, bool, bool) {
        if self.done {
            return (vec![0.0; self.snakes.len()], true, false);
        }

        // Store previous distances to food for reward shaping (toroidal)
        let mut prev_distances: Vec<f32> = Vec::new();
        for snake in &self.snakes {
            if snake.is_alive() {
                prev_distances.push(self.toroidal_distance(&snake.head, &self.food));
            } else {
                prev_distances.push(f32::MAX); // Dead snakes don't get distance rewards
            }
        }

        // Apply actions and move all snakes (with wraparound boundaries)
        for (i, &action) in actions.iter().enumerate() {
            if i < self.snakes.len() && self.snakes[i].is_alive() {
                let new_direction = Direction::from_action(action);
                self.snakes[i].change_direction(new_direction);
                self.snakes[i].move_forward_wrap(self.width, self.height);
            }
        }

        self.steps += 1;

        let mut agent_rewards = vec![0.0; self.snakes.len()];
        let mut any_alive = false;

        // Check collisions for each snake
        for i in 0..self.snakes.len() {
            if !self.snakes[i].is_alive() {
                continue;
            }

            // Wall collision removed - using torus/wraparound boundaries
            // Snakes now wrap around the edges instead of dying

            // Check self collision
            if self.snakes[i].collides_with_self() {
                self.snakes[i].alive = false;
                agent_rewards[i] -= 0.1; // Minimal death penalty to encourage risk-taking
                continue;
            }

            // Check collision with other snakes
            for j in 0..self.snakes.len() {
                if i == j || !self.snakes[j].is_alive() {
                    continue;
                }
                // Check if snake i's head collides with snake j's body
                if self.snakes[j].get_all_positions().contains(&self.snakes[i].head) {
                    self.snakes[i].alive = false;
                    agent_rewards[i] -= 0.1; // Minimal death penalty to encourage risk-taking
                    break;
                }
            }

            if !self.snakes[i].is_alive() {
                continue;
            }

            // Check food collection
            if self.snakes[i].eats_food(&self.food) {
                agent_rewards[i] += 1.0;
                self.snakes[i].grow();

                // Generate new food
                loop {
                    let x = self.rng.random_range(0..self.width);
                    let y = self.rng.random_range(0..self.height);
                    let new_food = Position::new(x, y);

                    // Make sure food doesn't spawn on any snake
                    let mut on_snake = false;
                    for snake in &self.snakes {
                        if snake.get_all_positions().contains(&new_food) {
                            on_snake = true;
                            break;
                        }
                    }

                    if !on_snake {
                        self.food = new_food;
                        break;
                    }
                }
            }

            if self.snakes[i].is_alive() {
                any_alive = true;
                // Survival reward per step
                agent_rewards[i] += 0.01;

                // Strong reward for longer snakes (encourages aggressive eating)
                if self.snakes[i].body.len() > 3 {
                    let length_bonus = 1.0 * ((self.snakes[i].body.len() - 3) as f32);
                    agent_rewards[i] += length_bonus;
                }

                // Distance-based reward shaping: reward getting closer to food (toroidal)
                if i < prev_distances.len() {
                    let current_distance = self.toroidal_distance(&self.snakes[i].head, &self.food);
                    let distance_delta = prev_distances[i] - current_distance;
                    let distance_reward =
                        distance_delta / (self.width.max(self.height) as f32) * 2.0;
                    agent_rewards[i] += distance_reward;
                }
            }
        }

        // Check if all snakes are dead
        let terminated = !any_alive;
        if terminated {
            self.done = true;
        }

        // Check step limit
        let truncated = self.steps >= self.max_steps;
        if truncated {
            self.done = true;
        }

        (agent_rewards, terminated, truncated)
    }

    /// Execute single-agent step (for backward compatibility)
    pub fn step(&mut self, action: i64) -> StepResult {
        self.step_multi(&[action])
    }

    /// Get grid-based observation for a specific snake
    ///
    /// Returns a multi-channel grid representation:
    /// - Channel 0: Own snake body (1.0 where body is)
    /// - Channel 1: Own snake head (1.0 at head position)
    /// - Channel 2: Other snakes (1.0 where other snakes are)
    /// - Channel 3: Food (1.0 at food position)
    /// - Channel 4: Walls (1.0 at boundaries)
    ///
    /// Flattened as [C0_pixels..., C1_pixels..., C2_pixels..., ...]
    pub fn get_grid_observation(&self, snake_id: usize) -> Vec<f32> {
        if snake_id >= self.snakes.len() {
            // Return empty grid if invalid snake_id
            return vec![0.0; 5 * (self.width as usize) * (self.height as usize)];
        }

        let grid_size = (self.width as usize) * (self.height as usize);
        let mut obs = vec![0.0; 5 * grid_size];

        let own_snake = &self.snakes[snake_id];

        // Channel 0: Own snake body
        for pos in &own_snake.body {
            if pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height {
                let idx = (pos.y as usize) * (self.width as usize) + (pos.x as usize);
                obs[idx] = 1.0;
            }
        }

        // Channel 1: Own snake head
        if own_snake.head.x >= 0
            && own_snake.head.x < self.width
            && own_snake.head.y >= 0
            && own_snake.head.y < self.height
        {
            let head_idx =
                (own_snake.head.y as usize) * (self.width as usize) + (own_snake.head.x as usize);
            obs[grid_size + head_idx] = 1.0;
        }

        // Channel 2: Other snakes
        for (id, snake) in self.snakes.iter().enumerate() {
            if id != snake_id {
                for pos in &snake.body {
                    if pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height {
                        let idx = 2 * grid_size
                            + (pos.y as usize) * (self.width as usize)
                            + (pos.x as usize);
                        obs[idx] = 1.0;
                    }
                }
            }
        }

        // Channel 3: Food
        if self.food.x >= 0
            && self.food.x < self.width
            && self.food.y >= 0
            && self.food.y < self.height
        {
            let food_idx = 3 * grid_size
                + (self.food.y as usize) * (self.width as usize)
                + (self.food.x as usize);
            obs[food_idx] = 1.0;
        }

        // Channel 4: Walls (boundaries)
        // Top and bottom walls
        for x in 0..self.width as usize {
            obs[4 * grid_size + 0 * (self.width as usize) + x] = 1.0; // Top
            obs[4 * grid_size + ((self.height as usize - 1) * (self.width as usize)) + x] = 1.0; // Bottom
        }
        // Left and right walls
        for y in 0..self.height as usize {
            obs[4 * grid_size + y * (self.width as usize) + 0] = 1.0; // Left
            obs[4 * grid_size + y * (self.width as usize) + (self.width as usize - 1)] = 1.0; // Right
        }

        obs
    }

    /// Get current observation (for first snake, backward compatibility)
    pub fn get_observation(&self) -> Vec<f32> {
        if self.snakes.is_empty() {
            return vec![0.0; 6];
        }

        let snake = &self.snakes[0];
        let dx = (self.food.x - snake.head.x) as f32 / self.width as f32;
        let dy = (self.food.y - snake.head.y) as f32 / self.height as f32;

        let direction_onehot = match snake.direction {
            Direction::Up => [1.0, 0.0, 0.0, 0.0],
            Direction::Down => [0.0, 1.0, 0.0, 0.0],
            Direction::Left => [0.0, 0.0, 1.0, 0.0],
            Direction::Right => [0.0, 0.0, 0.0, 1.0],
        };

        vec![
            dx,
            dy,
            direction_onehot[0],
            direction_onehot[1],
            direction_onehot[2],
            direction_onehot[3],
        ]
    }

    /// Render current game state
    pub fn render(&self) -> GameState {
        let mut grid = vec![vec![Cell::Empty; self.width as usize]; self.height as usize];

        // Place food
        grid[self.food.y as usize][self.food.x as usize] = Cell::Food;

        // Place all snakes
        for snake in &self.snakes {
            for (i, &pos) in snake.body.iter().enumerate() {
                let cell = if i == 0 {
                    Cell::SnakeHead(snake.id)
                } else {
                    Cell::SnakeBody(snake.id)
                };
                grid[pos.y as usize][pos.x as usize] = cell;
            }
        }

        GameState {
            grid,
            scores: self.snakes.iter().map(|s| s.length as i32).collect(),
            active_agents: self.snakes.iter().map(|s| s.is_alive()).collect(),
            episode: self.episode,
            steps: self.steps,
        }
    }
}

impl Environment for SnakeEnv {
    type Action = i64;
    type State = SnakeEnvState;

    fn reset(&mut self) {
        self.reset();
    }

    fn get_observation(&self) -> Vec<f32> {
        self.get_observation()
    }

    fn step(&mut self, action: i64) -> StepResult {
        self.step(action)
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![6], // [food_dx, food_dy, dir_up, dir_down, dir_left, dir_right]
            space_type: SpaceType::Box,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![4], // 4 directions
            space_type: SpaceType::Discrete(4),
        }
    }

    fn render(&self) -> Vec<u8> {
        vec![] // Rendering handled by GameState
    }

    fn close(&mut self) {
        // Nothing to clean up
    }

    fn clone_state(&self) -> SnakeEnvState {
        SnakeEnvState {
            width: self.width,
            height: self.height,
            snakes: self.snakes.clone(),
            num_agents: self.num_agents,
            food: self.food,
            episode: self.episode,
            steps: self.steps,
            max_steps: self.max_steps,
            done: self.done,
            rng: self.rng.clone(),
        }
    }

    fn restore_state(&mut self, state: &SnakeEnvState) {
        self.width = state.width;
        self.height = state.height;
        self.snakes = state.snakes.clone();
        self.num_agents = state.num_agents;
        self.food = state.food;
        self.episode = state.episode;
        self.steps = state.steps;
        self.max_steps = state.max_steps;
        self.done = state.done;
        self.rng = state.rng.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_restore_round_trips() {
        let mut env = SnakeEnv::new(10, 10);
        env.reset();

        // Step a few times to a non-initial state.
        for i in 0..5 {
            env.step((i % 4) as i64);
        }
        let snap = env.clone_state();

        // Snapshot must capture every visible field.
        assert_eq!(env.steps, snap.steps);
        assert_eq!(env.food, snap.food);
        assert_eq!(env.snakes.len(), snap.snakes.len());

        // Take an experimental step.
        let r1 = env.step(0);
        let post_food_1 = env.food;
        let post_steps_1 = env.steps;

        // Restore and take the same step again.
        env.restore_state(&snap);
        assert_eq!(env.steps, snap.steps);
        assert_eq!(env.food, snap.food);

        let r2 = env.step(0);

        // Snake's `step` is deterministic except for food respawn, which only
        // fires when food is eaten. With identical action and snapshot the
        // observation/reward must match.
        assert_eq!(r1.observation, r2.observation);
        assert_eq!(r1.reward, r2.reward);
        assert_eq!(r1.terminated, r2.terminated);
        assert_eq!(r1.truncated, r2.truncated);
        assert_eq!(env.food, post_food_1);
        assert_eq!(env.steps, post_steps_1);
    }

    /// Forces the food-respawn branch (which draws from the env RNG) to run
    /// across a `clone_state` / `restore_state` round-trip and asserts the
    /// respawned food lands in the same cell both times. This is the path that
    /// previously diverged because food respawn used the thread-local
    /// `rand::rng()` instead of the captured snapshot RNG.
    #[test]
    fn clone_restore_round_trips_when_food_is_eaten() {
        let mut env = SnakeEnv::new(10, 10);
        env.reset();

        // Action 0 maps to Direction::Up. The snake starts facing Right (Up is
        // not a reversal), so after the step the head moves up one cell. Place
        // the food there so the next step eats it and triggers an RNG respawn.
        let head = env.snakes[0].head;
        let food_before = Position::new(head.x, head.y - 1);
        env.food = food_before;

        let snap = env.clone_state();

        let r1 = env.step(0);
        let food_after_1 = env.food;

        env.restore_state(&snap);
        let r2 = env.step(0);
        let food_after_2 = env.food;

        // The step must have eaten the food and respawned it elsewhere via the
        // env RNG (confirming the respawn branch actually fired).
        assert_ne!(food_after_1, food_before, "expected food to be eaten and respawned");

        // The respawn must be reproducible across the restore: same new food
        // cell, same observation, same reward. Before the fix, respawn used the
        // thread-local RNG and these diverged intermittently.
        assert_eq!(food_after_1, food_after_2, "respawned food must match across restore");
        assert_eq!(r1.observation, r2.observation);
        assert_eq!(r1.reward, r2.reward);
    }
}
