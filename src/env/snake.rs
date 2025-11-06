//! Multi-player Snake environment
//!
//! A competitive N-player snake game where agents compete for food.
//! - Grid-based world
//! - Each snake grows when eating food
//! - Game ends when all snakes die (collision with wall/self/others)
//! - Rewards: +1 for food, -1 for death, small time penalty to encourage efficiency

use super::{Environment, SpaceInfo, SpaceType, StepResult, StepInfo};
#[cfg(feature = "training")]
use crate::multi_agent::environment::{MultiAgentEnvironment, MultiAgentResult};
use anyhow::Result;
use rand::Rng;
use std::collections::VecDeque;

/// Direction a snake can move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn from_action(action: i64) -> Self {
        match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        }
    }

    fn to_delta(self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

/// Position on the grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn add(&self, dx: i32, dy: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

/// A single snake
#[derive(Debug, Clone)]
pub struct Snake {
    pub body: VecDeque<Position>,
    pub direction: Direction,
    pub alive: bool,
    pub score: i32,
}

impl Snake {
    fn new(start_pos: Position, direction: Direction) -> Self {
        let mut body = VecDeque::new();
        body.push_back(start_pos);
        Self {
            body,
            direction,
            alive: true,
            score: 0,
        }
    }

    fn head(&self) -> Position {
        *self.body.front().unwrap()
    }

    fn contains(&self, pos: &Position) -> bool {
        self.body.contains(pos)
    }

    fn grow(&mut self, new_head: Position) {
        self.body.push_front(new_head);
    }

    fn move_forward(&mut self, new_head: Position) {
        self.body.push_front(new_head);
        self.body.pop_back();
    }
}

/// Multi-player Snake environment
pub struct SnakeEnv {
    /// Grid dimensions
    pub width: i32,
    pub height: i32,

    /// Number of players
    num_agents: usize,

    /// All snakes
    pub snakes: Vec<Snake>,

    /// Food positions
    pub food: Vec<Position>,

    /// Max food on grid
    max_food: usize,

    /// Steps in current episode
    steps: usize,

    /// Max steps per episode
    max_steps: usize,

    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl SnakeEnv {
    /// Create a new Snake environment
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width
    /// * `height` - Grid height
    /// * `num_agents` - Number of snakes/players
    pub fn new(width: i32, height: i32, num_agents: usize) -> Self {
        let mut env = Self {
            width,
            height,
            num_agents,
            snakes: Vec::new(),
            food: Vec::new(),
            max_food: 3,
            steps: 0,
            max_steps: 500,
            rng: rand::thread_rng(),
        };
        env.reset().unwrap();
        env
    }

    /// Spawn food at a random empty position
    fn spawn_food(&mut self) {
        if self.food.len() >= self.max_food {
            return;
        }

        // Try to find empty position (max 100 attempts)
        for _ in 0..100 {
            let x = self.rng.gen_range(0..self.width);
            let y = self.rng.gen_range(0..self.height);
            let pos = Position::new(x, y);

            // Check if position is empty
            let occupied = self.food.contains(&pos)
                || self.snakes.iter().any(|s| s.contains(&pos));

            if !occupied {
                self.food.push(pos);
                return;
            }
        }
    }

    /// Check if position is valid (inside grid)
    fn is_valid_position(&self, pos: &Position) -> bool {
        pos.x >= 0 && pos.x < self.width && pos.y >= 0 && pos.y < self.height
    }

    /// Check if position collides with any snake
    fn collides_with_snake(&self, pos: &Position, exclude_idx: Option<usize>) -> bool {
        self.snakes.iter().enumerate().any(|(i, snake)| {
            if Some(i) == exclude_idx {
                false
            } else {
                snake.alive && snake.contains(pos)
            }
        })
    }

    /// Get observation for a specific agent
    fn get_observation_vec(&self, agent_id: usize) -> Vec<f32> {
        let snake = &self.snakes[agent_id];
        let head = snake.head();

        let mut obs = Vec::with_capacity(self.observation_dim());

        // Normalize positions to [0, 1]
        let norm_x = head.x as f32 / self.width as f32;
        let norm_y = head.y as f32 / self.height as f32;

        // Agent's head position (2)
        obs.push(norm_x);
        obs.push(norm_y);

        // Agent's direction as one-hot (4)
        obs.push(if snake.direction == Direction::Up { 1.0 } else { 0.0 });
        obs.push(if snake.direction == Direction::Down { 1.0 } else { 0.0 });
        obs.push(if snake.direction == Direction::Left { 1.0 } else { 0.0 });
        obs.push(if snake.direction == Direction::Right { 1.0 } else { 0.0 });

        // Snake length (1)
        obs.push((snake.body.len() as f32) / 10.0); // Normalize by max expected length

        // Nearest food position (2) - relative to head
        if let Some(nearest_food) = self.food.iter().min_by_key(|f| {
            (f.x - head.x).abs() + (f.y - head.y).abs()
        }) {
            obs.push((nearest_food.x - head.x) as f32 / self.width as f32);
            obs.push((nearest_food.y - head.y) as f32 / self.height as f32);
        } else {
            obs.push(0.0);
            obs.push(0.0);
        }

        // Danger in each direction (4) - 1 if dangerous, 0 if safe
        for dir in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
            let (dx, dy) = dir.to_delta();
            let next_pos = head.add(dx, dy);
            let danger = !self.is_valid_position(&next_pos)
                || self.collides_with_snake(&next_pos, Some(agent_id));
            obs.push(if danger { 1.0 } else { 0.0 });
        }

        // Number of alive opponents (1)
        let alive_opponents = self.snakes.iter()
            .enumerate()
            .filter(|(i, s)| *i != agent_id && s.alive)
            .count();
        obs.push(alive_opponents as f32 / self.num_agents as f32);

        obs
    }

    fn observation_dim(&self) -> usize {
        // 2 (head pos) + 4 (direction) + 1 (length) + 2 (food) + 4 (danger) + 1 (opponents) = 14
        14
    }

    /// Internal multi-agent step implementation
    /// This is used by both the single-agent Environment trait and the multi-agent trait
    fn step_multi_impl(&mut self, actions: &[i64]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<bool>, Vec<bool>) {
        self.steps += 1;

        let mut rewards = vec![0.0; self.num_agents];
        let mut terminated = vec![false; self.num_agents];

        // Update directions based on actions
        for (i, &action) in actions.iter().enumerate() {
            if self.snakes[i].alive {
                let new_direction = Direction::from_action(action);
                // Prevent 180-degree turns
                let opposite = match self.snakes[i].direction {
                    Direction::Up => Direction::Down,
                    Direction::Down => Direction::Up,
                    Direction::Left => Direction::Right,
                    Direction::Right => Direction::Left,
                };
                if new_direction != opposite {
                    self.snakes[i].direction = new_direction;
                }
            }
        }

        // Move all snakes
        let mut new_heads = Vec::new();
        for snake in &self.snakes {
            if snake.alive {
                let (dx, dy) = snake.direction.to_delta();
                new_heads.push(Some(snake.head().add(dx, dy)));
            } else {
                new_heads.push(None);
            }
        }

        // Check collisions and update snakes
        for i in 0..self.num_agents {
            if let Some(new_head) = new_heads[i] {
                let mut ate_food = false;

                // Check wall collision
                if !self.is_valid_position(&new_head) {
                    self.snakes[i].alive = false;
                    rewards[i] = -1.0;
                    terminated[i] = true;
                    continue;
                }

                // Check self collision
                if self.snakes[i].contains(&new_head) {
                    self.snakes[i].alive = false;
                    rewards[i] = -1.0;
                    terminated[i] = true;
                    continue;
                }

                // Check collision with other snakes
                if self.collides_with_snake(&new_head, Some(i)) {
                    self.snakes[i].alive = false;
                    rewards[i] = -1.0;
                    terminated[i] = true;
                    continue;
                }

                // Check food
                if let Some(food_idx) = self.food.iter().position(|f| *f == new_head) {
                    self.food.remove(food_idx);
                    self.snakes[i].grow(new_head);
                    self.snakes[i].score += 1;
                    rewards[i] = 1.0;
                    ate_food = true;
                    self.spawn_food();
                }

                if !ate_food {
                    self.snakes[i].move_forward(new_head);
                    // Small time penalty to encourage eating food quickly
                    rewards[i] = -0.01;
                }
            }
        }

        // Get observations for all agents
        let observations = (0..self.num_agents)
            .map(|i| self.get_observation_vec(i))
            .collect();

        // Check if episode is done
        let alive_count = self.snakes.iter().filter(|s| s.alive).count();
        let all_done = alive_count == 0 || self.steps >= self.max_steps;

        let truncated = vec![all_done && self.steps >= self.max_steps; self.num_agents];
        if all_done {
            for i in 0..self.num_agents {
                terminated[i] = terminated[i] || alive_count == 0;
            }
        }

        (observations, rewards, terminated, truncated)
    }
}

impl Environment for SnakeEnv {
    type Observation = Vec<f32>;
    type Action = i64;

    fn reset(&mut self) -> Result<Self::Observation> {
        self.snakes.clear();
        self.food.clear();
        self.steps = 0;

        // Spawn snakes at different positions
        for i in 0..self.num_agents {
            let x = (self.width / (self.num_agents as i32 + 1)) * (i as i32 + 1);
            let y = self.height / 2;
            let direction = if i % 2 == 0 {
                Direction::Right
            } else {
                Direction::Left
            };
            self.snakes.push(Snake::new(Position::new(x, y), direction));
        }

        // Spawn initial food
        for _ in 0..self.max_food {
            self.spawn_food();
        }

        Ok(self.get_observation_vec(0))
    }

    fn step(&mut self, action: Self::Action) -> Result<StepResult<Self::Observation>> {
        // Single-agent mode - just control the first snake
        let actions = vec![action];
        let (observations, rewards, terminated, truncated) = self.step_multi_impl(&actions);

        Ok(StepResult {
            observation: observations[0].clone(),
            reward: rewards[0],
            terminated: terminated[0],
            truncated: truncated[0],
            info: StepInfo::default(),
        })
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![self.observation_dim()],
            dtype: SpaceType::Continuous,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![],
            dtype: SpaceType::Discrete(4), // Up, Down, Left, Right
        }
    }
}

#[cfg(feature = "training")]
impl MultiAgentEnvironment for SnakeEnv {
    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn get_observation(&self, agent_id: usize) -> Self::Observation {
        self.get_observation_vec(agent_id)
    }

    fn step_multi(&mut self, actions: &[Self::Action]) -> MultiAgentResult<Self> {
        let (observations, rewards, terminated, truncated) = self.step_multi_impl(actions);
        MultiAgentResult::new(observations, rewards, terminated, truncated)
    }

    fn active_agents(&self) -> Vec<bool> {
        self.snakes.iter().map(|s| s.alive).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_creation() {
        let env = SnakeEnv::new(10, 10, 2);
        assert_eq!(env.num_agents(), 2);
        assert_eq!(env.snakes.len(), 2);
    }

    #[test]
    fn test_snake_reset() {
        let mut env = SnakeEnv::new(10, 10, 2);
        env.reset().unwrap();
        assert_eq!(env.snakes.len(), 2);
        assert!(env.snakes[0].alive);
        assert!(env.snakes[1].alive);
        assert!(env.food.len() > 0);
    }

    #[test]
    fn test_snake_observation() {
        let env = SnakeEnv::new(10, 10, 2);
        let obs = env.get_observation(0);
        assert_eq!(obs.len(), 14);
    }

    #[test]
    fn test_snake_step() {
        let mut env = SnakeEnv::new(10, 10, 2);
        env.reset().unwrap();
        let actions = vec![0, 1]; // Both move
        let result = env.step_multi(&actions);
        assert_eq!(result.observations.len(), 2);
        assert_eq!(result.rewards.len(), 2);
    }

    #[test]
    fn test_wall_collision() {
        let mut env = SnakeEnv::new(5, 5, 1);
        env.reset().unwrap();
        // Force snake to edge
        env.snakes[0].body.clear();
        env.snakes[0].body.push_back(Position::new(0, 0));
        env.snakes[0].direction = Direction::Left;

        let result = env.step_multi(&[2]); // Move left into wall
        assert!(!env.snakes[0].alive);
        assert_eq!(result.rewards[0], -1.0);
    }
}
