//! Snake environment implementation
//!
//! This module implements the SnakeEnv struct and the Environment trait
//! for both single-agent and multi-agent snake games.

use super::{snake::Snake, types::{Direction, Position, GameState, Cell}};
use crate::env::{Environment, SpaceInfo, SpaceType, StepResult, StepInfo};
use anyhow::Result;
use rand::Rng;

/// Multi-agent Snake environment
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
}

impl SnakeEnv {
    /// Create new snake environment with specified number of agents
    pub fn new_multi(width: i32, height: i32, num_agents: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Create snakes in different corners
        let mut snakes = Vec::new();
        let positions = [
            (width / 4, height / 4, Direction::Right),      // Top-left
            (3 * width / 4, height / 4, Direction::Left),   // Top-right
            (width / 4, 3 * height / 4, Direction::Right),  // Bottom-left
            (3 * width / 4, 3 * height / 4, Direction::Left), // Bottom-right
        ];

        for i in 0..num_agents.min(4) {
            let (x, y, dir) = positions[i];
            let start_pos = Position::new(x, y);
            snakes.push(Snake::new(i, start_pos, dir));
        }

        // Generate initial food
        let food_pos = Position::new(
            rng.gen_range(0..width),
            rng.gen_range(0..height),
        );

        Self {
            width,
            height,
            snakes,
            num_agents,
            food: food_pos,
            episode: 0,
            steps: 0,
            max_steps: 1000,
            done: false,
        }
    }

    /// Create new single-agent snake environment (for backward compatibility)
    pub fn new(width: i32, height: i32) -> Self {
        Self::new_multi(width, height, 1)
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();

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
            rng.gen_range(0..self.width),
            rng.gen_range(0..self.height),
        );

        self.episode += 1;
        self.steps = 0;
        self.done = false;
    }

    /// Execute multi-agent step with actions for all snakes
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

        // Apply actions and move all snakes
        for (i, &action) in actions.iter().enumerate() {
            if i < self.snakes.len() && self.snakes[i].is_alive() {
                let new_direction = Direction::from_action(action);
                self.snakes[i].change_direction(new_direction);
                self.snakes[i].move_forward();
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

            // Check wall collision
            if self.snakes[i].collides_with_wall(self.width, self.height) {
                self.snakes[i].alive = false;
                total_reward -= 1.0;
                continue;
            }

            // Check self collision
            if self.snakes[i].collides_with_self() {
                self.snakes[i].alive = false;
                total_reward -= 1.0;
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
                    total_reward -= 1.0;
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
                let mut rng = rand::thread_rng();
                loop {
                    let x = rng.gen_range(0..self.width);
                    let y = rng.gen_range(0..self.height);
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
                total_reward -= 0.01; // Small time penalty for each alive snake
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
            let idx = (pos.y as usize) * (self.width as usize) + (pos.x as usize);
            obs[idx] = 1.0;
        }

        // Channel 1: Own snake head
        let head_idx = (own_snake.head.y as usize) * (self.width as usize) + (own_snake.head.x as usize);
        obs[grid_size + head_idx] = 1.0;

        // Channel 2: Other snakes
        for (id, snake) in self.snakes.iter().enumerate() {
            if id != snake_id {
                for pos in &snake.body {
                    let idx = 2 * grid_size + (pos.y as usize) * (self.width as usize) + (pos.x as usize);
                    obs[idx] = 1.0;
                }
            }
        }

        // Channel 3: Food
        let food_idx = 3 * grid_size + (self.food.y as usize) * (self.width as usize) + (self.food.x as usize);
        obs[food_idx] = 1.0;

        // Channel 4: Walls (boundaries)
        // Top and bottom walls
        for x in 0..self.width as usize {
            obs[4 * grid_size + 0 * (self.width as usize) + x] = 1.0;  // Top
            obs[4 * grid_size + ((self.height as usize - 1) * (self.width as usize)) + x] = 1.0;  // Bottom
        }
        // Left and right walls
        for y in 0..self.height as usize {
            obs[4 * grid_size + y * (self.width as usize) + 0] = 1.0;  // Left
            obs[4 * grid_size + y * (self.width as usize) + (self.width as usize - 1)] = 1.0;  // Right
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
            dx, dy,
            direction_onehot[0], direction_onehot[1],
            direction_onehot[2], direction_onehot[3],
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
}
