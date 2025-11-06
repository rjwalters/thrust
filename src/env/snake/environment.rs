//! Snake environment implementation
//!
//! This module implements the SnakeEnv struct and the Environment trait
//! for single-agent snake games.

use super::{snake::Snake, types::{Direction, Position, GameState, Cell}};
use crate::env::{Environment, SpaceInfo, SpaceType, StepResult, StepInfo};
use anyhow::Result;
use rand::Rng;

/// Single-agent Snake environment
#[derive(Debug, Clone)]
pub struct SnakeEnv {
    /// Grid width
    pub width: i32,
    /// Grid height
    pub height: i32,
    /// Current snake
    pub snake: Snake,
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
    /// Create new snake environment
    pub fn new(width: i32, height: i32) -> Self {
        let mut rng = rand::thread_rng();

        // Start snake in center
        let start_pos = Position::new(width / 2, height / 2);
        let snake = Snake::new(0, start_pos, Direction::Right);

        // Generate initial food
        let food_pos = Position::new(
            rng.gen_range(0..width),
            rng.gen_range(0..height),
        );

        Self {
            width,
            height,
            snake,
            food: food_pos,
            episode: 0,
            steps: 0,
            max_steps: 1000,
            done: false,
        }
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();

        // Reset snake
        let start_pos = Position::new(self.width / 2, self.height / 2);
        self.snake = Snake::new(0, start_pos, Direction::Right);

        // Generate new food
        self.food = Position::new(
            rng.gen_range(0..self.width),
            rng.gen_range(0..self.height),
        );

        self.episode += 1;
        self.steps = 0;
        self.done = false;
    }

    /// Execute action and return step result
    pub fn step(&mut self, action: i64) -> StepResult {
        if self.done {
            return StepResult {
                observation: self.get_observation(),
                reward: 0.0,
                terminated: true,
                truncated: false,
                info: StepInfo {
                    episode: self.episode,
                    steps: self.steps,
                },
            };
        }

        // Change direction
        let new_direction = Direction::from_action(action);
        self.snake.change_direction(new_direction);

        // Move snake
        self.snake.move_forward();
        self.steps += 1;

        // Check collisions
        let wall_collision = self.snake.collides_with_wall(self.width, self.height);
        let self_collision = self.snake.collides_with_self();

        let mut reward = -0.01; // Small time penalty
        let mut terminated = false;

        if wall_collision || self_collision {
            // Death penalty
            reward = -1.0;
            terminated = true;
            self.done = true;
        } else if self.snake.eats_food(&self.food) {
            // Food reward
            reward = 1.0;
            self.snake.grow();

            // Generate new food
            let mut rng = rand::thread_rng();
            loop {
                let x = rng.gen_range(0..self.width);
                let y = rng.gen_range(0..self.height);
                let new_food = Position::new(x, y);

                if !self.snake.get_all_positions().contains(&new_food) {
                    self.food = new_food;
                    break;
                }
            }
        }

        // Check step limit
        let truncated = self.steps >= self.max_steps;
        if truncated {
            self.done = true;
        }

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: StepInfo {
                episode: self.episode,
                steps: self.steps,
            },
        }
    }

    /// Get current observation
    pub fn get_observation(&self) -> Vec<f32> {
        // Simple observation: relative food position
        let dx = (self.food.x - self.snake.head.x) as f32 / self.width as f32;
        let dy = (self.food.y - self.snake.head.y) as f32 / self.height as f32;

        // Snake direction (one-hot encoded)
        let direction_onehot = match self.snake.direction {
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

        // Place snake
        for (i, &pos) in self.snake.body.iter().enumerate() {
            let cell = if i == 0 {
                Cell::SnakeHead(self.snake.id)
            } else {
                Cell::SnakeBody(self.snake.id)
            };
            grid[pos.y as usize][pos.x as usize] = cell;
        }

        GameState {
            grid,
            scores: vec![self.snake.length as i32],
            active_agents: vec![self.snake.is_alive()],
            episode: self.episode,
            steps: self.steps,
        }
    }
}

impl Environment for SnakeEnv {
    fn reset(&mut self) {
        self.reset();
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
