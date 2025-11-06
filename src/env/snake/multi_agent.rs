//! Multi-agent Snake environment implementation
//!
//! This module implements the MultiAgentEnvironment trait for
//! competitive multi-player snake games.

use super::{snake::{Snake, Food}, types::{Direction, Position, GameState, Cell}};
use crate::multi_agent::environment::{MultiAgentEnvironment, MultiAgentResult};
use rand::Rng;

/// Multi-agent Snake environment
#[derive(Debug, Clone)]
pub struct MultiAgentSnakeEnv {
    /// Grid width
    pub width: i32,
    /// Grid height
    pub height: i32,
    /// All snakes in the game
    pub snakes: Vec<Snake>,
    /// Food position
    pub food: Food,
    /// Which snakes are still active
    pub active_agents: Vec<bool>,
    /// Episode counter
    pub episode: usize,
    /// Step counter
    pub steps: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Done flag
    pub done: bool,
}

impl MultiAgentSnakeEnv {
    /// Create new multi-agent snake environment
    pub fn new(width: i32, height: i32, num_agents: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut snakes = Vec::new();

        // Create snakes at different starting positions
        for i in 0..num_agents {
            let start_x = (width / (num_agents + 1) * (i + 1) as i32).min(width - 1);
            let start_y = height / 2;
            let start_pos = Position::new(start_x, start_y);

            // Alternate starting directions
            let direction = if i % 2 == 0 { Direction::Up } else { Direction::Down };

            snakes.push(Snake::new(i, start_pos, direction));
        }

        // Generate initial food
        let food = Food::generate_random(width, height, &snakes, &mut rng);

        Self {
            width,
            height,
            snakes,
            food,
            active_agents: vec![true; num_agents],
            episode: 0,
            steps: 0,
            max_steps: 1000,
            done: false,
        }
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();

        // Reset all snakes
        for (i, snake) in self.snakes.iter_mut().enumerate() {
            let start_x = (self.width / (self.snakes.len() + 1) * (i + 1) as i32).min(self.width - 1);
            let start_y = self.height / 2;
            let start_pos = Position::new(start_x, start_y);

            // Alternate starting directions
            let direction = if i % 2 == 0 { Direction::Up } else { Direction::Down };

            *snake = Snake::new(i, start_pos, direction);
            self.active_agents[i] = true;
        }

        // Generate new food
        self.food = Food::generate_random(self.width, self.height, &self.snakes, &mut rng);

        self.episode += 1;
        self.steps = 0;
        self.done = false;
    }

    /// Execute actions for all agents
    pub fn step(&mut self, actions: &[i64]) -> MultiAgentResult {
        if self.done {
            return MultiAgentResult {
                observations: self.get_observations(),
                rewards: vec![0.0; self.snakes.len()],
                terminated: vec![true; self.snakes.len()],
                truncated: vec![false; self.snakes.len()],
                active_agents: self.active_agents.clone(),
            };
        }

        let mut rewards = vec![-0.01; self.snakes.len()]; // Small time penalty
        let mut terminated = vec![false; self.snakes.len()];

        // Change directions first (before movement)
        for (i, &action) in actions.iter().enumerate() {
            if self.active_agents[i] {
                let new_direction = Direction::from_action(action);
                self.snakes[i].change_direction(new_direction);
            }
        }

        // Move all snakes
        for snake in &mut self.snakes {
            if self.active_agents[snake.id] {
                snake.move_forward();
            }
        }

        self.steps += 1;

        // Check food consumption
        let mut food_eaten = false;
        for (i, snake) in self.snakes.iter().enumerate() {
            if self.active_agents[i] && snake.eats_food(&self.food.position) {
                rewards[i] = 1.0; // Food reward
                snake.grow();
                food_eaten = true;
                break; // Only one snake can eat at a time
            }
        }

        // Generate new food if eaten
        if food_eaten {
            let mut rng = rand::thread_rng();
            self.food = Food::generate_random(self.width, self.height, &self.snakes, &mut rng);
        }

        // Check collisions
        for i in 0..self.snakes.len() {
            if !self.active_agents[i] {
                continue;
            }

            let (snake_collisions, wall_collisions) = self.check_collisions(i);

            if snake_collisions || wall_collisions {
                rewards[i] = -1.0; // Death penalty
                terminated[i] = true;
                self.active_agents[i] = false;
            }
        }

        // Check if all agents are dead
        let all_dead = self.active_agents.iter().all(|&active| !active);
        if all_dead {
            self.done = true;
        }

        // Check step limit
        let truncated = vec![self.steps >= self.max_steps; self.snakes.len()];
        if self.steps >= self.max_steps {
            self.done = true;
        }

        MultiAgentResult {
            observations: self.get_observations(),
            rewards,
            terminated,
            truncated,
            active_agents: self.active_agents.clone(),
        }
    }

    /// Check collisions for a specific snake
    fn check_collisions(&self, snake_id: usize) -> (bool, bool) {
        let snake = &self.snakes[snake_id];

        // Wall collision
        let wall_collision = snake.collides_with_wall(self.width, self.height);

        // Self collision
        let self_collision = snake.collides_with_self();

        // Other snake collisions
        let mut other_collision = false;
        for (i, other_snake) in self.snakes.iter().enumerate() {
            if i != snake_id {
                if snake.collides_with_other(other_snake) {
                    other_collision = true;
                    break;
                }
            }
        }

        (self_collision || other_collision, wall_collision)
    }

    /// Get observations for all agents
    pub fn get_observations(&self) -> Vec<Vec<f32>> {
        self.snakes.iter().enumerate().map(|(i, snake)| {
            if self.active_agents[i] {
                self.get_observation_for_snake(i)
            } else {
                vec![0.0; 6] // Zero observation for dead agents
            }
        }).collect()
    }

    /// Get observation for a specific snake
    fn get_observation_for_snake(&self, snake_id: usize) -> Vec<f32> {
        let snake = &self.snakes[snake_id];

        // Relative food position
        let dx = (self.food.position.x - snake.head.x) as f32 / self.width as f32;
        let dy = (self.food.position.y - snake.head.y) as f32 / self.height as f32;

        // Snake direction (one-hot encoded)
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
        grid[self.food.position.y as usize][self.food.position.x as usize] = Cell::Food;

        // Place all snakes
        for snake in &self.snakes {
            for (i, &pos) in snake.body.iter().enumerate() {
                if pos.y >= 0 && pos.y < self.height && pos.x >= 0 && pos.x < self.width {
                    let cell = if i == 0 {
                        Cell::SnakeHead(snake.id)
                    } else {
                        Cell::SnakeBody(snake.id)
                    };
                    grid[pos.y as usize][pos.x as usize] = cell;
                }
            }
        }

        let scores: Vec<i32> = self.snakes.iter().map(|s| s.length as i32).collect();

        GameState {
            grid,
            scores,
            active_agents: self.active_agents.clone(),
            episode: self.episode,
            steps: self.steps,
        }
    }
}

impl MultiAgentEnvironment for MultiAgentSnakeEnv {
    fn reset(&mut self) {
        self.reset();
    }

    fn step(&mut self, actions: &[i64]) -> MultiAgentResult {
        self.step(actions)
    }

    fn num_agents(&self) -> usize {
        self.snakes.len()
    }

    fn observation_space(&self) -> Vec<crate::env::SpaceInfo> {
        vec![crate::env::SpaceInfo {
            shape: vec![6], // [food_dx, food_dy, dir_up, dir_down, dir_left, dir_right]
            space_type: crate::env::SpaceType::Box,
        }; self.snakes.len()]
    }

    fn action_space(&self) -> Vec<crate::env::SpaceInfo> {
        vec![crate::env::SpaceInfo {
            shape: vec![4], // 4 directions
            space_type: crate::env::SpaceType::Discrete(4),
        }; self.snakes.len()]
    }

    fn active_agents(&self) -> Vec<bool> {
        self.active_agents.clone()
    }

    fn render(&self) -> Vec<u8> {
        // For now, return empty vector - rendering handled by GameState
        vec![]
    }

    fn close(&mut self) {
        // Nothing to clean up
    }
}
