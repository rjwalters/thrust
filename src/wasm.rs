//! WebAssembly bindings for browser-based visualization
//!
//! This module provides JavaScript bindings for our Rust environments
//! so they can be rendered in the browser.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::env::{Environment, cartpole::CartPole, snake::SnakeEnv};

/// WASM bindings for CartPole environment
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmCartPole {
    env: CartPole,
    episode: u32,
    total_steps: u32,
    best_score: u32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmCartPole {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            env: CartPole::new(),
            episode: 0,
            total_steps: 0,
            best_score: 0,
        }
    }

    /// Reset the environment and start a new episode
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Vec<f32> {
        self.env.reset();
        self.episode += 1;
        self.env.get_state().to_vec()
    }

    /// Take a step in the environment
    /// Returns: [obs0, obs1, obs2, obs3, reward, terminated, truncated]
    #[wasm_bindgen]
    pub fn step(&mut self, action: i64) -> Vec<f32> {
        let result = self.env.step(action);
        self.total_steps += 1;

        let mut output = result.observation.clone();
        output.push(result.reward);
        output.push(if result.terminated { 1.0 } else { 0.0 });
        output.push(if result.truncated { 1.0 } else { 0.0 });

        // Update best score if episode ended
        if result.terminated || result.truncated {
            let episode_steps = self.total_steps;
            if episode_steps > self.best_score {
                self.best_score = episode_steps;
            }
        }

        output
    }

    /// Get current episode number
    #[wasm_bindgen]
    pub fn get_episode(&self) -> u32 {
        self.episode
    }

    /// Get total steps in current episode
    #[wasm_bindgen]
    pub fn get_steps(&self) -> u32 {
        self.total_steps
    }

    /// Get best score
    #[wasm_bindgen]
    pub fn get_best_score(&self) -> u32 {
        self.best_score
    }

    /// Get the current state for rendering
    /// Returns: [x, x_dot, theta, theta_dot]
    #[wasm_bindgen]
    pub fn get_state(&self) -> Vec<f32> {
        self.env.get_state().to_vec()
    }
}

/// WASM bindings for Snake environment
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSnake {
    env: SnakeEnv,
    episode: u32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSnake {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32) -> Self {
        console_error_panic_hook::set_once();
        Self {
            env: SnakeEnv::new(width, height),
            episode: 0,
        }
    }

    /// Reset the environment
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.env.reset();
        self.episode += 1;
    }

    /// Step the environment with a single action
    /// action: 0 = up, 1 = down, 2 = left, 3 = right
    #[wasm_bindgen]
    pub fn step(&mut self, action: i32) {
        use crate::env::Environment;
        let _ = self.env.step(action as i64);
    }

    /// Get observation for a specific agent
    #[wasm_bindgen]
    pub fn get_observation(&self, _agent_id: usize) -> Vec<f32> {
        // For now, return empty - we'll implement this when we have multi-agent support
        vec![]
    }

    /// Get number of agents
    #[wasm_bindgen]
    pub fn num_agents(&self) -> usize {
        1 // Single-agent environment
    }

    /// Get active agents (alive = true, dead = false)
    #[wasm_bindgen]
    pub fn active_agents(&self) -> Vec<u8> {
        vec![if self.env.snake.is_alive() { 1 } else { 0 }]
    }

    /// Get grid dimensions
    #[wasm_bindgen]
    pub fn get_width(&self) -> i32 {
        self.env.width
    }

    #[wasm_bindgen]
    pub fn get_height(&self) -> i32 {
        self.env.height
    }

    /// Get snake positions for rendering
    /// Returns flattened array: [len, x0, y0, x1, y1, ...]
    #[wasm_bindgen]
    pub fn get_snake_positions(&self) -> Vec<i32> {
        let mut positions = Vec::new();

        positions.push(self.env.snake.body.len() as i32);
        for pos in &self.env.snake.body {
            positions.push(pos.x);
            positions.push(pos.y);
        }

        positions
    }

    /// Get food position
    /// Returns array: [x, y]
    #[wasm_bindgen]
    pub fn get_food_positions(&self) -> Vec<i32> {
        vec![self.env.food.x, self.env.food.y]
    }

    /// Get current episode
    #[wasm_bindgen]
    pub fn get_episode(&self) -> u32 {
        self.episode
    }
}

// Add panic hook for better error messages in browser console
#[cfg(feature = "wasm")]
use console_error_panic_hook;
