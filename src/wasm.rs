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
        match self.env.reset() {
            Ok(obs) => {
                self.episode += 1;
                obs
            }
            Err(e) => {
                web_sys::console::error_1(&format!("Reset error: {}", e).into());
                vec![0.0; 4]
            }
        }
    }

    /// Take a step in the environment
    /// Returns: [obs0, obs1, obs2, obs3, reward, terminated, truncated]
    #[wasm_bindgen]
    pub fn step(&mut self, action: i64) -> Vec<f32> {
        match self.env.step(action) {
            Ok(result) => {
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
            Err(e) => {
                web_sys::console::error_1(&format!("Step error: {}", e).into());
                vec![0.0; 7]
            }
        }
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
    pub fn new(width: i32, height: i32, num_agents: usize) -> Self {
        console_error_panic_hook::set_once();
        Self {
            env: SnakeEnv::new(width, height, num_agents),
            episode: 0,
        }
    }

    /// Reset the environment
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        if let Err(e) = self.env.reset() {
            web_sys::console::error_1(&format!("Reset error: {}", e).into());
        }
        self.episode += 1;
    }

    /// Step with actions for all agents
    /// actions: array of action indices (one per agent)
    #[wasm_bindgen]
    pub fn step(&mut self, actions: &[i64]) {
        // Call the environment's step_multi method directly
        use crate::env::Environment;
        // For simplicity in WASM, just step without returning anything
        // Rendering will query state separately
        for &action in actions {
            let _ = self.env.step(action);
        }
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
        self.env.snakes.len()
    }

    /// Get active agents (alive = true, dead = false)
    #[wasm_bindgen]
    pub fn active_agents(&self) -> Vec<u8> {
        self.env.snakes.iter().map(|s| if s.alive { 1 } else { 0 }).collect()
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
    /// Returns flattened array: [agent0_len, agent0_x0, agent0_y0, agent0_x1, agent0_y1, ..., agent1_len, ...]
    #[wasm_bindgen]
    pub fn get_snake_positions(&self) -> Vec<i32> {
        let mut positions = Vec::new();

        for snake in &self.env.snakes {
            positions.push(snake.body.len() as i32);
            for pos in &snake.body {
                positions.push(pos.x);
                positions.push(pos.y);
            }
        }

        positions
    }

    /// Get food positions
    /// Returns flattened array: [x0, y0, x1, y1, ...]
    #[wasm_bindgen]
    pub fn get_food_positions(&self) -> Vec<i32> {
        self.env.food.iter().flat_map(|f| vec![f.x, f.y]).collect()
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
