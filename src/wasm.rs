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
    episode_steps: u32,
    best_score: u32,
    policy: Option<crate::policy::inference::InferenceModel>,
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
            episode_steps: 0,
            best_score: 0,
            policy: None,
        }
    }

    /// Reset the environment and start a new episode
    #[wasm_bindgen]
    pub fn reset(&mut self) -> Vec<f32> {
        // Update best score before resetting
        if self.episode_steps > self.best_score {
            self.best_score = self.episode_steps;
        }

        self.env.reset();
        self.episode += 1;
        self.episode_steps = 0;
        self.env.get_state().to_vec()
    }

    /// Take a step in the environment
    /// Returns: [obs0, obs1, obs2, obs3, reward, terminated, truncated]
    #[wasm_bindgen]
    pub fn step(&mut self, action: i32) -> Vec<f32> {
        let result = self.env.step(action as i64);
        self.episode_steps += 1;

        let mut output = result.observation.clone();
        output.push(result.reward);
        output.push(if result.terminated { 1.0 } else { 0.0 });
        output.push(if result.truncated { 1.0 } else { 0.0 });

        // Update best score if episode ended
        if result.terminated || result.truncated {
            if self.episode_steps > self.best_score {
                self.best_score = self.episode_steps;
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
        self.episode_steps
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

    /// Load policy from JSON string
    /// The JSON should contain the InferenceModel structure
    #[wasm_bindgen]
    pub fn load_policy_json(&mut self, json: &str) -> Result<(), JsValue> {
        let policy: crate::policy::inference::InferenceModel =
            serde_json::from_str(json)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse policy JSON: {}", e)))?;

        self.policy = Some(policy);
        Ok(())
    }

    /// Get policy action using the loaded model
    /// Returns action index (0 or 1) or -1 if no policy loaded
    #[wasm_bindgen]
    pub fn get_policy_action(&self) -> i32 {
        if let Some(ref policy) = self.policy {
            let state = self.env.get_state();
            policy.get_action(&state) as i32
        } else {
            -1 // No policy loaded
        }
    }

    /// Check if policy is loaded
    #[wasm_bindgen]
    pub fn has_policy(&self) -> bool {
        self.policy.is_some()
    }
}

/// WASM bindings for Snake environment
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmSnake {
    env: SnakeEnv,
    episode: u32,
    policy: Option<crate::inference::snake::SnakeCNNInference>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmSnake {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32, num_agents: usize) -> Self {
        console_error_panic_hook::set_once();
        Self {
            env: SnakeEnv::new_multi(width, height, num_agents),
            episode: 0,
            policy: None,
        }
    }

    /// Reset the environment
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.env.reset();
        self.episode += 1;
    }

    /// Step the environment with actions for all agents
    /// actions: array of actions where each is 0=up, 1=down, 2=left, 3=right
    #[wasm_bindgen]
    pub fn step(&mut self, actions: &[i32]) {
        let actions_i64: Vec<i64> = actions.iter().map(|&a| a as i64).collect();
        let _ = self.env.step_multi(&actions_i64);
    }

    /// Get grid observation for a specific agent
    /// Returns flattened grid [channels * height * width] where:
    /// - Channel 0: Own snake body
    /// - Channel 1: Own snake head
    /// - Channel 2: Other snakes
    /// - Channel 3: Food
    /// - Channel 4: Walls
    #[wasm_bindgen]
    pub fn get_observation(&self, agent_id: usize) -> Vec<f32> {
        self.env.get_grid_observation(agent_id)
    }

    /// Get number of agents
    #[wasm_bindgen]
    pub fn num_agents(&self) -> usize {
        self.env.num_agents
    }

    /// Get active agents (alive = true, dead = false)
    #[wasm_bindgen]
    pub fn active_agents(&self) -> Vec<u8> {
        self.env.snakes.iter().map(|s| if s.is_alive() { 1 } else { 0 }).collect()
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

    /// Get all snake positions for rendering
    /// Returns flattened array: [num_snakes, len0, x0, y0, x1, y1, ..., len1, x0, y0, ...]
    #[wasm_bindgen]
    pub fn get_snake_positions(&self) -> Vec<i32> {
        let mut positions = Vec::new();

        // First, add the number of snakes
        positions.push(self.env.snakes.len() as i32);

        // Then add each snake's positions
        for snake in &self.env.snakes {
            positions.push(snake.body.len() as i32);
            for pos in &snake.body {
                positions.push(pos.x);
                positions.push(pos.y);
            }
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

    /// Load policy from JSON string
    /// The JSON should contain the SnakeCNNInference model structure
    #[wasm_bindgen]
    pub fn load_policy_json(&mut self, json: &str) -> Result<(), JsValue> {
        let policy: crate::inference::snake::SnakeCNNInference =
            serde_json::from_str(json)
                .map_err(|e| JsValue::from_str(&format!("Failed to parse policy JSON: {}", e)))?;

        self.policy = Some(policy);
        Ok(())
    }

    /// Get policy action for a specific agent using the loaded model
    /// Returns action index (0=up, 1=down, 2=left, 3=right) or -1 if no policy loaded
    #[wasm_bindgen]
    pub fn get_policy_action(&self, agent_id: usize) -> i32 {
        if let Some(ref policy) = self.policy {
            let observation = self.env.get_grid_observation(agent_id);
            policy.get_action(&observation) as i32
        } else {
            -1 // No policy loaded
        }
    }

    /// Check if policy is loaded
    #[wasm_bindgen]
    pub fn has_policy(&self) -> bool {
        self.policy.is_some()
    }
}

// Add panic hook for better error messages in browser console
#[cfg(feature = "wasm")]
use console_error_panic_hook;
