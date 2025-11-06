//! CartPole-v1 environment
//!
//! A classic reinforcement learning benchmark where a pole is balanced on a
//! cart. The goal is to prevent the pole from falling over by applying forces
//! to the cart.
//!
//! # Physics
//!
//! The cart-pole system follows these dynamics:
//! - State: [x, x_dot, theta, theta_dot] (cart position, cart velocity, pole
//!   angle, pole angular velocity)
//! - Actions: 0 (push left) or 1 (push right)
//! - Reward: +1 for each timestep the pole stays upright
//! - Termination: Pole angle > 12° or cart position > 2.4
//!
//! # Reference
//!
//! Based on OpenAI Gym CartPole-v1:
//! <https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py>

use anyhow::Result;
use rand::Rng;

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// CartPole-v1 environment
///
/// A pole is attached to a cart moving along a frictionless track.
/// The goal is to balance the pole by applying forces to the cart.
#[derive(Debug)]
pub struct CartPole {
    // State variables
    x: f32,         // Cart position
    x_dot: f32,     // Cart velocity
    theta: f32,     // Pole angle (radians)
    theta_dot: f32, // Pole angular velocity

    // Episode tracking
    steps: usize,
    max_steps: usize,

    // Physics constants (matching Gym CartPole-v1)
    gravity: f32,
    #[allow(dead_code)]
    mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    length: f32,           // Half-length of pole
    pole_mass_length: f32, // pole_mass * length
    force_mag: f32,
    tau: f32, // Time step

    // Thresholds
    theta_threshold: f32,
    x_threshold: f32,
}

impl CartPole {
    /// Create a new CartPole environment with default parameters
    ///
    /// Physics constants match OpenAI Gym CartPole-v1:
    /// - gravity = 9.8 m/s²
    /// - cart mass = 1.0 kg
    /// - pole mass = 0.1 kg
    /// - pole half-length = 0.5 m
    /// - force magnitude = 10.0 N
    /// - timestep = 0.02 s
    pub fn new() -> Self {
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5; // Half-length of pole
        let pole_mass_length = mass_pole * length;
        let force_mag = 10.0;
        let tau = 0.02;
        let theta_threshold = 12.0 * 2.0 * std::f32::consts::PI / 360.0; // ~0.2094 radians
        let x_threshold = 2.4;
        let max_steps = 500;

        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_steps,
            gravity,
            mass_cart,
            mass_pole,
            total_mass,
            length,
            pole_mass_length,
            force_mag,
            tau,
            theta_threshold,
            x_threshold,
        }
    }

    /// Reset state to random initial conditions
    ///
    /// All state variables are initialized with small random perturbations
    /// around equilibrium (uniform distribution in [-0.05, 0.05])
    fn reset_state(&mut self) {
        let mut rng = rand::thread_rng();

        // Initialize with small random perturbations
        self.x = rng.gen_range(-0.05..0.05);
        self.x_dot = rng.gen_range(-0.05..0.05);
        self.theta = rng.gen_range(-0.05..0.05);
        self.theta_dot = rng.gen_range(-0.05..0.05);
    }

    /// Perform one physics simulation step using Euler integration
    ///
    /// Implements the cart-pole dynamics equations:
    /// ```text
    /// temp = (force + pole_mass_length * theta_dot² * sin(theta)) / total_mass
    /// theta_acc = (g * sin(theta) - cos(theta) * temp) /
    ///             (length * (4/3 - mass_pole * cos²(theta) / total_mass))
    /// x_acc = temp - pole_mass_length * theta_acc * cos(theta) / total_mass
    /// ```
    fn physics_step(&mut self, action: i64) {
        // Convert action to force: 0 = push left (-10N), 1 = push right (+10N)
        let force = if action == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // Compute accelerations using cart-pole dynamics
        let temp = (force + self.pole_mass_length * self.theta_dot * self.theta_dot * sin_theta)
            / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length
                * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;

        // Euler integration
        self.x_dot += self.tau * x_acc;
        self.x += self.tau * self.x_dot;
        self.theta_dot += self.tau * theta_acc;
        self.theta += self.tau * self.theta_dot;
    }

    /// Check if episode should terminate
    ///
    /// Termination conditions:
    /// - Cart position exceeds ±2.4
    /// - Pole angle exceeds ±12°
    fn is_terminated(&self) -> bool {
        self.x < -self.x_threshold
            || self.x > self.x_threshold
            || self.theta < -self.theta_threshold
            || self.theta > self.theta_threshold
    }

    /// Check if episode should be truncated (max steps reached)
    fn is_truncated(&self) -> bool {
        self.steps >= self.max_steps
    }

    /// Get current observation [x, x_dot, theta, theta_dot]
    fn get_observation(&self) -> Vec<f32> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPole {
    type Observation = Vec<f32>;
    type Action = i64;

    fn reset(&mut self) -> Result<Self::Observation> {
        self.reset_state();
        self.steps = 0;
        Ok(self.get_observation())
    }

    fn step(&mut self, action: Self::Action) -> Result<StepResult<Self::Observation>> {
        // Perform physics step
        self.physics_step(action);

        // Increment step counter
        self.steps += 1;

        // Check termination conditions
        let terminated = self.is_terminated();
        let truncated = self.is_truncated();

        // Reward is 1.0 for each step the pole stays upright
        // Episode ends when terminated or truncated
        let reward = if terminated || truncated { 0.0 } else { 1.0 };

        Ok(StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: StepInfo::default(),
        })
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![4], dtype: SpaceType::Continuous }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![], dtype: SpaceType::Discrete(2) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_init() {
        let env = CartPole::new();
        assert_eq!(env.max_steps, 500);
        assert_eq!(env.gravity, 9.8);
        assert_eq!(env.mass_cart, 1.0);
        assert_eq!(env.mass_pole, 0.1);
    }

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new();
        let obs = env.reset().unwrap();

        assert_eq!(obs.len(), 4, "Observation should have 4 elements");
        assert_eq!(env.steps, 0, "Steps should be reset to 0");

        // Check that initial state is small (close to equilibrium)
        for &val in &obs {
            assert!(val.abs() < 0.1, "Initial state should be small perturbation, got {}", val);
        }
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        let result = env.step(1).unwrap();

        assert_eq!(result.observation.len(), 4, "Observation should have 4 elements");
        assert!(result.reward == 0.0 || result.reward == 1.0, "Reward should be 0 or 1");
    }

    #[test]
    fn test_cartpole_termination() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        // Manually set state to exceed position threshold
        env.x = 3.0; // Exceeds x_threshold of 2.4

        let result = env.step(0).unwrap();
        assert!(
            result.terminated,
            "Episode should terminate when cart exceeds position threshold"
        );

        // Reset and test angle threshold
        env.reset().unwrap();
        env.theta = 0.5; // Exceeds theta_threshold of ~0.2094 radians

        let result = env.step(0).unwrap();
        assert!(result.terminated, "Episode should terminate when pole exceeds angle threshold");
    }

    #[test]
    fn test_cartpole_truncation() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        // Manually set steps to max
        env.steps = env.max_steps - 1;

        let result = env.step(0).unwrap();
        assert!(result.truncated, "Episode should truncate at max steps");
    }

    #[test]
    fn test_cartpole_rewards() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        // First step should give reward
        let result = env.step(1).unwrap();
        if !result.terminated && !result.truncated {
            assert_eq!(result.reward, 1.0, "Should receive reward of 1.0 per step");
        }
    }

    #[test]
    fn test_cartpole_action_left() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        let x_before = env.x;
        env.step(0).unwrap(); // Action 0 = push left

        // With leftward force, cart should generally move left (though dynamics are
        // complex) Just verify physics ran without panicking
        assert_ne!(env.x, x_before, "State should change after step");
    }

    #[test]
    fn test_cartpole_action_right() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        let x_before = env.x;
        env.step(1).unwrap(); // Action 1 = push right

        // Just verify physics ran without panicking
        assert_ne!(env.x, x_before, "State should change after step");
    }

    #[test]
    fn test_cartpole_observation_space() {
        let env = CartPole::new();
        let obs_space = env.observation_space();

        assert_eq!(obs_space.shape, vec![4]);
        assert!(matches!(obs_space.dtype, SpaceType::Continuous));
    }

    #[test]
    fn test_cartpole_action_space() {
        let env = CartPole::new();
        let action_space = env.action_space();

        assert_eq!(action_space.shape, Vec::<usize>::new());
        assert!(matches!(action_space.dtype, SpaceType::Discrete(2)));
    }

    #[test]
    fn test_cartpole_episode() {
        let mut env = CartPole::new();
        env.reset().unwrap();

        let mut steps = 0;

        // Run episode with random actions
        for _ in 0..1000 {
            let action = steps % 2; // Alternate actions
            let result = env.step(action).unwrap();
            steps += 1;

            if result.terminated || result.truncated {
                break;
            }
        }

        assert!(steps > 0, "Episode should run at least one step");
        assert!(steps <= 500, "Episode should not exceed max_steps");
    }
}
