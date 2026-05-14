//! Pong environment
//!
//! A classic two-player Pong game for adversarial multi-agent reinforcement learning.
//! Two paddles compete to bounce a ball back and forth, earning points when the opponent
//! misses.
//!
//! # Game Mechanics
//!
//! - Two paddles (left and right) that can move up or down
//! - Ball bounces off walls and paddles
//! - Point scored when ball passes opponent's paddle
//! - First to score wins, or highest score after max_steps
//!
//! # State Space
//!
//! Each agent observes 6 values:
//! - Ball x position (normalized 0-1)
//! - Ball y position (normalized 0-1)
//! - Ball x velocity (normalized)
//! - Ball y velocity (normalized)
//! - Own paddle y position (normalized 0-1)
//! - Opponent paddle y position (normalized 0-1)
//!
//! # Action Space
//!
//! 3 discrete actions:
//! - 0: Move up
//! - 1: Stay
//! - 2: Move down
//!
//! # Rewards
//!
//! - +1 for scoring a point
//! - -1 for opponent scoring
//! - Small reward for hitting the ball (+0.1)
//!
//! # Reference
//!
//! Inspired by Atari Pong and multi-agent RL benchmarks

use rand::Rng;

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

/// Pong environment with two competing agents
#[derive(Debug, Clone)]
pub struct Pong {
    // Game dimensions
    width: f32,
    height: f32,

    // Ball state
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    ball_radius: f32,
    ball_speed: f32,

    // Paddle state
    paddle_left_y: f32,
    paddle_right_y: f32,
    paddle_width: f32,
    paddle_height: f32,
    paddle_speed: f32,

    // Score tracking
    score_left: i32,
    score_right: i32,
    max_score: i32,

    // Episode tracking
    steps: usize,
    max_steps: usize,

    // Control: which agent to act (0=left, 1=right, or both for simultaneous)
    active_agent: Option<usize>, // None means both act simultaneously

    // Self-play configuration
    mirror_observations: bool, // If true, right agent sees mirrored observations
}

impl Pong {
    /// Create a new Pong environment with default parameters
    pub fn new() -> Self {
        Self {
            width: 160.0,
            height: 210.0,
            ball_x: 80.0,
            ball_y: 105.0,
            ball_vx: 0.0,
            ball_vy: 0.0,
            ball_radius: 2.0,
            ball_speed: 2.0,
            paddle_left_y: 105.0,
            paddle_right_y: 105.0,
            paddle_width: 4.0,
            paddle_height: 15.0,
            paddle_speed: 4.0,
            score_left: 0,
            score_right: 0,
            max_score: 21,
            steps: 0,
            max_steps: 2000,
            active_agent: None, // Simultaneous actions
            mirror_observations: false,
        }
    }

    /// Create a Pong environment configured for self-play training
    ///
    /// # Arguments
    /// * `mirror_observations` - If true, the right agent receives mirrored observations,
    ///   allowing a single shared policy to play both sides
    ///
    /// # Example
    /// ```
    /// use thrust_rl::env::Pong;
    ///
    /// let pong = Pong::for_selfplay(true);
    /// // Both agents can now use the same policy network
    /// ```
    pub fn for_selfplay(mirror_observations: bool) -> Self {
        let mut env = Self::new();
        env.mirror_observations = mirror_observations;
        env
    }

    /// Reset ball to center with random velocity
    fn reset_ball(&mut self) {
        let mut rng = rand::thread_rng();

        self.ball_x = self.width / 2.0;
        self.ball_y = self.height / 2.0;

        // Random direction with some angle
        let angle: f32 = rng.gen_range(-0.5_f32..0.5_f32);
        let direction = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };

        self.ball_vx = direction * self.ball_speed * angle.cos();
        self.ball_vy = self.ball_speed * angle.sin();
    }

    /// Get normalized observation for an agent
    ///
    /// # Arguments
    /// * `agent_id` - 0 for left paddle, 1 for right paddle
    ///
    /// # Returns
    /// A 6-element vector: [ball_x, ball_y, ball_vx, ball_vy, own_paddle_y, opp_paddle_y]
    ///
    /// If `mirror_observations` is true and agent_id is 1, the observation is mirrored
    /// so the right agent "thinks" it's playing from the left side. This enables
    /// a single shared policy to control both agents in self-play.
    pub fn get_agent_observation(&self, agent_id: usize) -> Vec<f32> {
        let (own_paddle, opp_paddle) = if agent_id == 0 {
            (self.paddle_left_y, self.paddle_right_y)
        } else {
            (self.paddle_right_y, self.paddle_left_y)
        };

        // If mirroring is enabled for right agent, flip x-coordinates
        if self.mirror_observations && agent_id == 1 {
            vec![
                1.0 - (self.ball_x / self.width),           // Mirrored ball x
                self.ball_y / self.height,                   // Ball y unchanged
                -self.ball_vx / (self.ball_speed * 2.0),    // Flipped ball vx
                self.ball_vy / (self.ball_speed * 2.0),     // Ball vy unchanged
                own_paddle / self.height,                    // Own paddle y
                opp_paddle / self.height,                    // Opp paddle y
            ]
        } else {
            vec![
                self.ball_x / self.width,                    // Ball x (0-1)
                self.ball_y / self.height,                   // Ball y (0-1)
                self.ball_vx / (self.ball_speed * 2.0),      // Ball vx (normalized)
                self.ball_vy / (self.ball_speed * 2.0),      // Ball vy (normalized)
                own_paddle / self.height,                    // Own paddle y (0-1)
                opp_paddle / self.height,                    // Opp paddle y (0-1)
            ]
        }
    }

    /// Step with two actions (one for each agent)
    pub fn step_multi(&mut self, action_left: i64, action_right: i64) -> (StepResult, StepResult) {
        self.steps += 1;

        // Move paddles based on actions
        self.paddle_left_y = self.calculate_new_paddle_pos(self.paddle_left_y, action_left);
        self.paddle_right_y = self.calculate_new_paddle_pos(self.paddle_right_y, action_right);

        // Update ball position
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        let mut reward_left = 0.0;
        let mut reward_right = 0.0;
        let mut ball_hit = false;

        // Ball collision with top/bottom walls
        if self.ball_y - self.ball_radius <= 0.0 {
            self.ball_y = self.ball_radius;
            self.ball_vy = -self.ball_vy;
        } else if self.ball_y + self.ball_radius >= self.height {
            self.ball_y = self.height - self.ball_radius;
            self.ball_vy = -self.ball_vy;
        }

        // Ball collision with left paddle
        if self.ball_x - self.ball_radius <= self.paddle_width {
            if self.ball_y >= self.paddle_left_y - self.paddle_height / 2.0
                && self.ball_y <= self.paddle_left_y + self.paddle_height / 2.0
            {
                self.ball_x = self.paddle_width + self.ball_radius;
                self.ball_vx = -self.ball_vx;

                // Add spin based on where ball hit paddle
                let hit_pos = (self.ball_y - self.paddle_left_y) / (self.paddle_height / 2.0);
                self.ball_vy += hit_pos * 0.5;

                reward_left += 0.1; // Small reward for hitting ball
                ball_hit = true;
            }
        }

        // Ball collision with right paddle
        if self.ball_x + self.ball_radius >= self.width - self.paddle_width {
            if self.ball_y >= self.paddle_right_y - self.paddle_height / 2.0
                && self.ball_y <= self.paddle_right_y + self.paddle_height / 2.0
            {
                self.ball_x = self.width - self.paddle_width - self.ball_radius;
                self.ball_vx = -self.ball_vx;

                // Add spin
                let hit_pos = (self.ball_y - self.paddle_right_y) / (self.paddle_height / 2.0);
                self.ball_vy += hit_pos * 0.5;

                reward_right += 0.1; // Small reward for hitting ball
                ball_hit = true;
            }
        }

        // Limit ball speed
        let max_speed = self.ball_speed * 2.0;
        let speed = (self.ball_vx * self.ball_vx + self.ball_vy * self.ball_vy).sqrt();
        if speed > max_speed {
            self.ball_vx = (self.ball_vx / speed) * max_speed;
            self.ball_vy = (self.ball_vy / speed) * max_speed;
        }

        let mut terminated = false;

        // Check if ball went out of bounds (score)
        if self.ball_x < 0.0 {
            // Right player scores
            self.score_right += 1;
            reward_right += 1.0;
            reward_left -= 1.0;
            self.reset_ball();

            if self.score_right >= self.max_score {
                terminated = true;
            }
        } else if self.ball_x > self.width {
            // Left player scores
            self.score_left += 1;
            reward_left += 1.0;
            reward_right -= 1.0;
            self.reset_ball();

            if self.score_left >= self.max_score {
                terminated = true;
            }
        }

        // Truncate if max steps reached
        let truncated = self.steps >= self.max_steps;

        // Build results for both agents
        let obs_left = self.get_agent_observation(0);
        let obs_right = self.get_agent_observation(1);

        let result_left = StepResult {
            observation: obs_left,
            reward: reward_left,
            terminated,
            truncated,
            info: StepInfo::default(),
        };

        let result_right = StepResult {
            observation: obs_right,
            reward: reward_right,
            terminated,
            truncated,
            info: StepInfo::default(),
        };

        (result_left, result_right)
    }

    /// Calculate new paddle position based on action
    fn calculate_new_paddle_pos(&self, current_y: f32, action: i64) -> f32 {
        let new_y = match action {
            0 => current_y - self.paddle_speed, // Move up
            1 => current_y,                     // Stay
            2 => current_y + self.paddle_speed, // Move down
            _ => current_y,
        };

        // Clamp paddle position
        let min_y = self.paddle_height / 2.0;
        let max_y = self.height - self.paddle_height / 2.0;
        new_y.clamp(min_y, max_y)
    }

    /// Get current scores (left, right)
    pub fn get_scores(&self) -> (i32, i32) {
        (self.score_left, self.score_right)
    }

    /// Get ball position (for rendering)
    pub fn get_ball_pos(&self) -> (f32, f32) {
        (self.ball_x, self.ball_y)
    }

    /// Get paddle positions (for rendering)
    pub fn get_paddle_positions(&self) -> (f32, f32) {
        (self.paddle_left_y, self.paddle_right_y)
    }
}

impl Default for Pong {
    fn default() -> Self {
        Self::new()
    }
}

// Implement Environment trait for single-agent control (controls left paddle, right is random)
impl Environment for Pong {
    fn reset(&mut self) {
        self.ball_x = self.width / 2.0;
        self.ball_y = self.height / 2.0;
        self.paddle_left_y = self.height / 2.0;
        self.paddle_right_y = self.height / 2.0;
        self.score_left = 0;
        self.score_right = 0;
        self.steps = 0;
        self.reset_ball();
    }

    fn get_observation(&self) -> Vec<f32> {
        self.get_agent_observation(0) // Left paddle's perspective
    }

    fn step(&mut self, action: i64) -> StepResult {
        // For single-agent mode, use a simple opponent AI
        let opponent_action = self.simple_ai_action(1);
        let (result_left, _result_right) = self.step_multi(action, opponent_action);
        result_left
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![6],
            space_type: SpaceType::Box,
        }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo {
            shape: vec![1],
            space_type: SpaceType::Discrete(3), // Up, Stay, Down
        }
    }

    fn render(&self) -> Vec<u8> {
        // Simple ASCII-style rendering for debugging
        // Could be expanded to proper pixel rendering
        vec![]
    }

    fn close(&mut self) {
        // Nothing to clean up
    }
}

impl Pong {
    /// Simple AI opponent that tracks the ball
    fn simple_ai_action(&self, agent_id: usize) -> i64 {
        let paddle_y = if agent_id == 0 {
            self.paddle_left_y
        } else {
            self.paddle_right_y
        };

        let target = self.ball_y;
        let diff = target - paddle_y;

        if diff < -5.0 {
            0 // Move up
        } else if diff > 5.0 {
            2 // Move down
        } else {
            1 // Stay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pong_creation() {
        let pong = Pong::new();
        assert_eq!(pong.score_left, 0);
        assert_eq!(pong.score_right, 0);
    }

    #[test]
    fn test_observation_space() {
        let pong = Pong::new();
        let obs = pong.get_observation();
        assert_eq!(obs.len(), 6);

        // Check normalization
        for &val in &obs {
            assert!(val >= -1.0 && val <= 2.0); // Allows for some velocity
        }
    }

    #[test]
    fn test_step() {
        let mut pong = Pong::new();
        pong.reset();

        let result = pong.step(0); // Move up
        assert_eq!(result.observation.len(), 6);
    }

    #[test]
    fn test_multi_step() {
        let mut pong = Pong::new();
        pong.reset();

        let (result_left, result_right) = pong.step_multi(1, 1); // Both stay
        assert_eq!(result_left.observation.len(), 6);
        assert_eq!(result_right.observation.len(), 6);
    }
}
