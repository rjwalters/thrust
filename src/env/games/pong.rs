//! Single-player Pong environment.
//!
//! The agent controls the left paddle; the right paddle is a simple
//! rule-based tracker. Used as a PPO benchmark — see
//! `examples/games/pong/train_pong.rs`.

use rand::Rng;

use crate::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

const LEFT_X: f32 = 0.05;
const RIGHT_X: f32 = 0.95;
const BALL_R: f32 = 0.015;
const PADDLE_H: f32 = 0.1; // half-height (total = 0.2)
const BALL_SPEED: f32 = 0.018;
const PADDLE_SPEED: f32 = 0.025;
const OPPONENT_SPEED: f32 = 0.015;
const MAX_SCORE: u32 = 7;
const MAX_STEPS: usize = 2000;

/// Single-player Pong: agent controls left paddle, rule-based opponent on
/// right.
pub struct Pong {
    ball_x: f32,
    ball_y: f32,
    ball_dx: f32,
    ball_dy: f32,
    left_y: f32,
    right_y: f32,
    left_score: u32,
    right_score: u32,
    steps: usize,
}

impl Pong {
    /// Construct a new Pong environment with the ball served toward the agent.
    pub fn new() -> Self {
        let mut p = Self {
            ball_x: 0.5,
            ball_y: 0.5,
            ball_dx: BALL_SPEED,
            ball_dy: 0.0,
            left_y: 0.5,
            right_y: 0.5,
            left_score: 0,
            right_score: 0,
            steps: 0,
        };
        p.serve(true);
        p
    }

    fn serve(&mut self, toward_agent: bool) {
        let mut rng = rand::thread_rng();
        self.ball_x = 0.5;
        self.ball_y = 0.3 + rng.gen_range(0.0f32..0.4f32);
        self.ball_dx = if toward_agent {
            -BALL_SPEED
        } else {
            BALL_SPEED
        };
        self.ball_dy = rng.gen_range(-0.5f32..0.5f32) * BALL_SPEED;
    }

    fn clamp_paddle(y: f32) -> f32 {
        y.clamp(PADDLE_H, 1.0 - PADDLE_H)
    }

    /// Raw rendering state: [ball_x, ball_y, left_y, right_y, left_score,
    /// right_score]
    pub fn get_state(&self) -> [f32; 6] {
        [
            self.ball_x,
            self.ball_y,
            self.left_y,
            self.right_y,
            self.left_score as f32,
            self.right_score as f32,
        ]
    }

    /// Step the environment with explicit actions for **both** paddles.
    ///
    /// Unlike [`Environment::step`], which drives the right paddle with a
    /// rule-based tracker (see `OPPONENT_SPEED`), this method takes a direct
    /// action for the right paddle. Both action arguments use the same
    /// encoding as `Environment::step`: `0 = up, 1 = stay, 2 = down`.
    ///
    /// Used by self-play training where the right paddle is controlled by a
    /// frozen policy snapshot rather than the heuristic. Ball physics,
    /// collisions, scoring, and termination are otherwise identical to
    /// [`Environment::step`].
    pub fn step_two(&mut self, left_action: i64, right_action: i64) -> StepResult {
        // Left paddle: 0 = up, 1 = stay, 2 = down
        self.left_y = match left_action {
            0 => Self::clamp_paddle(self.left_y - PADDLE_SPEED),
            2 => Self::clamp_paddle(self.left_y + PADDLE_SPEED),
            _ => self.left_y,
        };

        // Right paddle: same encoding, no rule-based override
        self.right_y = match right_action {
            0 => Self::clamp_paddle(self.right_y - PADDLE_SPEED),
            2 => Self::clamp_paddle(self.right_y + PADDLE_SPEED),
            _ => self.right_y,
        };

        let old_bx = self.ball_x;
        self.ball_x += self.ball_dx;
        self.ball_y += self.ball_dy;

        // Top/bottom wall bounces
        if self.ball_y - BALL_R < 0.0 {
            self.ball_y = BALL_R;
            self.ball_dy = self.ball_dy.abs();
        } else if self.ball_y + BALL_R > 1.0 {
            self.ball_y = 1.0 - BALL_R;
            self.ball_dy = -self.ball_dy.abs();
        }

        let mut reward = 0.0f32;

        // Left paddle collision: ball's left edge crosses LEFT_X this step
        if old_bx - BALL_R >= LEFT_X && self.ball_x - BALL_R < LEFT_X && self.ball_dx < 0.0 {
            if (self.ball_y - self.left_y).abs() <= PADDLE_H {
                let hit_pos = (self.ball_y - self.left_y) / PADDLE_H;
                self.ball_dx = BALL_SPEED;
                self.ball_dy = (self.ball_dy + hit_pos * BALL_SPEED * 0.6)
                    .clamp(-BALL_SPEED * 1.2, BALL_SPEED * 1.2);
                self.ball_x = LEFT_X + BALL_R;
                reward = 0.1;
            }
            // else: miss — ball continues to left wall
        }

        // Right paddle collision: ball's right edge crosses RIGHT_X this step
        if old_bx + BALL_R <= RIGHT_X && self.ball_x + BALL_R > RIGHT_X && self.ball_dx > 0.0 {
            if (self.ball_y - self.right_y).abs() <= PADDLE_H {
                let hit_pos = (self.ball_y - self.right_y) / PADDLE_H;
                self.ball_dx = -BALL_SPEED;
                self.ball_dy = (self.ball_dy + hit_pos * BALL_SPEED * 0.6)
                    .clamp(-BALL_SPEED * 1.2, BALL_SPEED * 1.2);
                self.ball_x = RIGHT_X - BALL_R;
            }
            // else: agent scores when ball hits right wall below
        }

        // Scoring: ball exits screen
        if self.ball_x - BALL_R < 0.0 {
            self.right_score += 1;
            reward = -1.0;
            let toward = rand::thread_rng().gen_bool(0.5);
            self.serve(toward);
        } else if self.ball_x + BALL_R > 1.0 {
            self.left_score += 1;
            reward = 1.0;
            let toward = rand::thread_rng().gen_bool(0.5);
            self.serve(toward);
        }

        self.steps += 1;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated: self.left_score >= MAX_SCORE || self.right_score >= MAX_SCORE,
            truncated: self.steps >= MAX_STEPS,
            info: StepInfo::default(),
        }
    }
}

/// Mirror a Pong observation to the right paddle's perspective.
///
/// The observation layout is
/// `[ball_x, ball_y, ball_dx, ball_dy, left_y, right_y]` (all normalized to
/// `[-1, 1]`). To let a single policy network play either side, the right
/// paddle's view negates the x-axis (`ball_x`, `ball_dx`) and swaps the
/// paddle positions (`left_y` <-> `right_y`).
///
/// Calling `mirror_observation(&mirror_observation(obs))` returns the
/// original observation (round-trip identity).
pub fn mirror_observation(obs: &[f32]) -> Vec<f32> {
    assert_eq!(obs.len(), 6, "Pong observation must be 6-dimensional");
    vec![-obs[0], obs[1], -obs[2], obs[3], obs[5], obs[4]]
}

impl Default for Pong {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for Pong {
    type Action = i64;

    fn reset(&mut self) {
        self.left_y = 0.5;
        self.right_y = 0.5;
        self.left_score = 0;
        self.right_score = 0;
        self.steps = 0;
        let toward = rand::thread_rng().gen_bool(0.5);
        self.serve(toward);
    }

    fn get_observation(&self) -> Vec<f32> {
        // All normalized to [-1, 1]
        vec![
            self.ball_x * 2.0 - 1.0,
            self.ball_y * 2.0 - 1.0,
            self.ball_dx / BALL_SPEED,
            self.ball_dy / BALL_SPEED,
            self.left_y * 2.0 - 1.0,
            self.right_y * 2.0 - 1.0,
        ]
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Agent (left paddle): 0 = up, 1 = stay, 2 = down
        self.left_y = match action {
            0 => Self::clamp_paddle(self.left_y - PADDLE_SPEED),
            2 => Self::clamp_paddle(self.left_y + PADDLE_SPEED),
            _ => self.left_y,
        };

        // Opponent (right paddle): tracks ball, speed-limited
        let diff = (self.ball_y - self.right_y).clamp(-OPPONENT_SPEED, OPPONENT_SPEED);
        self.right_y = Self::clamp_paddle(self.right_y + diff);

        let old_bx = self.ball_x;
        self.ball_x += self.ball_dx;
        self.ball_y += self.ball_dy;

        // Top/bottom wall bounces
        if self.ball_y - BALL_R < 0.0 {
            self.ball_y = BALL_R;
            self.ball_dy = self.ball_dy.abs();
        } else if self.ball_y + BALL_R > 1.0 {
            self.ball_y = 1.0 - BALL_R;
            self.ball_dy = -self.ball_dy.abs();
        }

        let mut reward = 0.0f32;

        // Left paddle collision: ball's left edge crosses LEFT_X this step
        if old_bx - BALL_R >= LEFT_X && self.ball_x - BALL_R < LEFT_X && self.ball_dx < 0.0 {
            if (self.ball_y - self.left_y).abs() <= PADDLE_H {
                let hit_pos = (self.ball_y - self.left_y) / PADDLE_H;
                self.ball_dx = BALL_SPEED;
                self.ball_dy = (self.ball_dy + hit_pos * BALL_SPEED * 0.6)
                    .clamp(-BALL_SPEED * 1.2, BALL_SPEED * 1.2);
                self.ball_x = LEFT_X + BALL_R;
                reward = 0.1;
            }
            // else: miss — ball continues to left wall
        }

        // Right paddle collision: ball's right edge crosses RIGHT_X this step
        if old_bx + BALL_R <= RIGHT_X && self.ball_x + BALL_R > RIGHT_X && self.ball_dx > 0.0 {
            if (self.ball_y - self.right_y).abs() <= PADDLE_H {
                let hit_pos = (self.ball_y - self.right_y) / PADDLE_H;
                self.ball_dx = -BALL_SPEED;
                self.ball_dy = (self.ball_dy + hit_pos * BALL_SPEED * 0.6)
                    .clamp(-BALL_SPEED * 1.2, BALL_SPEED * 1.2);
                self.ball_x = RIGHT_X - BALL_R;
            }
            // else: agent scores when ball hits right wall below
        }

        // Scoring: ball exits screen
        if self.ball_x - BALL_R < 0.0 {
            self.right_score += 1;
            reward = -1.0;
            let toward = rand::thread_rng().gen_bool(0.5);
            self.serve(toward);
        } else if self.ball_x + BALL_R > 1.0 {
            self.left_score += 1;
            reward = 1.0;
            let toward = rand::thread_rng().gen_bool(0.5);
            self.serve(toward);
        }

        self.steps += 1;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated: self.left_score >= MAX_SCORE || self.right_score >= MAX_SCORE,
            truncated: self.steps >= MAX_STEPS,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![6], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo { shape: vec![3], space_type: SpaceType::Discrete(3) }
    }

    fn render(&self) -> Vec<u8> {
        vec![]
    }

    fn close(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `step_two` with both actions = "stay" (1) must leave both paddles in
    /// place. This proves the rule-based right-paddle heuristic is bypassed:
    /// with `step` the right paddle would chase the ball, but with
    /// `step_two(1, 1)` it stays put while the ball still moves.
    #[test]
    fn step_two_stay_stay_freezes_paddles() {
        let mut p = Pong::new();
        let left_y0 = p.left_y;
        let right_y0 = p.right_y;
        let ball_x0 = p.ball_x;

        // Run several steps with "stay" actions
        for _ in 0..10 {
            let _ = p.step_two(1, 1);
        }

        // Both paddles must be unchanged (rule-based opponent is bypassed)
        assert_eq!(p.left_y, left_y0, "Left paddle moved despite stay action");
        assert_eq!(p.right_y, right_y0, "Right paddle moved — heuristic was not bypassed");

        // The ball must have moved (sanity check that the physics still ran)
        assert_ne!(p.ball_x, ball_x0, "Ball did not move during step_two");
    }

    /// `step_two(0, 0)` moves both paddles up. Confirms right-paddle action
    /// is honored verbatim.
    #[test]
    fn step_two_drives_right_paddle_independently() {
        let mut p = Pong::new();
        let right_y0 = p.right_y;

        // Drive right paddle UP without touching the heuristic
        let _ = p.step_two(1, 0);
        assert!(p.right_y < right_y0, "Right paddle did not move up with action=0");

        // Drive right paddle DOWN
        let before = p.right_y;
        let _ = p.step_two(1, 2);
        assert!(p.right_y > before, "Right paddle did not move down with action=2");
    }

    /// Round-trip: `mirror(mirror(obs)) == obs`.
    #[test]
    fn mirror_observation_round_trip() {
        let p = Pong::new();
        let obs = p.get_observation();
        let mirrored = mirror_observation(&obs);
        let unmirrored = mirror_observation(&mirrored);
        assert_eq!(obs.len(), 6);
        assert_eq!(unmirrored.len(), 6);
        for (a, b) in obs.iter().zip(unmirrored.iter()) {
            assert!((a - b).abs() < 1e-7, "round-trip mismatch: {a} vs {b}");
        }
    }

    /// The mirror transformation must negate x components and swap paddle
    /// positions while leaving y components untouched.
    #[test]
    fn mirror_observation_semantics() {
        let obs = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let m = mirror_observation(&obs);
        assert_eq!(m, vec![-0.3, 0.4, -0.5, 0.6, 0.8, 0.7]);
    }
}
