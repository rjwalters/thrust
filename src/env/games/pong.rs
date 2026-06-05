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

/// Single-player Pong: agent controls left paddle, rule-based opponent on right.
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
        self.ball_dx = if toward_agent { -BALL_SPEED } else { BALL_SPEED };
        self.ball_dy = rng.gen_range(-0.5f32..0.5f32) * BALL_SPEED;
    }

    fn clamp_paddle(y: f32) -> f32 {
        y.clamp(PADDLE_H, 1.0 - PADDLE_H)
    }

    /// Raw rendering state: [ball_x, ball_y, left_y, right_y, left_score, right_score]
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
}

impl Default for Pong {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for Pong {
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
        if old_bx - BALL_R >= LEFT_X
            && self.ball_x - BALL_R < LEFT_X
            && self.ball_dx < 0.0
        {
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
        if old_bx + BALL_R <= RIGHT_X
            && self.ball_x + BALL_R > RIGHT_X
            && self.ball_dx > 0.0
        {
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
