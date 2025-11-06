//! Snake entity and movement logic
//!
//! This module handles individual snake behavior, movement,
//! collision detection, and growth mechanics.

use super::types::{Direction, Position};
use std::collections::VecDeque;

/// Individual snake in the game
#[derive(Debug, Clone)]
pub struct Snake {
    /// Snake ID (for multi-agent games)
    pub id: usize,
    /// Head position
    pub head: Position,
    /// Body positions (deque for efficient head/tail operations)
    pub body: VecDeque<Position>,
    /// Current direction
    pub direction: Direction,
    /// Length of snake
    pub length: usize,
}

impl Snake {
    /// Create new snake at position
    pub fn new(id: usize, start_pos: Position, start_direction: Direction) -> Self {
        let mut body = VecDeque::new();
        body.push_back(start_pos);

        Self {
            id,
            head: start_pos,
            body,
            direction: start_direction,
            length: 1,
        }
    }

    /// Move snake in current direction
    pub fn move_forward(&mut self) {
        let new_head = self.head.add_direction(self.direction);

        // Add new head position
        self.body.push_front(new_head);
        self.head = new_head;

        // Remove tail if not growing
        if self.body.len() > self.length {
            self.body.pop_back();
        }
    }

    /// Change direction (with validation)
    pub fn change_direction(&mut self, new_direction: Direction) {
        // Prevent immediate reversal
        if new_direction != self.direction.opposite() {
            self.direction = new_direction;
        }
    }

    /// Grow snake by one segment
    pub fn grow(&mut self) {
        self.length += 1;
    }

    /// Check if snake collides with walls
    pub fn collides_with_wall(&self, width: i32, height: i32) -> bool {
        !self.head.in_bounds(width, height)
    }

    /// Check if snake collides with itself
    pub fn collides_with_self(&self) -> bool {
        // Check if head collides with any body segment
        self.body.iter().skip(1).any(|&pos| pos == self.head)
    }

    /// Check if snake collides with another snake
    pub fn collides_with_other(&self, other: &Snake) -> bool {
        // Check head collision with other snake's body
        other.body.iter().any(|&pos| pos == self.head)
    }

    /// Check if snake eats food at position
    pub fn eats_food(&self, food_pos: &Position) -> bool {
        self.head == *food_pos
    }

    /// Get all positions occupied by snake
    pub fn get_all_positions(&self) -> Vec<Position> {
        self.body.iter().cloned().collect()
    }

    /// Get snake's current length
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if snake is alive (has length > 0)
    pub fn is_alive(&self) -> bool {
        self.length > 0
    }
}

/// Food pellet in the game
#[derive(Debug, Clone)]
pub struct Food {
    pub position: Position,
}

impl Food {
    /// Create food at position
    pub fn new(position: Position) -> Self {
        Self { position }
    }

    /// Generate random food position avoiding snakes
    pub fn generate_random(
        width: i32,
        height: i32,
        snakes: &[Snake],
        rng: &mut impl rand::Rng,
    ) -> Self {
        loop {
            let x = rng.gen_range(0..width);
            let y = rng.gen_range(0..height);
            let pos = Position::new(x, y);

            // Check if position is occupied by any snake
            let occupied = snakes.iter().any(|snake| {
                snake.get_all_positions().contains(&pos)
            });

            if !occupied {
                return Self::new(pos);
            }
        }
    }
}
