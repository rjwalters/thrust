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
    /// Whether the snake is alive
    pub alive: bool,
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
            alive: true,
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

    /// Move snake in current direction with wraparound boundaries (torus)
    pub fn move_forward_wrap(&mut self, width: i32, height: i32) {
        let new_head = self.head.add_direction(self.direction).wrap(width, height);

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

    /// Check if snake is alive
    pub fn is_alive(&self) -> bool {
        self.alive
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_creation() {
        let pos = Position::new(5, 5);
        let snake = Snake::new(0, pos, Direction::Right);

        assert_eq!(snake.id, 0);
        assert_eq!(snake.head, pos);
        assert_eq!(snake.direction, Direction::Right);
        assert_eq!(snake.length, 1);
        assert_eq!(snake.body.len(), 1);
        assert!(snake.is_alive());
    }

    #[test]
    fn test_snake_movement() {
        let start_pos = Position::new(5, 5);
        let mut snake = Snake::new(0, start_pos, Direction::Right);

        snake.move_forward();
        assert_eq!(snake.head, Position::new(6, 5));
        assert_eq!(snake.body.len(), 1); // Tail removed since not growing

        snake.grow();
        snake.move_forward();
        assert_eq!(snake.head, Position::new(7, 5));
        assert_eq!(snake.body.len(), 2); // Body grew
        assert_eq!(snake.length, 2);
    }

    #[test]
    fn test_snake_direction_change() {
        let mut snake = Snake::new(0, Position::new(5, 5), Direction::Right);

        // Valid direction change
        snake.change_direction(Direction::Down);
        assert_eq!(snake.direction, Direction::Down);

        // Invalid direction change (immediate reversal should be ignored)
        snake.change_direction(Direction::Up);
        assert_eq!(snake.direction, Direction::Down); // Should remain Down
    }

    #[test]
    fn test_snake_collision_detection() {
        let mut snake = Snake::new(0, Position::new(1, 1), Direction::Right);
        snake.grow();
        snake.move_forward(); // Now at (2,1) with body at (1,1)

        // Wall collision
        assert!(snake.collides_with_wall(3, 3)); // Grid is 3x3 (0-2)

        // Self collision - move to create collision
        let mut snake2 = Snake::new(0, Position::new(5, 5), Direction::Right);
        snake2.grow();
        snake2.move_forward();
        snake2.change_direction(Direction::Down);
        snake2.move_forward();
        snake2.change_direction(Direction::Left);
        snake2.move_forward();
        // Now should be colliding with itself
        assert!(snake2.collides_with_self());
    }

    #[test]
    fn test_snake_food_eating() {
        let snake_pos = Position::new(5, 5);
        let food_pos = Position::new(5, 5);
        let snake = Snake::new(0, snake_pos, Direction::Right);

        assert!(snake.eats_food(&food_pos));

        let food_pos2 = Position::new(6, 5);
        assert!(!snake.eats_food(&food_pos2));
    }

    #[test]
    fn test_food_generation() {
        let mut rng = rand::thread_rng();
        let snakes = vec![
            Snake::new(0, Position::new(0, 0), Direction::Right),
            Snake::new(1, Position::new(2, 2), Direction::Right),
        ];

        let food = Food::generate_random(5, 5, &snakes, &mut rng);
        let food_pos = food.position;

        // Food should be within bounds
        assert!(food_pos.in_bounds(5, 5));

        // Food should not be on any snake
        for snake in &snakes {
            assert!(!snake.get_all_positions().contains(&food_pos));
        }
    }
}
