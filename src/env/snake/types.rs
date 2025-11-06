//! Basic types for the Snake environment
//!
//! This module defines the fundamental types used in the Snake game
//! including directions, positions, and game state.

/// Direction a snake can move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    /// Create direction from action index
    pub fn from_action(action: i64) -> Self {
        match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        }
    }

    /// Convert direction to (dx, dy) delta
    pub fn to_delta(self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }

    /// Get opposite direction (for validation)
    pub fn opposite(self) -> Self {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

/// Position on the grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    /// Create new position
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Add direction delta to position
    pub fn add_direction(&self, direction: Direction) -> Self {
        let (dx, dy) = direction.to_delta();
        Self::new(self.x + dx, self.y + dy)
    }

    /// Check if position is within bounds
    pub fn in_bounds(&self, width: i32, height: i32) -> bool {
        self.x >= 0 && self.x < width && self.y >= 0 && self.y < height
    }

    /// Manhattan distance to another position
    pub fn manhattan_distance(&self, other: &Position) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

/// Game cell content
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Food,
    SnakeHead(usize), // Snake ID
    SnakeBody(usize), // Snake ID
}

/// Game state for rendering
#[derive(Debug, Clone)]
pub struct GameState {
    pub grid: Vec<Vec<Cell>>,
    pub scores: Vec<i32>,
    pub active_agents: Vec<bool>,
    pub episode: usize,
    pub steps: usize,
}

impl GameState {
    pub fn new(width: usize, height: usize, num_agents: usize) -> Self {
        Self {
            grid: vec![vec![Cell::Empty; width]; height],
            scores: vec![0; num_agents],
            active_agents: vec![true; num_agents],
            episode: 0,
            steps: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_from_action() {
        assert_eq!(Direction::from_action(0), Direction::Up);
        assert_eq!(Direction::from_action(1), Direction::Down);
        assert_eq!(Direction::from_action(2), Direction::Left);
        assert_eq!(Direction::from_action(3), Direction::Right);
        assert_eq!(Direction::from_action(4), Direction::Right); // Out of bounds
    }

    #[test]
    fn test_direction_to_delta() {
        assert_eq!(Direction::Up.to_delta(), (0, -1));
        assert_eq!(Direction::Down.to_delta(), (0, 1));
        assert_eq!(Direction::Left.to_delta(), (-1, 0));
        assert_eq!(Direction::Right.to_delta(), (1, 0));
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::Up.opposite(), Direction::Down);
        assert_eq!(Direction::Down.opposite(), Direction::Up);
        assert_eq!(Direction::Left.opposite(), Direction::Right);
        assert_eq!(Direction::Right.opposite(), Direction::Left);
    }

    #[test]
    fn test_position_new() {
        let pos = Position::new(5, 10);
        assert_eq!(pos.x, 5);
        assert_eq!(pos.y, 10);
    }

    #[test]
    fn test_position_add_direction() {
        let pos = Position::new(5, 5);

        assert_eq!(pos.add_direction(Direction::Up), Position::new(5, 4));
        assert_eq!(pos.add_direction(Direction::Down), Position::new(5, 6));
        assert_eq!(pos.add_direction(Direction::Left), Position::new(4, 5));
        assert_eq!(pos.add_direction(Direction::Right), Position::new(6, 5));
    }

    #[test]
    fn test_position_in_bounds() {
        let pos = Position::new(5, 5);

        assert!(pos.in_bounds(10, 10));
        assert!(!pos.in_bounds(5, 10)); // x = 5, width = 5 (0-4)
        assert!(!pos.in_bounds(10, 5)); // y = 5, height = 5 (0-4)
    }

    #[test]
    fn test_position_manhattan_distance() {
        let pos1 = Position::new(0, 0);
        let pos2 = Position::new(3, 4);

        assert_eq!(pos1.manhattan_distance(&pos2), 7);
        assert_eq!(pos2.manhattan_distance(&pos1), 7);
    }

    #[test]
    fn test_game_state_creation() {
        let state = GameState::new(5, 5, 2);

        assert_eq!(state.grid.len(), 5);
        assert_eq!(state.grid[0].len(), 5);
        assert_eq!(state.scores.len(), 2);
        assert_eq!(state.active_agents.len(), 2);
        assert!(state.active_agents.iter().all(|&x| x)); // All should be true
        assert_eq!(state.episode, 0);
        assert_eq!(state.steps, 0);
    }
}
