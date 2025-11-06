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
