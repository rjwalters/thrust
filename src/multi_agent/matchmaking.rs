//! Matchmaking strategies for multi-agent games
//!
//! Determines which agents play together in each game instance.

use super::population::{AgentId, Population};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Trait for matchmaking strategies
pub trait Matchmaker: Send {
    /// Create matches for the next round of games
    ///
    /// # Arguments
    ///
    /// * `population` - The agent population
    /// * `num_games` - Number of parallel games to create
    /// * `agents_per_game` - Number of agents in each game
    ///
    /// # Returns
    ///
    /// Vector of matches, where each match is a vector of agent IDs
    fn create_matches(
        &mut self,
        population: &Population,
        num_games: usize,
        agents_per_game: usize,
    ) -> Vec<Vec<AgentId>>;
}

/// Matchmaking strategy configuration
#[derive(Debug, Clone)]
pub enum MatchmakingStrategy {
    /// Random selection
    Random,

    /// Round-robin (each plays each equally)
    RoundRobin,

    /// Fitness-based (similar skill levels)
    FitnessBased { window_size: usize },

    /// Self-play (agent plays copies of itself)
    SelfPlay,
}

impl MatchmakingStrategy {
    /// Create a matchmaker instance from this strategy
    pub fn create_matchmaker(&self) -> Box<dyn Matchmaker> {
        match self {
            Self::Random => Box::new(RandomMatchmaker::new()),
            Self::RoundRobin => Box::new(RoundRobinMatchmaker::new()),
            Self::FitnessBased { window_size } => {
                Box::new(FitnessBasedMatchmaker::new(*window_size))
            }
            Self::SelfPlay => Box::new(SelfPlayMatchmaker::new()),
        }
    }
}

/// Random matchmaking - randomly sample agents for each game
pub struct RandomMatchmaker {
    rng: StdRng,
}

impl RandomMatchmaker {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }
}

impl Matchmaker for RandomMatchmaker {
    fn create_matches(
        &mut self,
        population: &Population,
        num_games: usize,
        agents_per_game: usize,
    ) -> Vec<Vec<AgentId>> {
        let pop_size = population.agents.len();
        let mut matches = Vec::with_capacity(num_games);

        for _ in 0..num_games {
            let mut game_agents = Vec::with_capacity(agents_per_game);
            for _ in 0..agents_per_game {
                let agent_id = self.rng.gen_range(0..pop_size);
                game_agents.push(agent_id);
            }
            matches.push(game_agents);
        }

        matches
    }
}

/// Round-robin matchmaking - ensure each agent plays every other agent equally
pub struct RoundRobinMatchmaker {
    current_round: usize,
}

impl RoundRobinMatchmaker {
    pub fn new() -> Self {
        Self { current_round: 0 }
    }
}

impl Matchmaker for RoundRobinMatchmaker {
    fn create_matches(
        &mut self,
        population: &Population,
        num_games: usize,
        agents_per_game: usize,
    ) -> Vec<Vec<AgentId>> {
        let pop_size = population.agents.len();
        let mut matches = Vec::with_capacity(num_games);

        // Simple round-robin: rotate agents through games
        for game_id in 0..num_games {
            let mut game_agents = Vec::with_capacity(agents_per_game);
            let offset = (self.current_round + game_id) % pop_size;

            for i in 0..agents_per_game {
                let agent_id = (offset + i) % pop_size;
                game_agents.push(agent_id);
            }
            matches.push(game_agents);
        }

        self.current_round += 1;
        matches
    }
}

/// Fitness-based matchmaking - match agents with similar skill levels
pub struct FitnessBasedMatchmaker {
    window_size: usize,
    rng: StdRng,
}

impl FitnessBasedMatchmaker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            rng: StdRng::from_entropy(),
        }
    }
}

impl Matchmaker for FitnessBasedMatchmaker {
    fn create_matches(
        &mut self,
        population: &Population,
        num_games: usize,
        agents_per_game: usize,
    ) -> Vec<Vec<AgentId>> {
        // Sort agents by fitness
        let mut ranked: Vec<(AgentId, f64)> = population
            .agents
            .iter()
            .map(|a| (a.id, a.fitness))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let pop_size = ranked.len();
        let mut matches = Vec::with_capacity(num_games);

        for _ in 0..num_games {
            // Pick a base rank within valid range
            let max_base = pop_size.saturating_sub(self.window_size);
            let base_rank = if max_base > 0 {
                self.rng.gen_range(0..max_base)
            } else {
                0
            };

            // Sample agents from window around base rank
            let mut game_agents = Vec::with_capacity(agents_per_game);
            let window_end = (base_rank + self.window_size).min(pop_size);

            for _ in 0..agents_per_game {
                let idx = self.rng.gen_range(base_rank..window_end);
                game_agents.push(ranked[idx].0);
            }
            matches.push(game_agents);
        }

        matches
    }
}

/// Self-play matchmaking - each game uses copies of a single agent
pub struct SelfPlayMatchmaker {
    rng: StdRng,
}

impl SelfPlayMatchmaker {
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }
}

impl Matchmaker for SelfPlayMatchmaker {
    fn create_matches(
        &mut self,
        population: &Population,
        num_games: usize,
        agents_per_game: usize,
    ) -> Vec<Vec<AgentId>> {
        let pop_size = population.agents.len();
        let mut matches = Vec::with_capacity(num_games);

        for _ in 0..num_games {
            // Pick one agent for this game
            let agent_id = self.rng.gen_range(0..pop_size);

            // All players in this game are the same agent
            let game_agents = vec![agent_id; agents_per_game];
            matches.push(game_agents);
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_agent::population::PopulationConfig;

    fn create_test_population(size: usize) -> Population {
        let config = PopulationConfig {
            size,
            ..Default::default()
        };
        Population::new(config, 4, 2, 64)
    }

    #[test]
    fn test_random_matchmaker() {
        let pop = create_test_population(8);
        let mut mm = RandomMatchmaker::new();

        let matches = mm.create_matches(&pop, 4, 4);

        assert_eq!(matches.len(), 4);
        for game in matches {
            assert_eq!(game.len(), 4);
            for &agent_id in &game {
                assert!(agent_id < 8);
            }
        }
    }

    #[test]
    fn test_round_robin_matchmaker() {
        let pop = create_test_population(8);
        let mut mm = RoundRobinMatchmaker::new();

        let matches1 = mm.create_matches(&pop, 2, 4);
        let matches2 = mm.create_matches(&pop, 2, 4);

        assert_eq!(matches1.len(), 2);
        assert_eq!(matches2.len(), 2);

        // Different rounds should have different matches
        assert_ne!(matches1, matches2);
    }

    #[test]
    fn test_fitness_based_matchmaker() {
        let mut pop = create_test_population(8);

        // Assign different fitness values
        for i in 0..8 {
            pop.update_fitness(i, (i * 10) as f64);
        }

        let mut mm = FitnessBasedMatchmaker::new(3);
        let matches = mm.create_matches(&pop, 4, 4);

        assert_eq!(matches.len(), 4);
        for game in matches {
            assert_eq!(game.len(), 4);
        }
    }

    #[test]
    fn test_self_play_matchmaker() {
        let pop = create_test_population(8);
        let mut mm = SelfPlayMatchmaker::new();

        let matches = mm.create_matches(&pop, 4, 4);

        assert_eq!(matches.len(), 4);
        for game in matches {
            assert_eq!(game.len(), 4);
            // All agents in a game should be the same
            let first = game[0];
            assert!(game.iter().all(|&id| id == first));
        }
    }
}
