# Multi-Agent Training Architecture for Thrust

**Goal**: Efficient population-based training for cooperative and competitive multi-agent games, leveraging Rust's performance and safety guarantees.

## Vision

Thrust will be the **first pure-Rust RL library** with first-class support for:
- **Self-play training**: Agents play against copies of themselves
- **Population training**: Multiple diverse agents train simultaneously
- **Mixed scenarios**: Cooperative, competitive, and mixed-motive games
- **Heterogeneous compute**: CPU simulation + multi-GPU learning

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Main Thread: Coordinator                        │
│  - Spawns worker threads                                         │
│  - Monitors training metrics                                     │
│  - Handles checkpointing                                         │
│  - Manages population lifecycle                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌──────▼────────┐
│ Simulator      │  │ Learner 0     │  │ Learner N     │
│ Thread         │  │ Thread        │  │ Thread        │
│                │  │               │  │               │
│ • EnvPool      │  │ • Agent 0     │  │ • Agent N     │
│ • N parallel   │  │ • PPO Trainer │  │ • PPO Trainer │
│   games        │  │ • GPU 0       │  │ • GPU N%GPUs  │
│ • Matchmaking  │  │ • Own exp     │  │ • Own exp     │
│ • Experience   │  │               │  │               │
│   routing      │  │ Future:       │  │ Future:       │
│                │  │ • Shared exp  │  │ • Shared exp  │
│ CPU-bound      │  │               │  │               │
└────────────────┘  └───────────────┘  └───────────────┘
        │                   │                   │
        │    crossbeam      │                   │
        │    channels       │                   │
        └───────────────────┴───────────────────┘
```

## Core Components

### 1. Population Manager

```rust
pub struct Population {
    /// Individual agent policies
    agents: Vec<Agent>,

    /// Population configuration
    config: PopulationConfig,

    /// Shared metrics across all agents
    metrics: Arc<RwLock<PopulationMetrics>>,
}

pub struct Agent {
    /// Unique agent identifier
    id: AgentId,

    /// Policy network (CPU copy for inference)
    policy: Arc<RwLock<MlpPolicy>>,

    /// Agent-specific metrics
    fitness: f64,
    games_played: usize,
    total_reward: f64,

    /// Policy version (for staleness tracking)
    version: AtomicU64,
}

pub struct PopulationConfig {
    /// Number of agents in population
    size: usize,

    /// Matchmaking strategy
    matchmaking: MatchmakingStrategy,

    /// Learning mode
    learning_mode: LearningMode,

    /// Update frequency (steps between policy syncs)
    update_interval: usize,
}

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

pub enum LearningMode {
    /// Each agent learns only from own experience
    OnPolicy,

    /// Agents can sample from shared buffer
    OffPolicy { buffer_size: usize },

    /// Mix of both
    Hybrid { on_policy_ratio: f64 },
}
```

### 2. Multi-Agent Environment Trait

```rust
pub trait MultiAgentEnvironment: Environment {
    /// Number of agents in the game
    fn num_agents(&self) -> usize;

    /// Get observation for specific agent
    fn get_observation(&self, agent_id: usize) -> Self::Observation;

    /// Step with multiple actions (one per agent)
    fn step_multi(&mut self, actions: &[Self::Action]) -> MultiAgentResult<Self>;

    /// Which agents are still active
    fn active_agents(&self) -> Vec<bool>;
}

pub struct MultiAgentResult<E: MultiAgentEnvironment> {
    /// Observations for each agent
    pub observations: Vec<E::Observation>,

    /// Rewards for each agent
    pub rewards: Vec<f32>,

    /// Terminal states for each agent
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,

    /// Additional info
    pub info: HashMap<String, Value>,
}
```

### 3. Simulator Thread

```rust
pub struct GameSimulator<E: MultiAgentEnvironment> {
    /// Pool of parallel environments
    env_pool: EnvPool<E>,

    /// Population of agents
    population: Arc<RwLock<Population>>,

    /// Send experiences to learner threads
    experience_senders: Vec<Sender<Experience>>,

    /// Receive policy updates from learners
    policy_receiver: Receiver<PolicyUpdate>,

    /// Matchmaking strategy
    matchmaker: Box<dyn Matchmaker>,
}

impl<E: MultiAgentEnvironment> GameSimulator<E> {
    pub fn run(&mut self) {
        loop {
            // 1. Matchmaking: assign agents to games
            let matches = self.matchmaker.create_matches(
                self.population.read().unwrap(),
                self.env_pool.len(),
            );

            // 2. Reset environments with assigned agents
            for (env_id, agent_ids) in matches.iter().enumerate() {
                self.env_pool.reset_env(env_id).unwrap();
            }

            // 3. Run episodes
            let experiences = self.run_episodes(&matches);

            // 4. Route experiences to learners
            self.route_experiences(experiences);

            // 5. Update policies from learners
            self.sync_policies();
        }
    }

    fn run_episodes(&mut self, matches: &[Vec<AgentId>]) -> Vec<(AgentId, Experience)> {
        let mut all_experiences = Vec::new();

        // Run until all envs are done
        while !self.env_pool.all_done() {
            // Get current observations
            let observations = self.env_pool.get_observations();

            // Each agent selects action
            for (env_id, agent_ids) in matches.iter().enumerate() {
                let mut actions = Vec::new();

                for agent_id in agent_ids {
                    let agent_obs = observations[env_id][*agent_id];
                    let policy = self.population.read().unwrap()
                        .agents[*agent_id].policy.read().unwrap();

                    // Get action from policy (no grad)
                    let action = policy.get_action(&agent_obs).0;
                    actions.push(action);
                }

                // Step environment
                let results = self.env_pool.step_env(env_id, &actions).unwrap();

                // Record experiences
                for (i, agent_id) in agent_ids.iter().enumerate() {
                    let experience = Experience {
                        observation: results.observations[i].clone(),
                        action: actions[i],
                        reward: results.rewards[i],
                        next_observation: results.next_observations[i].clone(),
                        terminated: results.terminated[i],
                        truncated: results.truncated[i],
                        agent_id: *agent_id,
                    };
                    all_experiences.push((*agent_id, experience));
                }
            }
        }

        all_experiences
    }

    fn route_experiences(&self, experiences: Vec<(AgentId, Experience)>) {
        for (agent_id, exp) in experiences {
            self.experience_senders[agent_id].send(exp).unwrap();
        }
    }
}
```

### 4. Learner Thread

```rust
pub struct PolicyLearner {
    /// Agent ID this learner is training
    agent_id: AgentId,

    /// Policy network (GPU copy)
    policy: MlpPolicy,

    /// PPO trainer
    trainer: PPOTrainer<MlpPolicy>,

    /// Receive experiences from simulator
    experience_receiver: Receiver<Experience>,

    /// Send policy updates to simulator
    policy_sender: Sender<PolicyUpdate>,

    /// Local experience buffer
    buffer: RolloutBuffer,

    /// Training configuration
    config: LearnerConfig,
}

impl PolicyLearner {
    pub fn train(mut self) {
        let mut step = 0;

        loop {
            // 1. Collect experiences until buffer is full
            self.buffer.reset();

            while !self.buffer.is_full() {
                match self.experience_receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok(exp) => self.buffer.add_experience(exp),
                    Err(_) => continue,
                }
            }

            // 2. Compute advantages
            self.buffer.compute_advantages(
                self.config.gamma,
                self.config.gae_lambda,
            );

            // 3. Train on batch
            let batch = self.buffer.get_batch();
            let stats = self.trainer.train_step_with_policy(
                &self.policy,
                &batch.observations,
                &batch.actions,
                &batch.old_log_probs,
                &batch.old_values,
                &batch.advantages,
                &batch.returns,
                |p, obs, acts| p.evaluate_actions(obs, acts),
            ).unwrap();

            tracing::info!(
                "Agent {} | Step {} | Loss: {:.3} | Policy: {:.3} | Entropy: {:.3}",
                self.agent_id,
                step,
                stats.total_loss,
                stats.policy_loss,
                stats.entropy,
            );

            step += 1;

            // 4. Periodically send policy update to simulator
            if step % self.config.update_interval == 0 {
                let weights = self.policy.export_weights();
                let update = PolicyUpdate {
                    agent_id: self.agent_id,
                    weights,
                    version: step,
                };
                self.policy_sender.send(update).unwrap();
            }
        }
    }
}
```

### 5. Coordinator

```rust
pub struct MultiAgentTrainer<E: MultiAgentEnvironment> {
    /// Population configuration
    population_config: PopulationConfig,

    /// Environment factory
    env_factory: Box<dyn Fn() -> E>,

    /// Number of parallel games
    num_games: usize,

    /// Training handles
    handles: Vec<JoinHandle<()>>,
}

impl<E: MultiAgentEnvironment> MultiAgentTrainer<E> {
    pub fn new(
        population_config: PopulationConfig,
        env_factory: Box<dyn Fn() -> E>,
        num_games: usize,
    ) -> Self {
        Self {
            population_config,
            env_factory,
            num_games,
            handles: Vec::new(),
        }
    }

    pub fn train(&mut self) -> Result<()> {
        // 1. Create population
        let population = Arc::new(RwLock::new(
            Population::new(self.population_config.clone())
        ));

        // 2. Create communication channels
        let (exp_senders, exp_receivers): (Vec<_>, Vec<_>) =
            (0..self.population_config.size)
                .map(|_| crossbeam_channel::unbounded())
                .unzip();

        let (policy_sender, policy_receiver) = crossbeam_channel::unbounded();

        // 3. Spawn learner threads (one per agent)
        for agent_id in 0..self.population_config.size {
            let learner = PolicyLearner {
                agent_id,
                policy: population.read().unwrap().agents[agent_id]
                    .policy.read().unwrap().clone(),
                trainer: PPOTrainer::new(/* config */)?,
                experience_receiver: exp_receivers[agent_id].clone(),
                policy_sender: policy_sender.clone(),
                buffer: RolloutBuffer::new(/* config */),
                config: LearnerConfig::default(),
            };

            let handle = std::thread::spawn(move || {
                learner.train();
            });

            self.handles.push(handle);
        }

        // 4. Spawn simulator thread
        let simulator = GameSimulator {
            env_pool: EnvPool::new(&self.env_factory, self.num_games),
            population: population.clone(),
            experience_senders: exp_senders,
            policy_receiver,
            matchmaker: self.population_config.matchmaking.create_matchmaker(),
        };

        let sim_handle = std::thread::spawn(move || {
            simulator.run();
        });

        self.handles.push(sim_handle);

        // 5. Main thread: monitoring and checkpointing
        loop {
            std::thread::sleep(Duration::from_secs(60));

            // Log population metrics
            let pop = population.read().unwrap();
            tracing::info!("Population Metrics:");
            for agent in &pop.agents {
                tracing::info!(
                    "  Agent {} | Fitness: {:.2} | Games: {}",
                    agent.id,
                    agent.fitness,
                    agent.games_played,
                );
            }

            // Save checkpoint
            self.save_checkpoint(&pop)?;
        }
    }
}
```

## Example Usage

### Self-Play (Single Agent)

```rust
// Train a single agent playing against itself
let config = PopulationConfig {
    size: 1,
    matchmaking: MatchmakingStrategy::SelfPlay,
    learning_mode: LearningMode::OnPolicy,
    update_interval: 100,
};

let mut trainer = MultiAgentTrainer::new(
    config,
    Box::new(|| CartPole::new()),
    num_games: 64,
);

trainer.train()?;
```

### Population Training (Diverse Agents)

```rust
// Train 8 diverse agents
let config = PopulationConfig {
    size: 8,
    matchmaking: MatchmakingStrategy::RoundRobin,
    learning_mode: LearningMode::OnPolicy,
    update_interval: 100,
};

let mut trainer = MultiAgentTrainer::new(
    config,
    Box::new(|| BucketBrigade::new(scenario)),
    num_games: 64,
);

trainer.train()?;
```

### Competitive Training

```rust
// Two populations compete
let config = PopulationConfig {
    size: 16,  // 8 red team + 8 blue team
    matchmaking: MatchmakingStrategy::TeamBased {
        teams: vec![0..8, 8..16],
    },
    learning_mode: LearningMode::OnPolicy,
    update_interval: 100,
};

let mut trainer = MultiAgentTrainer::new(
    config,
    Box::new(|| CompetitiveGame::new()),
    num_games: 64,
);

trainer.train()?;
```

## Implementation Phases

### Phase 1: Core Infrastructure ✅ (Mostly Done)
- ✅ Environment trait
- ✅ EnvPool for parallel envs
- ✅ PPO trainer
- ✅ MlpPolicy
- ✅ RolloutBuffer

### Phase 2: Multi-Agent Basics ✅ (Core Infrastructure Complete)
- ✅ `MultiAgentEnvironment` trait
- ✅ `Population` and `Agent` structs
- ✅ `GameSimulator` thread (structure complete, run loops TODO)
- ✅ `PolicyLearner` thread (structure complete, train loop TODO)
- ✅ Basic matchmaking (Random, RoundRobin, FitnessBased, SelfPlay)
- ✅ Thread-safe with `Send` trait (using `StdRng` instead of `ThreadRng`)
- ✅ 15 unit tests passing
- [ ] Communication channels (crossbeam) - next step
- [ ] Complete simulator run loops
- [ ] Complete learner training loops

### Phase 3: Advanced Features
- [ ] Off-policy learning with shared buffer
- [ ] Fitness-based matchmaking
- [ ] Policy staleness tracking
- [ ] Importance sampling for off-policy
- [ ] Population metrics and analysis

### Phase 4: Research Features
- [ ] Nash equilibrium computation
- [ ] Population diversity metrics
- [ ] Evolution-inspired operators (mutation, crossover)
- [ ] Multi-objective optimization

## Performance Targets

- **Throughput**: 10,000+ steps/sec with 8 agents on single GPU
- **GPU Utilization**: >90% across all learner threads
- **Memory**: <4GB per learner (including buffer)
- **Scalability**: Linear speedup with additional GPUs

## Advantages over Python Implementations

1. **Type Safety**: Compile-time guarantees for agent IDs, channels, etc.
2. **Zero-Copy**: `Arc<RwLock<>>` for shared policies (no serialization)
3. **Native Threading**: No GIL, true parallelism
4. **Performance**: Rust envs ~100x faster than Python
5. **Memory Safety**: No data races, use-after-free, etc.

## Integration with BucketBrigade

```rust
// In bucket-brigade repo, use Thrust for training:
use thrust_rl::multi_agent::*;

impl MultiAgentEnvironment for BucketBrigade {
    fn num_agents(&self) -> usize { 4 }

    fn step_multi(&mut self, actions: &[Action]) -> MultiAgentResult<Self> {
        // BucketBrigade-specific logic
        self.step_internal(actions)
    }
}

// Then train with Thrust:
let config = PopulationConfig { /* ... */ };
let trainer = MultiAgentTrainer::new(
    config,
    Box::new(|| BucketBrigade::new("trivial_cooperation")),
    num_games: 128,
);
trainer.train()?;
```

## Next Steps

1. **Create `src/multi_agent/` module**
2. **Implement `MultiAgentEnvironment` trait**
3. **Build `GameSimulator` and `PolicyLearner`**
4. **Add example: `examples/train_multi_agent.rs`**
5. **Benchmark: Compare vs Python population training**
6. **Document: Add to README and docs**

This will be a **killer feature** for Thrust - the first pure-Rust library with production-ready multi-agent training!
