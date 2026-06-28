//! Best-response **improvability oracle** for bucket-brigade (issue #259).
//!
//! # Why this exists
//!
//! The PPO best-response (BR) on the bucket-brigade no-convergence cells
//! fits its critic well (EV ≈ 0.57 at 8192 rollout, PR #256) yet does **not**
//! raise `mean_ep_return` — see the "Stage-1 cluster probe sweep" in
//! `docs/research/2026-06-bucket-brigade-validation.md`. Before instrumenting
//! or patching the PPO update (the failure mode of the closed predecessors
//! #239 / #252 / #241), issue #259 imposes an **improvability gate**: first
//! establish, via a *non-PPO* method, whether a better-than-uniform BR even
//! *exists* against the frozen uniform opponents `train_br_probe` trains
//! against.
//!
//! This module is that non-PPO method. It does **not** use a neural network,
//! Burn, or PPO at all. It scores a battery of hand-crafted and randomly
//! searched **scripted** policies for the single BR agent (agent 0) against
//! `N − 1` **frozen uniform-random** opponents, and reports the best
//! achievable per-step team return (and per-step BR-agent return) versus the
//! all-uniform baseline.
//!
//! ## What "uniform opponents" means here
//!
//! `train_br_probe` freezes `N − 1` *freshly-initialized* MLP policies as the
//! BR's opponents. A freshly-initialized softmax-over-logits policy is
//! approximately uniform over the factored action space. This oracle uses the
//! clean idealization — **exactly uniform-random** opponents — which is
//! reproducible, NN-free, and the natural ceiling reference. The conclusion
//! (flat vs improvable) transfers to the near-uniform-net case `train_br_probe`
//! actually uses.
//!
//! ## What the oracle searches over
//!
//! A single BR agent's only lever in this game is *how it fights fires*. The
//! [`FirefighterParams`] family captures that lever:
//!
//! * `scope_owned_only` — fight only round-robin-owned houses (the specialist)
//!   vs. **any** burning house (a more aggressive firefighter).
//! * `work_prob` — probability of WORKing a burning house in scope (1.0 = the
//!   specialist; < 1.0 interpolates toward REST).
//!
//! The family contains the [`crate::multi_agent::bucket_brigade_baselines`]
//! `specialist_action` policy as the `(owned_only, work_prob = 1.0)` point —
//! i.e. the literal `gap_closed_cell == 1.0` reference endpoint. Searching the
//! family therefore bounds the achievable team return from below by at least
//! the strongest hand-crafted baseline, and the randomized sweep probes the
//! firefighting-intensity axis for anything the hand-crafted points miss.
//!
//! ## Interpreting the result
//!
//! * **Ceiling ≈ uniform baseline (flat gap):** no scripted BR materially beats
//!   uniform, so the PPO BR's flat `mean_ep_return` is a property of the *game*
//!   (3 random opponents ruin the village regardless of agent 0), not a bug in
//!   the policy update. The #134 direction is exhausted at the BR level.
//! * **Ceiling materially beats uniform:** a real improvable gap exists and the
//!   PPO-update diagnosis (issue #259 AC#2) is warranted, with this ceiling as
//!   the target.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    env::games::bucket_brigade::BucketBrigadeMaEnv,
    multi_agent::bucket_brigade_baselines::specialist_action,
};

/// House state code: currently on fire (the only state a firefighter reacts
/// to). Mirrors the engine's `BURNING = 1`.
const HOUSE_BURNING: u8 = 1;
/// Action mode / signal code: do not work this night.
const MODE_REST: i64 = 0;
/// Action mode / signal code: work the targeted house this night.
const MODE_WORK: i64 = 1;

/// The single agent we compute a best-response for. The other `N − 1` agents
/// are frozen uniform-random opponents. Matches `train_br_probe`'s `BR_AGENT`.
pub const BR_AGENT: usize = 0;

/// Aggregated return statistics from evaluating one scripted BR policy against
/// frozen uniform opponents over many episodes.
#[derive(Debug, Clone, Copy)]
pub struct OracleEval {
    /// Number of episodes rolled out.
    pub episodes: usize,
    /// Total env steps across all episodes (episodes terminate when all houses
    /// are safe or all ruined, after `min_nights`).
    pub total_steps: usize,
    /// Sum over all steps of the *team* reward (sum across all `N` agents).
    pub team_return_sum: f64,
    /// Sum over all steps of the *BR agent's own* reward.
    pub br_return_sum: f64,
}

impl OracleEval {
    /// Mean per-step team return (sum across all agents, averaged over steps).
    pub fn per_step_team(&self) -> f64 {
        if self.total_steps == 0 {
            f64::NAN
        } else {
            self.team_return_sum / self.total_steps as f64
        }
    }

    /// Mean per-step BR-agent return.
    pub fn per_step_br(&self) -> f64 {
        if self.total_steps == 0 {
            f64::NAN
        } else {
            self.br_return_sum / self.total_steps as f64
        }
    }

    /// Mean per-episode team return.
    pub fn per_episode_team(&self) -> f64 {
        if self.episodes == 0 {
            f64::NAN
        } else {
            self.team_return_sum / self.episodes as f64
        }
    }

    /// Mean per-episode BR-agent return. Directly comparable to
    /// `train_br_probe`'s logged `mean_ep_return` (agent-0 summed episode
    /// return in native payoff units).
    pub fn per_episode_br(&self) -> f64 {
        if self.episodes == 0 {
            f64::NAN
        } else {
            self.br_return_sum / self.episodes as f64
        }
    }

    /// Mean episode length in env steps.
    pub fn mean_ep_len(&self) -> f64 {
        if self.episodes == 0 {
            f64::NAN
        } else {
            self.total_steps as f64 / self.episodes as f64
        }
    }
}

/// Sample a uniform-random factored action `[house, mode, signal]` over
/// `MultiDiscrete([num_houses, 2, 2])`. This is exactly one frozen opponent's
/// behavior, and also the BR-agent baseline policy.
fn uniform_action(num_houses: usize, rng: &mut StdRng) -> [i64; 3] {
    [
        rng.random_range(0..num_houses) as i64,
        rng.random_range(0..2),
        rng.random_range(0..2),
    ]
}

/// Read the global `houses` state slice out of an agent's flat observation.
///
/// The flat observation layout (see
/// `crate::env::games::bucket_brigade::flatten_observation`) places the
/// `num_houses` house-state codes at offset `1..1 + num_houses` (after the
/// leading normalized `agent_id` scalar). House codes: `0 = SAFE`,
/// `1 = BURNING`, `2 = RUINED`.
fn house_state(flat_obs: &[f32], h: usize) -> u8 {
    flat_obs[1 + h] as u8
}

/// Parameters of the scripted **firefighter** policy family the oracle
/// searches over for the BR agent.
#[derive(Debug, Clone, Copy)]
pub struct FirefighterParams {
    /// If `true`, only fight round-robin-owned houses (`h % num_agents ==
    /// agent`) — the specialist's ownership discipline. If `false`, fight the
    /// lowest-index burning house **anywhere** on the ring.
    pub scope_owned_only: bool,
    /// Probability of issuing WORK (with an honest signal) when a burning house
    /// exists in scope. `1.0` reproduces the deterministic specialist /
    /// aggressive firefighter; lower values interpolate toward REST.
    pub work_prob: f32,
}

impl FirefighterParams {
    /// Compute the firefighter action for the BR agent given its flat
    /// observation.
    ///
    /// * Find the lowest-index BURNING house in scope (owned-only or any).
    /// * With probability `work_prob`, WORK it with an honest signal (`[h,
    ///   WORK, WORK]`); otherwise REST.
    /// * If no burning house is in scope, REST on house 0.
    fn action(
        &self,
        flat_obs: &[f32],
        agent_id: usize,
        num_agents: usize,
        num_houses: usize,
        rng: &mut StdRng,
    ) -> [i64; 3] {
        let mut target: Option<usize> = None;
        for h in 0..num_houses {
            if self.scope_owned_only && h % num_agents != agent_id {
                continue;
            }
            if house_state(flat_obs, h) == HOUSE_BURNING {
                target = Some(h);
                break;
            }
        }
        match target {
            Some(h) if rng.random::<f32>() < self.work_prob => [h as i64, MODE_WORK, MODE_WORK],
            _ => [0, MODE_REST, MODE_REST],
        }
    }
}

/// A named scripted BR policy the oracle can evaluate.
enum BrPolicy {
    /// Uniform-random over `MultiDiscrete([num_houses, 2, 2])` — the baseline
    /// (identical in distribution to a frozen opponent).
    Uniform,
    /// Always REST on house 0.
    AlwaysRest,
    /// The round-robin specialist baseline (`bucket_brigade_baselines`).
    Specialist,
    /// A member of the [`FirefighterParams`] family.
    Firefighter(FirefighterParams),
}

impl BrPolicy {
    fn action(
        &self,
        flat_obs: &[f32],
        num_agents: usize,
        num_houses: usize,
        rng: &mut StdRng,
    ) -> [i64; 3] {
        match self {
            BrPolicy::Uniform => uniform_action(num_houses, rng),
            BrPolicy::AlwaysRest => [0, MODE_REST, MODE_REST],
            BrPolicy::Specialist => specialist_action(flat_obs, BR_AGENT, num_agents, num_houses),
            BrPolicy::Firefighter(p) => p.action(flat_obs, BR_AGENT, num_agents, num_houses, rng),
        }
    }
}

/// Roll out `episode_seeds.len()` episodes with the BR agent following
/// `policy` and every other agent acting uniform-random, accumulating team and
/// BR-agent return.
///
/// Each episode resets the env with the corresponding seed in `episode_seeds`,
/// so passing the **same** seed list to every candidate is a variance-reduction
/// control: all candidates face the same opponent-randomness and env-dynamics
/// stream. `step_cap` bounds pathologically long episodes (the env terminates
/// on its own once all houses are safe or ruined after `min_nights`).
fn evaluate(
    env: &mut BucketBrigadeMaEnv,
    policy: &BrPolicy,
    num_agents: usize,
    num_houses: usize,
    episode_seeds: &[u64],
    rng: &mut StdRng,
    step_cap: usize,
) -> OracleEval {
    let mut team_return_sum = 0.0_f64;
    let mut br_return_sum = 0.0_f64;
    let mut total_steps = 0_usize;

    for &seed in episode_seeds {
        let mut obs = env.reset(Some(seed));
        for _ in 0..step_cap {
            let actions: Vec<[u8; 3]> = (0..num_agents)
                .map(|a| {
                    let act = if a == BR_AGENT {
                        policy.action(&obs[a], num_agents, num_houses, rng)
                    } else {
                        uniform_action(num_houses, rng)
                    };
                    [act[0] as u8, act[1] as u8, act[2] as u8]
                })
                .collect();
            let res = env.step(&actions);
            let step_team: f64 = res.rewards.iter().map(|&r| r as f64).sum();
            team_return_sum += step_team;
            br_return_sum += res.rewards[BR_AGENT] as f64;
            total_steps += 1;
            obs = res.observations;
            if res.done {
                break;
            }
        }
    }

    OracleEval { episodes: episode_seeds.len(), total_steps, team_return_sum, br_return_sum }
}

/// One labeled row of the oracle report.
#[derive(Debug, Clone)]
pub struct OracleRow {
    /// Human-readable policy label.
    pub label: String,
    /// Evaluation statistics for this policy against frozen uniform opponents.
    pub eval: OracleEval,
}

/// Full result of an improvability-gate run on one cell.
#[derive(Debug, Clone)]
pub struct OracleReport {
    /// The all-uniform baseline row (BR agent also uniform-random).
    pub baseline: OracleRow,
    /// Every evaluated candidate (hand-crafted battery + best searched
    /// firefighter), in evaluation order.
    pub candidates: Vec<OracleRow>,
    /// Index into `candidates` of the row with the highest per-step team
    /// return (the ceiling).
    pub best_idx: usize,
}

impl OracleReport {
    /// The ceiling row: the candidate achieving the highest per-step team
    /// return.
    pub fn best(&self) -> &OracleRow {
        &self.candidates[self.best_idx]
    }

    /// Absolute improvement in per-step team return of the ceiling over the
    /// all-uniform baseline. Positive ⇒ a scripted BR beats uniform.
    pub fn team_gap_per_step(&self) -> f64 {
        self.best().eval.per_step_team() - self.baseline.eval.per_step_team()
    }

    /// Ceiling's per-step team improvement as a fraction of the magnitude of
    /// the baseline per-step team return. A small fraction ⇒ a "flat" gap.
    ///
    /// Using `|baseline|` as the denominator makes this a scale-free,
    /// cell-agnostic measure of how much head-room a single BR agent has.
    pub fn team_gap_fraction(&self) -> f64 {
        let base = self.baseline.eval.per_step_team();
        if base == 0.0 {
            return f64::NAN;
        }
        self.team_gap_per_step() / base.abs()
    }
}

/// Run the full improvability-gate oracle on one cell.
///
/// Builds the candidate battery (uniform baseline, always-rest, specialist,
/// owned-only and any-house deterministic firefighters), runs a randomized
/// search over [`FirefighterParams`] (`num_search` candidates, each scored on
/// `search_episodes` episodes), then re-scores the baseline and every
/// hand-crafted candidate plus the best searched firefighter on the full
/// `eval_episodes` episode set for an apples-to-apples comparison.
///
/// All candidates are evaluated against the **same** per-episode seed stream
/// (variance reduction). The opponent / stochastic-policy RNG is reseeded from
/// `seed` before each candidate so candidates differ only in the BR policy.
///
/// # Arguments
///
/// * `env` — a constructed cell env (the caller fixes `(β, κ, c)`).
/// * `num_agents`, `num_houses` — env topology.
/// * `eval_episodes` — episodes used for the final reported numbers.
/// * `search_episodes` — episodes used to score each searched firefighter.
/// * `num_search` — number of random firefighters to sample.
/// * `seed` — base RNG seed for opponents, stochastic policies, and the search.
/// * `step_cap` — per-episode step bound.
#[allow(clippy::too_many_arguments)]
pub fn run_oracle(
    env: &mut BucketBrigadeMaEnv,
    num_agents: usize,
    num_houses: usize,
    eval_episodes: usize,
    search_episodes: usize,
    num_search: usize,
    seed: u64,
    step_cap: usize,
) -> OracleReport {
    // Shared per-episode seed streams (variance reduction across candidates).
    let eval_seeds: Vec<u64> = (0..eval_episodes as u64).map(|i| seed ^ (0x9E3779B9 ^ i)).collect();
    let search_seeds: Vec<u64> =
        (0..search_episodes as u64).map(|i| seed ^ (0x85EBCA6B ^ i)).collect();

    // Score a policy on the eval seed set with a freshly seeded RNG so
    // candidates are compared under identical opponent randomness. A free
    // function (not a closure) so it does not hold a long-lived borrow of
    // `env` across the search loop below.
    fn score_eval(
        env: &mut BucketBrigadeMaEnv,
        policy: &BrPolicy,
        num_agents: usize,
        num_houses: usize,
        eval_seeds: &[u64],
        seed: u64,
        step_cap: usize,
    ) -> OracleEval {
        let mut rng = StdRng::seed_from_u64(seed);
        evaluate(env, policy, num_agents, num_houses, eval_seeds, &mut rng, step_cap)
    }
    macro_rules! score {
        ($policy:expr) => {
            score_eval(env, &$policy, num_agents, num_houses, &eval_seeds, seed, step_cap)
        };
    }

    // --- Baseline: BR agent uniform-random (all four agents uniform). ---
    let baseline =
        OracleRow { label: "uniform (baseline)".to_string(), eval: score!(BrPolicy::Uniform) };

    // --- Hand-crafted battery. ---
    let mut candidates: Vec<OracleRow> =
        vec![OracleRow { label: "uniform".to_string(), eval: baseline.eval }];
    candidates
        .push(OracleRow { label: "always_rest".to_string(), eval: score!(BrPolicy::AlwaysRest) });
    candidates
        .push(OracleRow { label: "specialist".to_string(), eval: score!(BrPolicy::Specialist) });
    candidates.push(OracleRow {
        label: "firefighter[owned, work=1.0]".to_string(),
        eval: score!(BrPolicy::Firefighter(FirefighterParams {
            scope_owned_only: true,
            work_prob: 1.0,
        })),
    });
    candidates.push(OracleRow {
        label: "firefighter[any, work=1.0]".to_string(),
        eval: score!(BrPolicy::Firefighter(FirefighterParams {
            scope_owned_only: false,
            work_prob: 1.0,
        })),
    });

    // --- Randomized search over the firefighter family. ---
    // Cheap CEM-lite: sample `num_search` random firefighters, score each on
    // the (smaller) search seed set, and keep the one with the best per-step
    // team return. This probes the firefighting-intensity axis (work_prob) and
    // ownership scope for anything the hand-crafted points miss.
    let mut search_rng = StdRng::seed_from_u64(seed ^ 0xD1B54A32);
    let mut best_params: Option<FirefighterParams> = None;
    let mut best_search_team = f64::NEG_INFINITY;
    for _ in 0..num_search {
        let params = FirefighterParams {
            scope_owned_only: search_rng.random::<bool>(),
            work_prob: search_rng.random::<f32>(),
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let eval = evaluate(
            env,
            &BrPolicy::Firefighter(params),
            num_agents,
            num_houses,
            &search_seeds,
            &mut rng,
            step_cap,
        );
        if eval.per_step_team() > best_search_team {
            best_search_team = eval.per_step_team();
            best_params = Some(params);
        }
    }
    if let Some(params) = best_params {
        candidates.push(OracleRow {
            label: format!(
                "search_best firefighter[{}, work={:.3}]",
                if params.scope_owned_only {
                    "owned"
                } else {
                    "any"
                },
                params.work_prob
            ),
            eval: score!(BrPolicy::Firefighter(params)),
        });
    }

    // Ceiling = candidate with the highest per-step team return.
    let best_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.eval.per_step_team().partial_cmp(&b.eval.per_step_team()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    OracleReport { baseline, candidates, best_idx }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::games::bucket_brigade::{NUM_HOUSES, registry},
        multi_agent::bucket_brigade_baselines::BucketBrigadeCell,
    };

    fn make_cell_env(cell: BucketBrigadeCell, num_agents: usize, seed: u64) -> BucketBrigadeMaEnv {
        let (beta, kappa, cost) = cell.parameters();
        let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
            .expect("minimal_specialization-v1 must resolve in the registry");
        scenario.prob_fire_spreads_to_neighbor = beta;
        scenario.prob_solo_agent_extinguishes_fire = kappa;
        scenario.cost_to_work_one_night = cost;
        BucketBrigadeMaEnv::new(scenario, num_agents, Some(seed))
    }

    /// The oracle runs end-to-end on the canonical cell and produces finite,
    /// sane statistics: negative per-step team return (the env is a penalty
    /// landscape), positive episode lengths, and a ceiling that is at least as
    /// good as the uniform baseline (the baseline is itself a candidate).
    #[test]
    fn oracle_runs_and_is_sane() {
        let mut env = make_cell_env(BucketBrigadeCell::Beta05, 4, 42);
        let report = run_oracle(
            &mut env, 4, NUM_HOUSES, // eval_episodes
            30, // search_episodes
            10, // num_search
            8, // seed
            42, // step_cap
            500,
        );

        assert!(report.baseline.eval.per_step_team().is_finite());
        assert!(report.baseline.eval.per_step_team() < 0.0, "env is a penalty landscape");
        assert!(report.baseline.eval.mean_ep_len() > 0.0);
        for row in &report.candidates {
            assert!(row.eval.per_step_team().is_finite(), "candidate {} non-finite", row.label);
            assert!(row.eval.episodes == 30);
        }
        // The ceiling can never be worse than uniform: uniform is in the set.
        assert!(report.team_gap_per_step() >= -1e-9, "ceiling must be >= baseline");
    }

    /// The specialist endpoint of the firefighter family is reachable and the
    /// `FirefighterParams { owned, work=1.0 }` member matches
    /// `specialist_action` on a hand-built observation with one owned house
    /// burning.
    #[test]
    fn firefighter_owned_matches_specialist_on_burning_owned() {
        let mut houses = vec![0u8; NUM_HOUSES];
        houses[4] = HOUSE_BURNING; // agent 0 owns house 4 (4 % 4 == 0)
        let mut flat = vec![0.0f32; 1 + NUM_HOUSES + 64];
        for (i, &h) in houses.iter().enumerate() {
            flat[1 + i] = h as f32;
        }
        let mut rng = StdRng::seed_from_u64(0);
        let params = FirefighterParams { scope_owned_only: true, work_prob: 1.0 };
        let ff = params.action(&flat, 0, 4, NUM_HOUSES, &mut rng);
        let spec = specialist_action(&flat, 0, 4, NUM_HOUSES);
        assert_eq!(
            ff, spec,
            "owned work=1.0 firefighter must equal specialist when owned house burns"
        );
        assert_eq!(ff, [4, MODE_WORK, MODE_WORK]);
    }
}
