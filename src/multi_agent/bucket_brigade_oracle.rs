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
//! [`FirefighterParams`](crate::multi_agent::bucket_brigade_oracle::FirefighterParams)
//! family captures that lever:
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
use serde::Serialize;

use crate::{
    env::games::bucket_brigade::{BucketBrigadeMaEnv, registry},
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
///
/// Cloneable so a single policy can be replicated across the agents of a
/// coalition (see [`run_coalition_oracle`]).
#[derive(Debug, Clone, Copy)]
pub enum BrPolicy {
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
    /// Compute the action for `agent_id` under this policy given its flat
    /// observation. `agent_id` is threaded through so ownership-scoped policies
    /// (`Specialist`, owned-only `Firefighter`) resolve the correct
    /// round-robin-owned houses for *this* deviator — essential once a
    /// coalition scripts more than one agent.
    fn action(
        &self,
        flat_obs: &[f32],
        agent_id: usize,
        num_agents: usize,
        num_houses: usize,
        rng: &mut StdRng,
    ) -> [i64; 3] {
        match self {
            BrPolicy::Uniform => uniform_action(num_houses, rng),
            BrPolicy::AlwaysRest => [0, MODE_REST, MODE_REST],
            BrPolicy::Specialist => specialist_action(flat_obs, agent_id, num_agents, num_houses),
            BrPolicy::Firefighter(p) => p.action(flat_obs, agent_id, num_agents, num_houses, rng),
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
    // The k=1 improvability gate is the coalition oracle with a single
    // deviator (agent 0). Delegate so the two paths never drift apart.
    let coalition = [(BR_AGENT, *policy)];
    evaluate_coalition(env, &coalition, num_agents, num_houses, episode_seeds, rng, step_cap).eval
}

/// Return statistics for a coalition evaluation, carrying the **per-episode
/// per-step** team-return series in addition to the aggregate [`OracleEval`].
///
/// The per-episode series is the length-normalized statistic the episode-level
/// bootstrap CI on the team-return gap (issue #268) resamples. Length
/// normalization (mean reward *per step* within each episode) matters: raw
/// per-episode totals are dominated by episode-length variance (episodes run
/// 10–1000 steps), which swamps the per-step team-return signal #259 measured.
/// Resampling per-step means over episodes keeps the CI comparable to the #259
/// per-step numbers while still being an episode-level bootstrap.
#[derive(Debug, Clone)]
pub struct CoalitionEval {
    /// Aggregate statistics (per-step / per-episode means).
    pub eval: OracleEval,
    /// Mean *per-step* team return (summed across all `N` agents, averaged over
    /// that episode's steps) for each episode, in `episode_seeds` order.
    /// Length == `episode_seeds.len()`. Empty-episode (zero-step) entries are
    /// impossible: the env always runs `min_nights` before it can terminate.
    pub per_episode_team_per_step: Vec<f64>,
}

/// Roll out `episode_seeds.len()` episodes where every `(agent_id, policy)` in
/// `coalition` follows its assigned scripted policy and every remaining agent
/// acts uniform-random, accumulating team and coalition-return statistics.
///
/// This is the k≥1 generalization of `evaluate`: with a one-element coalition
/// `[(0, policy)]` it reproduces the original single-BR improvability gate;
/// with `k` elements it scripts `k` coordinated deviators against `N−k` frozen
/// uniform opponents (issue #268).
///
/// The reported `br_return_sum` in the returned [`OracleEval`] is the summed
/// **own** return of the coalition members (the quantity a coordinated method
/// would optimize). Each episode resets the env with the corresponding seed in
/// `episode_seeds`, so passing the **same** seed list to the baseline and to
/// every candidate is the variance-reduction control that makes the per-episode
/// gap series a paired comparison.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_coalition(
    env: &mut BucketBrigadeMaEnv,
    coalition: &[(usize, BrPolicy)],
    num_agents: usize,
    num_houses: usize,
    episode_seeds: &[u64],
    rng: &mut StdRng,
    step_cap: usize,
) -> CoalitionEval {
    let mut team_return_sum = 0.0_f64;
    let mut br_return_sum = 0.0_f64;
    let mut total_steps = 0_usize;
    let mut per_episode_team_per_step = Vec::with_capacity(episode_seeds.len());

    for &seed in episode_seeds {
        let mut obs = env.reset(Some(seed));
        let mut ep_team = 0.0_f64;
        let mut ep_steps = 0_usize;
        for _ in 0..step_cap {
            let actions: Vec<[u8; 3]> = (0..num_agents)
                .map(|a| {
                    let act = match coalition.iter().find(|(id, _)| *id == a) {
                        Some((id, policy)) => {
                            policy.action(&obs[a], *id, num_agents, num_houses, rng)
                        }
                        None => uniform_action(num_houses, rng),
                    };
                    [act[0] as u8, act[1] as u8, act[2] as u8]
                })
                .collect();
            let res = env.step(&actions);
            let step_team: f64 = res.rewards.iter().map(|&r| r as f64).sum();
            team_return_sum += step_team;
            ep_team += step_team;
            for (id, _) in coalition {
                br_return_sum += res.rewards[*id] as f64;
            }
            total_steps += 1;
            ep_steps += 1;
            obs = res.observations;
            if res.done {
                break;
            }
        }
        // Length-normalized per-step team return for this episode (see
        // `CoalitionEval` docs for why totals are unusable). `ep_steps` is
        // always >= 1 here (the env runs at least one step before `done`).
        per_episode_team_per_step.push(if ep_steps == 0 {
            0.0
        } else {
            ep_team / ep_steps as f64
        });
    }

    CoalitionEval {
        eval: OracleEval {
            episodes: episode_seeds.len(),
            total_steps,
            team_return_sum,
            br_return_sum,
        },
        per_episode_team_per_step,
    }
}

/// Percentile bootstrap 95%-style CI on the mean of a per-episode series.
///
/// Resamples `values` with replacement `n_boot` times, computes the mean of
/// each resample, and returns the `(alpha/2, 1−alpha/2)` empirical quantiles of
/// the bootstrap distribution of the mean. Self-contained (no external crate),
/// mirroring the episode-level resampling the conditional-entropy pipeline uses
/// elsewhere in bucket-brigade.
///
/// Returns `(NaN, NaN)` for an empty input. For `alpha = 0.05` the interval is
/// the 2.5th–97.5th percentile of resampled means. A lower bound strictly above
/// zero on the *gap* series is the decision rule for `k*` (issue #268).
pub fn bootstrap_mean_ci(
    values: &[f64],
    n_boot: usize,
    alpha: f64,
    rng: &mut StdRng,
) -> (f64, f64) {
    let n = values.len();
    if n == 0 || n_boot == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut means: Vec<f64> = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut acc = 0.0_f64;
        for _ in 0..n {
            let idx = rng.random_range(0..n);
            acc += values[idx];
        }
        means.push(acc / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let quantile = |q: f64| -> f64 {
        // Nearest-rank quantile on the sorted bootstrap means.
        let rank = (q * (n_boot as f64 - 1.0)).round() as usize;
        means[rank.min(n_boot - 1)]
    };
    (quantile(lo_q), quantile(hi_q))
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

/// One labeled coalition-candidate row: a policy assignment for the `k`
/// deviators plus its evaluation against the frozen uniform remainder.
#[derive(Debug, Clone)]
pub struct CoalitionRow {
    /// Human-readable coalition label (e.g. `"all_specialist"`).
    pub label: String,
    /// Aggregate evaluation statistics for this coalition.
    pub eval: OracleEval,
    /// Per-episode *per-step* team return, shared-seed-aligned with the
    /// baseline so `candidate − baseline` is a paired per-episode gap
    /// series.
    pub per_episode_team_per_step: Vec<f64>,
}

/// Full result of a coalition improvability-gate run on one cell for a fixed
/// coalition size `k` (issue #268).
#[derive(Debug, Clone)]
pub struct CoalitionOracleReport {
    /// Coalition size (number of scripted deviators).
    pub k: usize,
    /// The all-uniform baseline row (no deviators).
    pub baseline: CoalitionRow,
    /// Every evaluated coalition assignment, in evaluation order.
    pub candidates: Vec<CoalitionRow>,
    /// Index into `candidates` of the highest per-episode team return (the
    /// ceiling for this `k`).
    pub best_idx: usize,
    /// Point estimate the CI brackets: the **episode-mean** of the per-episode
    /// per-step team-return gap (ceiling − baseline), i.e. every episode
    /// weighted equally regardless of length. This is the statistic the
    /// episode-level bootstrap resamples, so it — not the step-weighted
    /// [`CoalitionOracleReport::team_gap_per_step`] — is the number that lies
    /// inside `[gap_ci_lo, gap_ci_hi]`.
    pub gap_mean: f64,
    /// Bootstrap CI lower bound on `gap_mean` (episode-level resampling).
    pub gap_ci_lo: f64,
    /// Bootstrap CI upper bound on `gap_mean` (episode-level resampling).
    pub gap_ci_hi: f64,
}

impl CoalitionOracleReport {
    /// The ceiling row: the coalition achieving the highest per-step team
    /// return at this `k`.
    pub fn best(&self) -> &CoalitionRow {
        &self.candidates[self.best_idx]
    }

    /// Absolute per-step team-return improvement of the ceiling over the
    /// all-uniform baseline. This is the quantity the bootstrap CI brackets.
    pub fn team_gap_per_step(&self) -> f64 {
        self.best().eval.per_step_team() - self.baseline.eval.per_step_team()
    }

    /// Ceiling's per-step team improvement as a fraction of `|baseline|` — a
    /// scale-free "how flat is it" measure. Small ⇒ flat.
    pub fn team_gap_fraction(&self) -> f64 {
        let base = self.baseline.eval.per_step_team();
        if base == 0.0 {
            return f64::NAN;
        }
        self.team_gap_per_step() / base.abs()
    }

    /// Whether this `k` clears the coordination threshold: a *statistically
    /// positive* team-return gap, i.e. the bootstrap CI lower bound is strictly
    /// above zero. `k*` for a cell is the smallest `k` for which this holds.
    pub fn is_k_star(&self) -> bool {
        self.gap_ci_lo > 0.0
    }
}

/// Build the coalition-candidate battery for `k` deviators (agents `0..k`).
///
/// Homogeneous assignments replicate one policy across all `k` deviators;
/// heterogeneous assignments (only meaningful for `k ≥ 2`) mix roles to match
/// the heterogeneous double-oracle profile shapes called out in issue #268
/// (one aggressive "Hero" firefighter, or one abstainer, plus specialists).
/// `search_best` is the best homogeneous firefighter found by the random
/// search.
fn coalition_candidates(k: usize, search_best: FirefighterParams) -> Vec<(String, Vec<BrPolicy>)> {
    let homogeneous = |label: &str, p: BrPolicy| (label.to_string(), vec![p; k]);
    let ff = |owned: bool| {
        BrPolicy::Firefighter(FirefighterParams { scope_owned_only: owned, work_prob: 1.0 })
    };

    let mut out = vec![
        homogeneous("all_always_rest", BrPolicy::AlwaysRest),
        homogeneous("all_specialist", BrPolicy::Specialist),
        homogeneous("all_firefighter[owned,work=1.0]", ff(true)),
        homogeneous("all_firefighter[any,work=1.0]", ff(false)),
        homogeneous(
            &format!(
                "all_search_best[{},work={:.3}]",
                if search_best.scope_owned_only {
                    "owned"
                } else {
                    "any"
                },
                search_best.work_prob
            ),
            BrPolicy::Firefighter(search_best),
        ),
    ];

    if k >= 2 {
        // One aggressive any-house "Hero" firefighter + (k−1) owned specialists.
        let mut hero_plus_specialists = vec![ff(false)];
        hero_plus_specialists.extend(std::iter::repeat_n(BrPolicy::Specialist, k - 1));
        out.push(("hero[any]+specialists".to_string(), hero_plus_specialists));

        // One abstainer (always-rest) + (k−1) owned specialists.
        let mut rest_plus_specialists = vec![BrPolicy::AlwaysRest];
        rest_plus_specialists.extend(std::iter::repeat_n(BrPolicy::Specialist, k - 1));
        out.push(("rest+specialists".to_string(), rest_plus_specialists));
    }

    out
}

/// Run the coalition improvability-gate oracle on one cell for a fixed
/// coalition size `k` (issue #268).
///
/// Freezes `N−k` uniform opponents and scripts the first `k` agents as
/// coordinated deviators. Evaluates the coalition-candidate battery (see
/// `coalition_candidates`) against the **same** per-episode seed stream as
/// the all-uniform baseline, then reports the ceiling gap and an episode-level
/// percentile bootstrap CI on the per-episode team-return gap. `k = 1`
/// reproduces the [`run_oracle`] single-BR gate.
///
/// # Arguments
///
/// * `k` — coalition size (`1..=num_agents`).
/// * `eval_episodes` / `search_episodes` / `num_search` — as in [`run_oracle`].
/// * `n_boot` — bootstrap resamples for the gap CI (e.g. 1000).
/// * `alpha` — CI significance level (e.g. 0.05 for a 95% CI).
#[allow(clippy::too_many_arguments)]
pub fn run_coalition_oracle(
    env: &mut BucketBrigadeMaEnv,
    num_agents: usize,
    num_houses: usize,
    k: usize,
    eval_episodes: usize,
    search_episodes: usize,
    num_search: usize,
    seed: u64,
    step_cap: usize,
    n_boot: usize,
    alpha: f64,
) -> CoalitionOracleReport {
    assert!(k >= 1 && k <= num_agents, "coalition size k={k} out of range 1..={num_agents}");

    // Shared per-episode seed streams (variance reduction across candidates and
    // the baseline). Identical derivation to `run_oracle` so k=1 lines up.
    let eval_seeds: Vec<u64> = (0..eval_episodes as u64).map(|i| seed ^ (0x9E3779B9 ^ i)).collect();
    let search_seeds: Vec<u64> =
        (0..search_episodes as u64).map(|i| seed ^ (0x85EBCA6B ^ i)).collect();

    // Assign a policy set to agents 0..k. Each candidate reseeds the RNG from
    // `seed` so candidates differ only in the coalition policy, not opponent
    // randomness.
    let assign = |policies: &[BrPolicy]| -> Vec<(usize, BrPolicy)> {
        policies.iter().enumerate().map(|(a, p)| (a, *p)).collect()
    };
    let score = |env: &mut BucketBrigadeMaEnv, policies: &[BrPolicy]| -> CoalitionEval {
        let coalition = assign(policies);
        let mut rng = StdRng::seed_from_u64(seed);
        evaluate_coalition(env, &coalition, num_agents, num_houses, &eval_seeds, &mut rng, step_cap)
    };

    // --- Baseline: all agents uniform (empty coalition). ---
    let baseline_eval = {
        let mut rng = StdRng::seed_from_u64(seed);
        evaluate_coalition(env, &[], num_agents, num_houses, &eval_seeds, &mut rng, step_cap)
    };
    let baseline = CoalitionRow {
        label: "all_uniform (baseline)".to_string(),
        eval: baseline_eval.eval,
        per_episode_team_per_step: baseline_eval.per_episode_team_per_step,
    };

    // --- Randomized search over the firefighter family (homogeneous over k). ---
    let mut search_rng = StdRng::seed_from_u64(seed ^ 0xD1B54A32);
    let mut best_params = FirefighterParams { scope_owned_only: true, work_prob: 1.0 };
    let mut best_search_team = f64::NEG_INFINITY;
    for _ in 0..num_search {
        let params = FirefighterParams {
            scope_owned_only: search_rng.random::<bool>(),
            work_prob: search_rng.random::<f32>(),
        };
        let coalition = assign(&vec![BrPolicy::Firefighter(params); k]);
        let mut rng = StdRng::seed_from_u64(seed);
        let eval = evaluate_coalition(
            env,
            &coalition,
            num_agents,
            num_houses,
            &search_seeds,
            &mut rng,
            step_cap,
        );
        if eval.eval.per_step_team() > best_search_team {
            best_search_team = eval.eval.per_step_team();
            best_params = params;
        }
    }

    // --- Score the full candidate battery on the eval seed set. ---
    let mut candidates: Vec<CoalitionRow> = Vec::new();
    for (label, policies) in coalition_candidates(k, best_params) {
        let ev = score(env, &policies);
        candidates.push(CoalitionRow {
            label,
            eval: ev.eval,
            per_episode_team_per_step: ev.per_episode_team_per_step,
        });
    }

    // Ceiling = candidate with the highest (aggregate) per-step team return.
    let best_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.eval.per_step_team().partial_cmp(&b.eval.per_step_team()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Episode-level paired per-step gap series (ceiling − baseline) → bootstrap CI.
    let gap_series: Vec<f64> = candidates[best_idx]
        .per_episode_team_per_step
        .iter()
        .zip(baseline.per_episode_team_per_step.iter())
        .map(|(c, b)| c - b)
        .collect();
    let gap_mean = if gap_series.is_empty() {
        f64::NAN
    } else {
        gap_series.iter().sum::<f64>() / gap_series.len() as f64
    };
    let mut boot_rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A);
    let (gap_ci_lo, gap_ci_hi) = bootstrap_mean_ci(&gap_series, n_boot, alpha, &mut boot_rng);

    CoalitionOracleReport { k, baseline, candidates, best_idx, gap_mean, gap_ci_lo, gap_ci_hi }
}

// ===========================================================================
// Phase-diagram sweep (issue #269)
// ===========================================================================
//
// The coalition oracle above measures k* on a *single* (β, κ, c) cell. Issue
// #269 sweeps k* across the full (β, κ, c) phase-diagram grid (the same grid
// `envs/bucket-brigade/experiments/scripts/compute_nash_phase_diagram.py`
// defines). The types and `run_phase_cell` helper below are the per-cell unit
// of that sweep: they own env construction from raw floats (the raw-float
// generalization of the `BucketBrigadeCell`-keyed constructor) and emit a
// serializable per-cell record so a driver can parallelize over cells (rayon)
// and write one JSON file per cell.

/// One `k`-slice of a phase-diagram cell sweep: the coalition-oracle summary
/// for a fixed coalition size `k` on one cell.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseKRecord {
    /// Coalition size (number of scripted deviators).
    pub k: usize,
    /// Episode-mean per-step team-return gap (ceiling − baseline). The point
    /// estimate the bootstrap CI brackets.
    pub gap_mean: f64,
    /// Bootstrap CI lower bound on `gap_mean`.
    pub gap_ci_lo: f64,
    /// Bootstrap CI upper bound on `gap_mean`.
    pub gap_ci_hi: f64,
    /// Step-weighted per-step team-return gap of the ceiling over the
    /// all-uniform baseline (the aggregate companion to `gap_mean`).
    pub team_gap_per_step: f64,
    /// Label of the ceiling coalition assignment at this `k`.
    pub best_coalition_policy: String,
    /// Whether this `k` clears the coordination threshold (CI lower bound
    /// strictly above zero).
    pub is_k_star: bool,
}

/// One phase-diagram cell record: the `k = 1..=k_max` coalition sweep for a
/// fixed `(β, κ, c)` triple plus the derived per-cell `k*`.
///
/// `cell_tag` uses the canonical `b{β:.2}_k{κ:.2}_c{c:.2}` format shared with
/// `compute_nash_phase_diagram.py::_cell_tag` and `BucketBrigadeCell::tag`, so
/// records join against the downstream verdict / entropy artifacts by tag.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseCellRecord {
    /// Canonical `b{β:.2}_k{κ:.2}_c{c:.2}` join key.
    pub cell_tag: String,
    /// Fire-spread probability β (`prob_fire_spreads_to_neighbor`).
    pub beta: f32,
    /// Single-agent extinguish probability κ
    /// (`prob_solo_agent_extinguishes_fire`).
    pub kappa: f32,
    /// Work cost c (`cost_to_work_one_night`).
    pub c: f32,
    /// Smallest `k` whose gap CI lower bound is strictly positive, or `None`
    /// if no `k <= k_max` clears zero (flat / near-degenerate cell).
    pub k_star: Option<usize>,
    /// Per-`k` coalition-oracle summaries, `k = 1..=k_max` in order.
    pub per_k: Vec<PhaseKRecord>,
}

/// Filesystem-safe cell tag: `b{β:.2}_k{κ:.2}_c{c:.2}`.
///
/// Byte-for-byte identical to `compute_nash_phase_diagram.py::_cell_tag` so the
/// Rust k* records and the Python Nash verdicts key against the same string.
pub fn phase_cell_tag(beta: f32, kappa: f32, c: f32) -> String {
    format!("b{beta:.2}_k{kappa:.2}_c{c:.2}")
}

/// Construct the bucket-brigade multi-agent env for a raw `(β, κ, c)` cell.
///
/// The raw-float generalization of the `BucketBrigadeCell`-keyed constructors
/// in `br_oracle.rs` / `train_br_probe.rs`: starts from the
/// `minimal_specialization-v1` base scenario (the family the asymmetric-NE
/// search was calibrated on) and overrides only the three swept fields, exactly
/// as the Python `_make_scenario` does.
pub fn make_phase_cell_env(
    beta: f32,
    kappa: f32,
    c: f32,
    num_agents: usize,
    seed: u64,
) -> BucketBrigadeMaEnv {
    let mut scenario = registry::get_scenario_by_id("minimal_specialization-v1")
        .expect("minimal_specialization-v1 must resolve in the registry");
    scenario.prob_fire_spreads_to_neighbor = beta;
    scenario.prob_solo_agent_extinguishes_fire = kappa;
    scenario.cost_to_work_one_night = c;
    BucketBrigadeMaEnv::new(scenario, num_agents, Some(seed))
}

/// Run the full `k = 1..=k_max` coalition sweep for one `(β, κ, c)` cell and
/// return its serializable [`PhaseCellRecord`] (issue #269).
///
/// Rebuilds a fresh env per `k` (identical to the single-cell sweep in
/// `br_oracle.rs`) so every `k` sees the same env-construction seed, and reuses
/// the shared per-episode seed stream inside [`run_coalition_oracle`] — the
/// same measurement protocol the #268 gate used. Self-contained (owns env
/// construction) so a driver can call it under `rayon::par_iter` with one cell
/// per task.
#[allow(clippy::too_many_arguments)]
pub fn run_phase_cell(
    beta: f32,
    kappa: f32,
    c: f32,
    num_agents: usize,
    num_houses: usize,
    k_max: usize,
    eval_episodes: usize,
    search_episodes: usize,
    num_search: usize,
    seed: u64,
    step_cap: usize,
    n_boot: usize,
    alpha: f64,
) -> PhaseCellRecord {
    let mut per_k = Vec::with_capacity(k_max);
    let mut k_star: Option<usize> = None;
    for k in 1..=k_max {
        let mut env = make_phase_cell_env(beta, kappa, c, num_agents, seed);
        let report = run_coalition_oracle(
            &mut env,
            num_agents,
            num_houses,
            k,
            eval_episodes,
            search_episodes,
            num_search,
            seed,
            step_cap,
            n_boot,
            alpha,
        );
        let is_star = report.is_k_star();
        if is_star && k_star.is_none() {
            k_star = Some(k);
        }
        per_k.push(PhaseKRecord {
            k,
            gap_mean: report.gap_mean,
            gap_ci_lo: report.gap_ci_lo,
            gap_ci_hi: report.gap_ci_hi,
            team_gap_per_step: report.team_gap_per_step(),
            best_coalition_policy: report.best().label.clone(),
            is_k_star: is_star,
        });
    }
    PhaseCellRecord { cell_tag: phase_cell_tag(beta, kappa, c), beta, kappa, c, k_star, per_k }
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
        let num_agents = 4;
        let eval_episodes = 30;
        let search_episodes = 10;
        let num_search = 8;
        let seed = 42;
        let step_cap = 500;
        let report = run_oracle(
            &mut env,
            num_agents,
            NUM_HOUSES,
            eval_episodes,
            search_episodes,
            num_search,
            seed,
            step_cap,
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
        let mut houses = [0u8; NUM_HOUSES];
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
