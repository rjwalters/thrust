//! Bucket-brigade reference baseline policies.
//!
//! Currently exposes the **specialist** policy (round-robin ownership,
//! fight burning-owned houses, REST otherwise, honest signal). This is a
//! direct port of `envs/bucket-brigade/bucket_brigade/baselines/specialist.py`
//! and is parameter-agnostic — it does NOT reference β, κ, or c at all, so
//! the same function works for every scenario (base + the three
//! no-convergence cells of the workshop-paper phase diagram). Only the
//! resulting per-step team reward differs per cell because the env dynamics
//! differ.
//!
//! The specialist baseline is used by
//! [`crate::multi_agent::bucket_brigade_metrics::gap_closed_cell`] as the `1.0`
//! reference endpoint per cell; see the `#[ignore]`-gated
//! `recompute_cell_baselines` test in
//! `tests/test_bucket_brigade_baselines.rs` for the Monte-Carlo procedure
//! that produces the cell-specific `MINSPEC_SPECIALIST_BETA0XX` constants.
//!
//! # Observation-layout coupling
//!
//! The specialist policy reads the `houses` slice of the per-agent
//! observation. That slice lives at offset `1..1+num_houses` in the flat
//! observation produced by the bucket-brigade env's `flatten_observation`
//! helper (private to `crate::env::games::bucket_brigade::env`):
//!
//! ```text
//! [agent_id_norm(1), houses(num_houses), signals(N), locations(N),
//!  last_actions(3*N), round1_signals(N), scenario_info(12)]
//! ```
//!
//! House state codes match the Python module's mirror of
//! `BucketBrigadeEnv`: `0 = SAFE, 1 = BURNING, 2 = RUINED`. The
//! specialist only checks for `BURNING`.

use crate::env::games::bucket_brigade::NUM_HOUSES;

/// House state code: house has burned to the ground (cannot be saved).
#[allow(dead_code)]
const HOUSE_RUINED: u8 = 2;
/// House state code: currently on fire (the only state the specialist
/// reacts to).
const HOUSE_BURNING: u8 = 1;
/// House state code: not on fire and not ruined. The specialist does
/// nothing in this state on owned houses (REST).
#[allow(dead_code)]
const HOUSE_SAFE: u8 = 0;

/// Action mode code: do not work this night.
const MODE_REST: i64 = 0;
/// Action mode code: work the targeted house this night.
const MODE_WORK: i64 = 1;

/// Compute the specialist action for a single agent given the flattened
/// per-agent observation.
///
/// Mirrors `bucket_brigade.baselines.specialist.specialist_action` (Python),
/// adapted to read the `houses` slice out of Thrust's flat observation
/// layout instead of the Python observation dict.
///
/// # Policy
///
/// Per agent `i`:
///
/// 1. `owned_houses = {h : h % num_agents == i}` (round-robin).
/// 2. `burning_owned = owned where houses[h] == BURNING`.
/// 3. If `burning_owned` is non-empty: action = `[min(burning_owned), WORK,
///    WORK]` (honest signal).
/// 4. Otherwise: action = `[min(owned_houses), REST, REST]`.
///
/// # Parameters
///
/// * `flat_obs` — the agent's flattened observation (length `obs_dim`). Only
///   the `houses` slice at offset `1..1+num_houses` is consulted.
/// * `agent_id` — which agent (0-indexed) we are computing the action for.
/// * `num_agents` — total number of agents in the scenario.
/// * `num_houses` — number of houses on the ring (typically [`NUM_HOUSES`] =
///   10).
///
/// # Returns
///
/// Length-3 `[house, mode, signal]` action vector matching
/// `MultiDiscrete([num_houses, 2, 2])`.
///
/// # Panics
///
/// Panics if `flat_obs.len() < 1 + num_houses` (the houses slice would
/// be out of bounds). Doesn't panic on degenerate ownership (no owned
/// houses) because round-robin guarantees at least one owned house when
/// `num_agents <= num_houses`; if the precondition is violated this
/// function falls back to `[0, REST, REST]` as a deterministic default,
/// matching the Python implementation's defensive branch.
pub fn specialist_action(
    flat_obs: &[f32],
    agent_id: usize,
    num_agents: usize,
    num_houses: usize,
) -> [i64; 3] {
    assert!(
        flat_obs.len() > num_houses,
        "flat_obs (len {}) too short for num_houses = {} (need at least {} entries)",
        flat_obs.len(),
        num_houses,
        1 + num_houses,
    );
    assert!(
        agent_id < num_agents,
        "agent_id {agent_id} out of range for num_agents {num_agents}"
    );

    // Round-robin ownership: agent `i` owns houses `h` where `h % num_agents == i`.
    // Iterate ascending so `min(burning_owned)` and `min(owned)` come out by
    // first-match.
    let mut first_owned: Option<usize> = None;
    let mut first_burning_owned: Option<usize> = None;
    for h in 0..num_houses {
        if h % num_agents == agent_id {
            if first_owned.is_none() {
                first_owned = Some(h);
            }
            // `houses` lives at offset 1 (after the agent_id_norm leading
            // scalar). The values are u8 state codes round-tripped through
            // f32 in `flatten_observation`.
            let state = flat_obs[1 + h] as u8;
            if state == HOUSE_BURNING && first_burning_owned.is_none() {
                first_burning_owned = Some(h);
                // We could break here, but iterating to the end is
                // O(num_houses) and avoids a `.find_map`
                // allocation; both are fine.
            }
        }
    }

    if let Some(h) = first_burning_owned {
        // Fight the fire on this owned house. Honest signal: signal == mode.
        return [h as i64, MODE_WORK, MODE_WORK];
    }
    if let Some(h) = first_owned {
        // No owned house is burning — rest. The house index is irrelevant to
        // env dynamics under REST, but we return an owned-house index for
        // determinism (matches the Python implementation).
        return [h as i64, MODE_REST, MODE_REST];
    }
    // Defensive: with round-robin and num_agents <= num_houses, every agent
    // owns at least one house. Fall back to house 0 + REST rather than
    // panic; matches the Python `if owned.size == 0` branch.
    [0, MODE_REST, MODE_REST]
}

/// Convenience wrapper that returns the canonical [`NUM_HOUSES`]-house
/// specialist action. Used by the cell-baseline Monte-Carlo harness.
pub fn specialist_action_canonical(
    flat_obs: &[f32],
    agent_id: usize,
    num_agents: usize,
) -> [i64; 3] {
    specialist_action(flat_obs, agent_id, num_agents, NUM_HOUSES)
}

/// The three no-convergence cells of the workshop-paper heterogeneous
/// phase diagram that PRs #126 (NFSP) and #129 (PSRO) exercise.
///
/// Each variant corresponds to a `(β, κ, c)` triple applied as an overlay
/// to the base `minimal_specialization-v1` frozen scenario:
///
/// * `β = prob_fire_spreads_to_neighbor`
/// * `κ = prob_solo_agent_extinguishes_fire`
/// * `c = cost_to_work_one_night`
///
/// All three cells share `κ = 0.1, c = 0.5`; they differ only in β. They are
/// all tagged `verdict: "no_convergence"` in
/// `envs/bucket-brigade/experiments/nash/phase_diagram/results.json`, which
/// is why the base-scenario baselines are an inadequate normalization point
/// (issue #128). Cell-specific baselines live in
/// [`crate::multi_agent::bucket_brigade_metrics`] as `MINSPEC_RANDOM_BETA0XX`
/// and `MINSPEC_SPECIALIST_BETA0XX` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketBrigadeCell {
    /// `(β = 0.1, κ = 0.1, c = 0.5)`.
    Beta01,
    /// `(β = 0.5, κ = 0.1, c = 0.5)` — the canonical cell PR #126 targets.
    Beta05,
    /// `(β = 0.9, κ = 0.1, c = 0.5)`.
    Beta09,
}

impl BucketBrigadeCell {
    /// The `(β, κ, c)` triple for this cell.
    pub fn parameters(&self) -> (f32, f32, f32) {
        match self {
            BucketBrigadeCell::Beta01 => (0.1, 0.1, 0.5),
            BucketBrigadeCell::Beta05 => (0.5, 0.1, 0.5),
            BucketBrigadeCell::Beta09 => (0.9, 0.1, 0.5),
        }
    }

    /// Short cell tag, matching the workshop-paper phase-diagram label
    /// convention (`b0.10_k0.10_c0.50` etc.). Used in log lines and the
    /// `recompute_cell_baselines` test output.
    pub fn tag(&self) -> &'static str {
        match self {
            BucketBrigadeCell::Beta01 => "b0.10_k0.10_c0.50",
            BucketBrigadeCell::Beta05 => "b0.50_k0.10_c0.50",
            BucketBrigadeCell::Beta09 => "b0.90_k0.10_c0.50",
        }
    }

    /// All three no-convergence cells in canonical order.
    pub fn all() -> [BucketBrigadeCell; 3] {
        [BucketBrigadeCell::Beta01, BucketBrigadeCell::Beta05, BucketBrigadeCell::Beta09]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal flat observation with a given `houses` slice. The
    /// remaining fields are zero-padded — the specialist only consults
    /// `houses`, so the padding is irrelevant.
    fn make_obs(houses: &[u8], num_houses: usize) -> Vec<f32> {
        let mut obs = vec![0.0; 1 + num_houses + 64];
        for (i, &h) in houses.iter().enumerate().take(num_houses) {
            obs[1 + i] = h as f32;
        }
        obs
    }

    /// All houses safe: every agent rests on its lowest-index owned house.
    /// Round-robin (`num_agents=4, num_houses=10`):
    ///   agent 0 -> [0, 4, 8] -> rest on house 0
    ///   agent 1 -> [1, 5, 9] -> rest on house 1
    ///   agent 2 -> [2, 6]    -> rest on house 2
    ///   agent 3 -> [3, 7]    -> rest on house 3
    #[test]
    fn all_safe_rests_on_lowest_owned() {
        let obs = make_obs(&[0; NUM_HOUSES], NUM_HOUSES);
        assert_eq!(specialist_action(&obs, 0, 4, NUM_HOUSES), [0, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 1, 4, NUM_HOUSES), [1, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 2, 4, NUM_HOUSES), [2, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 3, 4, NUM_HOUSES), [3, MODE_REST, MODE_REST]);
    }

    /// Single owned house burning: agent fights it, honest signal.
    /// Agent 0 owns [0, 4, 8]; setting house 4 = BURNING means agent 0
    /// should `[4, WORK, WORK]`. Other agents (who don't own house 4)
    /// should continue to REST on their lowest-index owned house.
    #[test]
    fn single_burning_owned_works_with_honest_signal() {
        let mut houses = [0u8; NUM_HOUSES];
        houses[4] = HOUSE_BURNING;
        let obs = make_obs(&houses, NUM_HOUSES);
        // Agent 0 owns house 4 (since 4 % 4 == 0): fights it.
        assert_eq!(specialist_action(&obs, 0, 4, NUM_HOUSES), [4, MODE_WORK, MODE_WORK]);
        // Agents 1, 2, 3 don't own house 4: rest on their first owned house.
        assert_eq!(specialist_action(&obs, 1, 4, NUM_HOUSES), [1, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 2, 4, NUM_HOUSES), [2, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 3, 4, NUM_HOUSES), [3, MODE_REST, MODE_REST]);
    }

    /// Multiple burning owned houses: agent picks the lowest-index one.
    /// Agent 1 owns [1, 5, 9]; setting houses 5 and 9 burning should
    /// make agent 1 fight house 5 (the lowest-index burning owned).
    #[test]
    fn multiple_burning_owned_picks_lowest() {
        let mut houses = [0u8; NUM_HOUSES];
        houses[5] = HOUSE_BURNING;
        houses[9] = HOUSE_BURNING;
        let obs = make_obs(&houses, NUM_HOUSES);
        assert_eq!(specialist_action(&obs, 1, 4, NUM_HOUSES), [5, MODE_WORK, MODE_WORK]);
    }

    /// Burning house not owned by the agent is ignored. Agent 1 owns
    /// [1, 5, 9]; setting house 0 burning (agent 0's house) should NOT
    /// make agent 1 fight it — agent 1 rests on its lowest-index owned
    /// house (1).
    #[test]
    fn burning_not_owned_is_ignored() {
        let mut houses = [0u8; NUM_HOUSES];
        houses[0] = HOUSE_BURNING;
        let obs = make_obs(&houses, NUM_HOUSES);
        assert_eq!(specialist_action(&obs, 1, 4, NUM_HOUSES), [1, MODE_REST, MODE_REST]);
    }

    /// Ruined houses do not count as burning. The specialist only
    /// reacts to `state == BURNING (1)`, not `RUINED (2)`.
    #[test]
    fn ruined_house_is_ignored() {
        let mut houses = [0u8; NUM_HOUSES];
        houses[4] = HOUSE_RUINED;
        let obs = make_obs(&houses, NUM_HOUSES);
        // Agent 0 owns house 4 but it's RUINED, not BURNING: rest on house 0.
        assert_eq!(specialist_action(&obs, 0, 4, NUM_HOUSES), [0, MODE_REST, MODE_REST]);
    }

    /// Python parity spot-check (hand-derived from the Python
    /// reference): with `houses = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]`,
    /// num_agents = 4, num_houses = 10:
    ///
    ///   - agent 0 owns [0, 4, 8]; house 4 burns -> [4, WORK, WORK]
    ///   - agent 1 owns [1, 5, 9]; house 1 burns -> [1, WORK, WORK]
    ///   - agent 2 owns [2, 6]; none burn -> [2, REST, REST]
    ///   - agent 3 owns [3, 7]; none burn -> [3, REST, REST]
    ///
    /// This matches the output of `specialist_action_joint` in Python
    /// for the same observation.
    #[test]
    fn matches_python_handpicked_observation() {
        let mut houses = [0u8; NUM_HOUSES];
        houses[1] = HOUSE_BURNING;
        houses[4] = HOUSE_BURNING;
        let obs = make_obs(&houses, NUM_HOUSES);
        assert_eq!(specialist_action(&obs, 0, 4, NUM_HOUSES), [4, MODE_WORK, MODE_WORK]);
        assert_eq!(specialist_action(&obs, 1, 4, NUM_HOUSES), [1, MODE_WORK, MODE_WORK]);
        assert_eq!(specialist_action(&obs, 2, 4, NUM_HOUSES), [2, MODE_REST, MODE_REST]);
        assert_eq!(specialist_action(&obs, 3, 4, NUM_HOUSES), [3, MODE_REST, MODE_REST]);
    }

    #[test]
    fn cell_parameters() {
        assert_eq!(BucketBrigadeCell::Beta01.parameters(), (0.1, 0.1, 0.5));
        assert_eq!(BucketBrigadeCell::Beta05.parameters(), (0.5, 0.1, 0.5));
        assert_eq!(BucketBrigadeCell::Beta09.parameters(), (0.9, 0.1, 0.5));
    }

    #[test]
    fn cell_tags() {
        assert_eq!(BucketBrigadeCell::Beta01.tag(), "b0.10_k0.10_c0.50");
        assert_eq!(BucketBrigadeCell::Beta05.tag(), "b0.50_k0.10_c0.50");
        assert_eq!(BucketBrigadeCell::Beta09.tag(), "b0.90_k0.10_c0.50");
    }
}
