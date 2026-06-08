//! Versioned scenario registry for Bucket Brigade (Rust mirror).
//!
//! This is the Rust-side counterpart of
//! `bucket_brigade/envs/registry.py` (shipped in bucket-brigade PR #379).
//! It exposes a *frozen-by-ID* registry of scenarios so that paper results
//! are reproducible across the Python (Gymnasium) and Rust (Thrust) bindings
//! by sharing the same scenario IDs.
//!
//! # Why mirror it on the Rust side?
//!
//! The upstream `bucket-brigade-core` crate exposes a flat
//! [`SCENARIOS`](bucket_brigade_core::SCENARIOS) map keyed by *unversioned*
//! base names (`"minimal_specialization"`, `"default"`, …) — that's the
//! engine's runtime scenario database. The Python `SCENARIO_VERSIONS`
//! registry layers a frozen versioned ID (`-v1`, `-v2`, …) on top of those
//! base names; each versioned ID maps to a frozen scenario factory.
//!
//! Thrust does not link the Python registry, so we re-publish the version
//! mapping in pure Rust here. Each entry is a `(base_name, default_num_agents)`
//! pair pointing at the matching `bucket-brigade-core` scenario. The mapping
//! is **append-only**: once an ID is published, its parameters MUST NOT
//! change. If a scenario's parameters need to evolve, add a new `-vN` entry
//! and leave the old one untouched, matching the Python policy.
//!
//! # Synchronization with Python
//!
//! Both sides resolve a versioned ID to the same underlying
//! `bucket-brigade-core` `Scenario` value, so a Thrust env constructed via
//! [`make`] is byte-equivalent to a Python env constructed via
//! `bucket_brigade.make()` for the same ID (same scenario, same num_agents).
//! When adding a new scenario, update both this file and the Python
//! `SCENARIO_VERSIONS` table together.
//!
//! # Parsing helpers
//!
//! [`parse_scenario_id`] splits an ID like `"minimal_specialization-v1"` into
//! `("minimal_specialization", 1)` for diagnostics; this mirrors
//! `bucket_brigade.envs.registry.parse_scenario_id`.

use bucket_brigade_core::{SCENARIOS, Scenario};

/// Default `num_agents` for frozen scenario IDs.
///
/// Mirrors `bucket_brigade.envs.registry.DEFAULT_NUM_AGENTS = 4`.
/// Every published Bucket Brigade result in this repo uses 4 agents.
/// Pinning this default into the registry is part of what makes an ID
/// *frozen*: [`make`] with a versioned ID and no `num_agents` override
/// always yields a 4-agent scenario.
pub const DEFAULT_NUM_AGENTS: usize = 4;

/// A frozen scenario entry: the base-name lookup key in
/// `bucket_brigade_core::SCENARIOS` plus a default `num_agents`.
#[derive(Debug, Clone, Copy)]
struct ScenarioVersion {
    /// Key into [`bucket_brigade_core::SCENARIOS`] — the unversioned base
    /// name (e.g. `"minimal_specialization"`).
    base: &'static str,
    /// Default `num_agents` baked into the frozen ID.
    default_num_agents: usize,
}

/// Versioned scenario registry. **APPEND-ONLY** in the version direction —
/// once an ID ships, its row MUST NOT be modified.
///
/// Each row maps a versioned ID to the base scenario name in
/// `bucket-brigade-core` plus the frozen `num_agents` default. This list is
/// the Rust mirror of `bucket_brigade.envs.registry.SCENARIO_VERSIONS`.
/// Keep the two in sync when adding new entries.
const SCENARIO_VERSIONS: &[(&str, ScenarioVersion)] = &[
    // Default-family scenarios.
    ("default-v1", ScenarioVersion { base: "default", default_num_agents: 4 }),
    ("hard-v1", ScenarioVersion { base: "hard", default_num_agents: 4 }),
    // Named test scenarios from `definitions/scenarios.json`.
    (
        "trivial_cooperation-v1",
        ScenarioVersion { base: "trivial_cooperation", default_num_agents: 4 },
    ),
    (
        "early_containment-v1",
        ScenarioVersion { base: "early_containment", default_num_agents: 4 },
    ),
    (
        "greedy_neighbor-v1",
        ScenarioVersion { base: "greedy_neighbor", default_num_agents: 4 },
    ),
    (
        "sparse_heroics-v1",
        ScenarioVersion { base: "sparse_heroics", default_num_agents: 4 },
    ),
    ("rest_trap-v1", ScenarioVersion { base: "rest_trap", default_num_agents: 4 }),
    (
        "chain_reaction-v1",
        ScenarioVersion { base: "chain_reaction", default_num_agents: 4 },
    ),
    (
        "deceptive_calm-v1",
        ScenarioVersion { base: "deceptive_calm", default_num_agents: 4 },
    ),
    (
        "overcrowding-v1",
        ScenarioVersion { base: "overcrowding", default_num_agents: 4 },
    ),
    (
        "mixed_motivation-v1",
        ScenarioVersion { base: "mixed_motivation", default_num_agents: 4 },
    ),
    // P3 / specialization diagnostics (issue #199 and follow-ups).
    (
        "minimal_specialization-v1",
        ScenarioVersion { base: "minimal_specialization", default_num_agents: 4 },
    ),
    // 2-house topology for PPO learnability diagnostics (#254).
    ("v2_minimal-v1", ScenarioVersion { base: "v2_minimal", default_num_agents: 4 }),
];

/// Return a sorted list of frozen scenario IDs.
///
/// Mirrors `bucket_brigade.envs.registry.list_versioned_scenarios()`.
pub fn list_versioned_scenarios() -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = SCENARIO_VERSIONS.iter().map(|(id, _)| *id).collect();
    ids.sort_unstable();
    ids
}

/// Split a frozen scenario ID into `(base_name, version)`.
///
/// # Errors
///
/// Returns an error if the ID does not match the `<name>-v<int>` shape.
///
/// Mirrors `bucket_brigade.envs.registry.parse_scenario_id`.
pub fn parse_scenario_id(scenario_id: &str) -> Result<(&str, u32), String> {
    let Some(idx) = scenario_id.rfind("-v") else {
        return Err(format!(
            "Invalid scenario ID {scenario_id:?}: expected '<name>-v<int>' shape \
             (e.g. 'minimal_specialization-v1')."
        ));
    };
    let (base, rest) = scenario_id.split_at(idx);
    let version_str = &rest[2..];
    if base.is_empty() || version_str.is_empty() {
        return Err(format!(
            "Invalid scenario ID {scenario_id:?}: expected '<name>-v<int>' shape with a \
             non-empty base name and integer version."
        ));
    }
    let version: u32 = version_str.parse().map_err(|_| {
        format!(
            "Invalid scenario ID {scenario_id:?}: version suffix must parse as a u32 \
             (e.g. 'v1', 'v2')."
        )
    })?;
    Ok((base, version))
}

/// The default `num_agents` baked into a versioned scenario ID.
///
/// Returns `None` if the ID is not registered. Use [`list_versioned_scenarios`]
/// to discover valid IDs.
pub fn default_num_agents_for(scenario_id: &str) -> Option<usize> {
    SCENARIO_VERSIONS.iter().find_map(|(id, v)| {
        if *id == scenario_id {
            Some(v.default_num_agents)
        } else {
            None
        }
    })
}

/// Look up a [`Scenario`] by its frozen versioned ID.
///
/// # Errors
///
/// Returns an error if `scenario_id` is not registered.
///
/// Mirrors `bucket_brigade.envs.registry.get_scenario_by_id`. The `num_agents`
/// parameter is forwarded to consumer code — the underlying
/// `bucket_brigade_core::Scenario` value itself does not carry a num_agents
/// field; consumers pass it to [`super::env::BucketBrigadeMaEnv::new`].
pub fn get_scenario_by_id(scenario_id: &str) -> Result<Scenario, String> {
    let entry = SCENARIO_VERSIONS
        .iter()
        .find_map(|(id, v)| if *id == scenario_id { Some(v) } else { None })
        .ok_or_else(|| {
            let available = list_versioned_scenarios().join(", ");
            format!("Unknown scenario ID {scenario_id:?}. Available IDs: {available}")
        })?;
    SCENARIOS.get(entry.base).cloned().ok_or_else(|| {
        format!(
            "Scenario ID {scenario_id:?} maps to base name {:?}, which is not present in \
                 bucket_brigade_core::SCENARIOS. The submodule and the Thrust registry are out of \
                 sync.",
            entry.base
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_id() {
        let (base, version) = parse_scenario_id("minimal_specialization-v1").unwrap();
        assert_eq!(base, "minimal_specialization");
        assert_eq!(version, 1);
    }

    #[test]
    fn parse_id_without_version_errors() {
        assert!(parse_scenario_id("minimal_specialization").is_err());
        assert!(parse_scenario_id("minimal_specialization-").is_err());
        assert!(parse_scenario_id("-v1").is_err());
        assert!(parse_scenario_id("foo-vbar").is_err());
    }

    #[test]
    fn parse_handles_dashes_in_base_name() {
        // We rfind("-v") so a base name containing other dashes survives.
        let (base, version) = parse_scenario_id("foo-bar-v2").unwrap();
        assert_eq!(base, "foo-bar");
        assert_eq!(version, 2);
    }

    #[test]
    fn registry_includes_minimal_specialization() {
        let ids = list_versioned_scenarios();
        assert!(ids.contains(&"minimal_specialization-v1"));
    }

    #[test]
    fn registry_is_sorted() {
        let ids = list_versioned_scenarios();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn get_scenario_returns_minimal_specialization() {
        let scenario = get_scenario_by_id("minimal_specialization-v1").unwrap();
        // Matches the Python factory `minimal_specialization_scenario`:
        // team_reward = 10.0, team_penalty = 10.0, reward_own = [50.0; n],
        // penalty_own = [100.0; n].
        assert_eq!(scenario.team_reward_house_survives, 10.0);
        assert_eq!(scenario.team_penalty_house_burns, 10.0);
        assert!(scenario.reward_own_house_survives.iter().all(|&v| v == 50.0));
        assert!(scenario.penalty_own_house_burns.iter().all(|&v| v == 100.0));
    }

    #[test]
    fn unknown_id_errors() {
        let err = get_scenario_by_id("nonexistent-v1").unwrap_err();
        assert!(err.contains("nonexistent-v1"));
        assert!(err.contains("minimal_specialization-v1"));
    }

    #[test]
    fn default_num_agents_lookup() {
        assert_eq!(default_num_agents_for("minimal_specialization-v1"), Some(4));
        assert_eq!(default_num_agents_for("nonexistent-v1"), None);
    }

    #[test]
    fn every_registered_id_resolves() {
        for id in list_versioned_scenarios() {
            get_scenario_by_id(id)
                .unwrap_or_else(|e| panic!("registry entry {id} does not resolve: {e}"));
        }
    }
}
