//! `MultiAgentEnvironment` impl for [`BucketBrigadeMaEnv`].
//!
//! Gated behind the `training` feature because that's the gate the
//! `crate::multi_agent` module lives under. The env itself
//! ([`super::env::BucketBrigadeMaEnv`]) is usable without `training` — only
//! the multi-discrete `MultiAgentEnvironment` adapter requires it.

use bucket_brigade_core::Action;

use super::env::{ACTION_DIMS, BucketBrigadeMaEnv};
use crate::multi_agent::{MultiAgentEnvironment, MultiAgentResult};

impl MultiAgentEnvironment for BucketBrigadeMaEnv {
    fn num_agents(&self) -> usize {
        BucketBrigadeMaEnv::num_agents(self)
    }

    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32> {
        // Re-use the env's flattener via a cheap one-agent slice. We can't
        // call the free `flatten_observation` directly here without exposing
        // it; this round-trip through `inner` is the simplest surface.
        let inner = self.inner();
        super::env::flatten_observation(&inner.get_observation(agent_id), self.num_houses())
    }

    fn agent_action_space(&self, _agent_id: usize) -> Vec<usize> {
        // Factored `[house_index, mode, signal]` — matches the env's
        // `action_dims()`; downstream policies (e.g.
        // `MultiDiscreteMlpPolicy`) should be constructed with these dims.
        BucketBrigadeMaEnv::action_dims(self).into_iter().map(|d| d as usize).collect()
    }

    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult {
        let num_agents = BucketBrigadeMaEnv::num_agents(self);
        assert_eq!(
            actions.len(),
            num_agents,
            "BucketBrigadeMaEnv::step_multi: expected {num_agents} actions, got {}",
            actions.len()
        );

        let num_houses = self.num_houses();
        let inner_actions: Vec<Action> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                assert_eq!(
                    a.len(),
                    ACTION_DIMS,
                    "BucketBrigadeMaEnv::step_multi: agent {i} action must have {ACTION_DIMS} \
                     dims ([house_index, mode, signal]), got {}",
                    a.len()
                );
                assert!(
                    (0..num_houses as i64).contains(&a[0]),
                    "BucketBrigadeMaEnv::step_multi: agent {i} house_index {} out of range \
                     0..{num_houses}",
                    a[0]
                );
                assert!(
                    a[1] == 0 || a[1] == 1,
                    "BucketBrigadeMaEnv::step_multi: agent {i} mode {} must be 0 or 1",
                    a[1]
                );
                assert!(
                    a[2] == 0 || a[2] == 1,
                    "BucketBrigadeMaEnv::step_multi: agent {i} signal {} must be 0 or 1",
                    a[2]
                );
                [a[0] as u8, a[1] as u8, a[2] as u8]
            })
            .collect();

        let result = self.step(&inner_actions);
        let n = num_agents;
        MultiAgentResult::new(
            result.observations,
            result.rewards,
            vec![result.done; n],
            vec![false; n],
        )
    }

    fn active_agents(&self) -> Vec<bool> {
        // The underlying engine has no per-agent termination signal; either
        // the whole night is still running or the episode is over.
        vec![true; BucketBrigadeMaEnv::num_agents(self)]
    }
}
