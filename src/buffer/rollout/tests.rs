//! Tests for rollout buffer functionality

#[cfg(test)]
mod gae_tests {
    use crate::buffer::rollout::{
        compute_advantages_multi_agent, gae::compute_gae_single_env, storage::RolloutBuffer,
    };

    #[test]
    fn test_gae_episode_boundaries() {
        // Test that GAE correctly handles episode boundaries
        // Episode 1: steps 0-2 (ends at step 2)
        // Episode 2: steps 3-4

        let rewards = vec![1.0, 1.0, 1.0, 2.0, 2.0]; // Different rewards for each episode
        let values = vec![0.5, 0.5, 0.5, 1.0, 1.0];
        let terminated = vec![false, false, true, false, false];

        let mut advantages = vec![0.0; 5];
        let mut returns = vec![0.0; 5];

        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            0.0,  // last_value = 0 for simplicity
            0.99, // gamma
            0.95, // gae_lambda
            &mut advantages,
            &mut returns,
        );

        // Episode 1 (steps 0-2) should have advantages independent of episode 2
        // Episode 2 (steps 3-4) should have different advantage values

        // For episode 1, step 2 (terminal):
        // delta = reward[2] + 0 - value[2] = 1.0 - 0.5 = 0.5
        // gae = delta (no accumulation because GAE was reset)
        // advantage[2] = 0.5
        assert!(
            (advantages[2] - 0.5).abs() < 0.01,
            "Terminal step advantage incorrect: {}",
            advantages[2]
        );

        // For episode 2, advantages should be computed independently
        // They should NOT be affected by episode 1 values
        println!("Advantages: {:?}", advantages);
        println!("Returns: {:?}", returns);

        // Episode 2 should have different advantages than episode 1
        // because rewards are different (2.0 vs 1.0)
        assert!(
            (advantages[3] - advantages[0]).abs() > 0.1,
            "Episode 2 advantages should differ from episode 1"
        );
    }

    #[test]
    fn test_gae_simple_episode() {
        // Single episode with constant rewards
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![2.0, 2.0, 2.0];
        let terminated = vec![false, false, true];

        let mut advantages = vec![0.0; 3];
        let mut returns = vec![0.0; 3];

        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            0.0,
            1.0, // gamma = 1.0 for simplicity
            1.0, // gae_lambda = 1.0
            &mut advantages,
            &mut returns,
        );

        // With gamma=1, gae_lambda=1, last_value=0:
        // Step 2 (terminal): delta = 1.0 + 0 - 2.0 = -1.0, gae = -1.0
        // Step 1: delta = 1.0 + 2.0 - 2.0 = 1.0, gae = 1.0 + 1.0*1.0*(-1.0) = 0.0
        // Step 0: delta = 1.0 + 2.0 - 2.0 = 1.0, gae = 1.0 + 1.0*1.0*0.0 = 1.0

        println!("Simple episode advantages: {:?}", advantages);
        assert!(
            (advantages[2] - (-1.0)).abs() < 0.01,
            "Step 2: expected -1.0, got {}",
            advantages[2]
        );
        assert!(
            (advantages[1] - 0.0).abs() < 0.01,
            "Step 1: expected 0.0, got {}",
            advantages[1]
        );
        assert!(
            (advantages[0] - 1.0).abs() < 0.01,
            "Step 0: expected 1.0, got {}",
            advantages[0]
        );
    }

    #[test]
    fn test_rollout_buffer_advantages() {
        // Test the full rollout buffer compute_advantages method
        let num_steps = 5;
        let num_envs = 2;
        let obs_dim = 1;

        let mut buffer = RolloutBuffer::new(num_steps, num_envs, obs_dim);

        // Add some dummy data
        for step in 0..num_steps {
            for env in 0..num_envs {
                let terminated = step == 2; // Episode ends at step 2
                buffer.add(
                    step,
                    env,
                    &[0.0],
                    0,
                    1.0, // reward
                    0.5, // value
                    0.0, // log_prob
                    terminated,
                    false,
                );
            }
        }

        // Compute advantages
        let last_values = vec![0.0; num_envs];
        buffer.compute_advantages(&last_values, 0.99, 0.95);

        // Check that advantages were computed
        let advantages = buffer.advantages();
        println!("Buffer advantages:");
        for (step, adv) in advantages.iter().enumerate().take(num_steps) {
            println!("  Step {}: {:?}", step, adv);
        }

        // Advantages should be different before and after episode boundary
        // (step 2 is terminal, step 3 starts new episode)
        assert!(advantages[3][0] != 0.0, "Advantages should be non-zero");
    }

    /// Hand-computed regression test for
    /// [`compute_advantages_multi_agent`].
    ///
    /// Setup: `num_steps = 2`, `num_envs = 2`, `num_agents = 2`, giving a
    /// flat buffer of 8 entries indexed as
    /// `t * num_envs * num_agents + env * num_agents + agent`. All
    /// rewards are 1.0, all values are 0.0, `gamma = 1.0`, `gae_lambda = 1.0`,
    /// and `last_values` is zero for every (env, agent).
    ///
    /// We mark `dones[3] = true` at step 0, slot 3
    /// (`env = 1, agent = 1`). Every other transition is non-terminal.
    ///
    /// Expected per (env, agent) trajectory (gamma = lambda = 1):
    ///
    /// * Non-terminating trajectory:
    ///   * Step 1: delta = 1 + 1 * 0 - 0 = 1, gae = 1, A_1 = 1.
    ///   * Step 0: delta = 1 + 1 * 0 - 0 = 1, gae = 1 + 1 * 1 = 2, A_0 = 2.
    /// * Terminating trajectory (env=1, agent=1, done at step 0):
    ///   * Step 1: delta = 1, gae = 1, A_1 = 1.
    ///   * Step 0 (terminal): gae resets via the (1 - done) mask, then delta =
    ///     1, gae = 1, A_0 = 1.
    ///
    /// Returns equal advantages here since all values are 0.
    #[test]
    fn test_compute_advantages_multi_agent_two_env_two_agent_with_termination() {
        let num_envs = 2usize;
        let num_agents = 2usize;
        let num_steps = 2usize;
        let stride = num_envs * num_agents;
        let total = num_steps * stride;

        let rewards = vec![1.0_f32; total];
        let values = vec![0.0_f32; total];
        let mut dones = vec![false; total];
        // Index 3 == step 0, env 1, agent 1: early termination.
        dones[3] = true;
        let last_values = vec![0.0_f32; stride];

        let gamma = 1.0_f32;
        let gae_lambda = 1.0_f32;

        let (advantages, returns) = compute_advantages_multi_agent(
            &rewards,
            &values,
            &dones,
            &last_values,
            num_envs,
            num_agents,
            gamma,
            gae_lambda,
        );

        // Layout sanity.
        assert_eq!(advantages.len(), total);
        assert_eq!(returns.len(), total);

        let eps = 1e-6_f32;

        // Non-terminating slots: (env, agent) in {(0,0), (0,1), (1,0)}.
        for (env, agent) in [(0, 0), (0, 1), (1, 0)] {
            let slot = env * num_agents + agent;
            let idx_step0 = slot; // step 0: 0 * stride + slot
            let idx_step1 = stride + slot; // step 1: 1 * stride + slot
            assert!(
                (advantages[idx_step0] - 2.0).abs() < eps,
                "(env={}, agent={}) step 0 advantage expected 2.0, got {}",
                env,
                agent,
                advantages[idx_step0]
            );
            assert!(
                (advantages[idx_step1] - 1.0).abs() < eps,
                "(env={}, agent={}) step 1 advantage expected 1.0, got {}",
                env,
                agent,
                advantages[idx_step1]
            );
        }

        // Terminating slot: (env=1, agent=1). The done at step 0 must
        // prevent the step-1 advantage from leaking back into step 0.
        let term_slot = num_agents + 1; // env=1, agent=1: 1 * num_agents + 1
        let term_step0 = term_slot; // step 0: 0 * stride + term_slot
        let term_step1 = stride + term_slot; // step 1: 1 * stride + term_slot
        assert!(
            (advantages[term_step0] - 1.0).abs() < eps,
            "terminating slot step 0 advantage expected 1.0 (no bootstrap, \
             no GAE accumulation), got {}",
            advantages[term_step0]
        );
        assert!(
            (advantages[term_step1] - 1.0).abs() < eps,
            "terminating slot step 1 advantage expected 1.0, got {}",
            advantages[term_step1]
        );

        // Returns == advantages + values; values are all zero so returns
        // should equal advantages.
        for i in 0..total {
            assert!(
                (returns[i] - advantages[i]).abs() < eps,
                "returns[{}] expected == advantages[{}] (values are zero), got R={}, A={}",
                i,
                i,
                returns[i],
                advantages[i]
            );
        }
    }

    /// `compute_advantages_multi_agent` with a single env and single
    /// agent must reduce to the same math as `compute_gae_single_env`.
    /// This pins down that the multi-agent generalization is a strict
    /// superset of the single-agent case.
    #[test]
    fn test_compute_advantages_multi_agent_degenerates_to_single_agent() {
        let rewards = vec![1.0_f32, 0.5, -0.25, 0.75];
        let values = vec![0.4_f32, 0.6, 0.8, 0.2];
        let terminated = vec![false, false, true, false];
        let bootstrap = 0.7_f32;
        let gamma = 0.99_f32;
        let gae_lambda = 0.95_f32;

        // Reference: single-agent in-place GAE.
        let mut ref_advantages = vec![0.0_f32; rewards.len()];
        let mut ref_returns = vec![0.0_f32; rewards.len()];
        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            bootstrap,
            gamma,
            gae_lambda,
            &mut ref_advantages,
            &mut ref_returns,
        );

        // Multi-agent path with num_envs=1, num_agents=1.
        let (advantages, returns) = compute_advantages_multi_agent(
            &rewards,
            &values,
            &terminated,
            &[bootstrap],
            1,
            1,
            gamma,
            gae_lambda,
        );

        let eps = 1e-6_f32;
        for i in 0..rewards.len() {
            assert!(
                (advantages[i] - ref_advantages[i]).abs() < eps,
                "advantage[{}] mismatch: multi-agent={}, single-agent ref={}",
                i,
                advantages[i],
                ref_advantages[i]
            );
            assert!(
                (returns[i] - ref_returns[i]).abs() < eps,
                "return[{}] mismatch: multi-agent={}, single-agent ref={}",
                i,
                returns[i],
                ref_returns[i]
            );
        }
    }
}
