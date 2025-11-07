//! Tests for rollout buffer functionality

#[cfg(test)]
mod gae_tests {
    use crate::buffer::rollout::{gae::compute_gae_single_env, storage::RolloutBuffer};

    #[test]
    fn test_gae_episode_boundaries() {
        // Test that GAE correctly handles episode boundaries
        // Episode 1: steps 0-2 (ends at step 2)
        // Episode 2: steps 3-4

        let rewards = vec![1.0, 1.0, 1.0, 2.0, 2.0];  // Different rewards for each episode
        let values = vec![0.5, 0.5, 0.5, 1.0, 1.0];
        let terminated = vec![false, false, true, false, false];

        let mut advantages = vec![0.0; 5];
        let mut returns = vec![0.0; 5];

        compute_gae_single_env(
            &rewards,
            &values,
            &terminated,
            0.0,  // last_value = 0 for simplicity
            0.99,  // gamma
            0.95,  // gae_lambda
            &mut advantages,
            &mut returns,
        );

        // Episode 1 (steps 0-2) should have advantages independent of episode 2
        // Episode 2 (steps 3-4) should have different advantage values

        // For episode 1, step 2 (terminal):
        // delta = reward[2] + 0 - value[2] = 1.0 - 0.5 = 0.5
        // gae = delta (no accumulation because GAE was reset)
        // advantage[2] = 0.5
        assert!((advantages[2] - 0.5).abs() < 0.01, "Terminal step advantage incorrect: {}", advantages[2]);

        // For episode 2, advantages should be computed independently
        // They should NOT be affected by episode 1 values
        println!("Advantages: {:?}", advantages);
        println!("Returns: {:?}", returns);

        // Episode 2 should have different advantages than episode 1
        // because rewards are different (2.0 vs 1.0)
        assert!((advantages[3] - advantages[0]).abs() > 0.1,
                "Episode 2 advantages should differ from episode 1");
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
            1.0,  // gamma = 1.0 for simplicity
            1.0,  // gae_lambda = 1.0
            &mut advantages,
            &mut returns,
        );

        // With gamma=1, gae_lambda=1, last_value=0:
        // Step 2 (terminal): delta = 1.0 + 0 - 2.0 = -1.0, gae = -1.0
        // Step 1: delta = 1.0 + 2.0 - 2.0 = 1.0, gae = 1.0 + 1.0*1.0*(-1.0) = 0.0
        // Step 0: delta = 1.0 + 2.0 - 2.0 = 1.0, gae = 1.0 + 1.0*1.0*0.0 = 1.0

        println!("Simple episode advantages: {:?}", advantages);
        assert!((advantages[2] - (-1.0)).abs() < 0.01, "Step 2: expected -1.0, got {}", advantages[2]);
        assert!((advantages[1] - 0.0).abs() < 0.01, "Step 1: expected 0.0, got {}", advantages[1]);
        assert!((advantages[0] - 1.0).abs() < 0.01, "Step 0: expected 1.0, got {}", advantages[0]);
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
                let terminated = step == 2;  // Episode ends at step 2
                buffer.add(
                    step,
                    env,
                    &vec![0.0],
                    0,
                    1.0,  // reward
                    0.5,  // value
                    0.0,  // log_prob
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
        for step in 0..num_steps {
            println!("  Step {}: {:?}", step, advantages[step]);
        }

        // Advantages should be different before and after episode boundary
        // (step 2 is terminal, step 3 starts new episode)
        assert!(advantages[3][0] != 0.0, "Advantages should be non-zero");
    }
}
