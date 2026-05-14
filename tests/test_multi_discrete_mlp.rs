//! Tests for `MultiDiscreteMlpPolicy`.

use tch::{Kind, Tensor};
use thrust_rl::policy::multi_discrete_mlp::MultiDiscreteMlpPolicy;

#[test]
fn forward_returns_per_dim_logits_and_value() {
    // Bucket-Brigade-style [10, 2] action space.
    let policy = MultiDiscreteMlpPolicy::new(42, vec![10, 2], 64);
    let obs = Tensor::randn([8, 42], (Kind::Float, policy.device()));

    let (logits_per_dim, values) = policy.forward(&obs);
    assert_eq!(logits_per_dim.len(), 2);
    assert_eq!(logits_per_dim[0].size(), vec![8, 10]);
    assert_eq!(logits_per_dim[1].size(), vec![8, 2]);
    assert_eq!(values.size(), vec![8]);
}

#[test]
fn get_action_returns_correct_shapes_and_ranges() {
    let policy = MultiDiscreteMlpPolicy::new(8, vec![5, 3, 2], 32);
    let obs = Tensor::randn([16, 8], (Kind::Float, policy.device()));

    let (actions, log_probs, values) = policy.get_action(&obs);
    assert_eq!(actions.size(), vec![16, 3]);
    assert_eq!(log_probs.size(), vec![16]);
    assert_eq!(values.size(), vec![16]);

    // Each per-dim action must lie in its range.
    let actions_vec: Vec<i64> = Vec::try_from(&actions.flatten(0, -1)).unwrap();
    let mut idx = 0;
    for _row in 0..16 {
        for (col, &max) in [5_i64, 3, 2].iter().enumerate() {
            let a = actions_vec[idx];
            assert!(a >= 0 && a < max, "actions[row, {col}] = {a} not in 0..{max}",);
            idx += 1;
        }
    }
}

#[test]
fn evaluate_actions_matches_get_action_log_probs() {
    // If we sample actions with get_action and then re-evaluate them with
    // evaluate_actions, we should recover the same summed log-prob.
    let policy = MultiDiscreteMlpPolicy::new(4, vec![3, 2], 16);
    let obs = Tensor::randn([32, 4], (Kind::Float, policy.device()));

    let (actions, sampled_lp, _values) = policy.get_action(&obs);
    let (recomputed_lp, _entropy, _values) = policy.evaluate_actions(&obs, &actions);

    // Tolerance is loose because the operations are floats, but they should
    // be near-identical.
    let diff = (sampled_lp - recomputed_lp).abs().max();
    let diff_val: f64 = f64::try_from(&diff).unwrap();
    assert!(
        diff_val < 1e-5,
        "evaluate_actions log_probs differ from get_action's by {}",
        diff_val
    );
}

#[test]
fn entropy_is_positive_for_uniform_init() {
    // With near-uniform action probabilities (orthogonal init + small output
    // gain), per-step entropy should be positive.
    let policy = MultiDiscreteMlpPolicy::new(4, vec![10, 2], 64);
    let obs = Tensor::randn([64, 4], (Kind::Float, policy.device()));

    let actions = Tensor::zeros([64, 2], (Kind::Int64, policy.device()));
    let (_lp, entropy, _v) = policy.evaluate_actions(&obs, &actions);

    let entropy_vec: Vec<f32> = Vec::try_from(&entropy).unwrap();
    let mean_entropy: f32 = entropy_vec.iter().sum::<f32>() / entropy_vec.len() as f32;
    assert!(mean_entropy > 0.0, "entropy should be > 0, got {}", mean_entropy);
}

#[test]
fn encoder_features_carries_gradient() {
    // Same gradient-flow test as for MlpPolicy, but on the multi-discrete
    // variant.
    let mut policy = MultiDiscreteMlpPolicy::new(4, vec![3, 2], 16);
    let mut opt = policy.optimizer(3e-4).unwrap();
    let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));

    let features = policy.encoder_features(&obs);
    let loss = features.sum(Kind::Float);
    opt.zero_grad();
    loss.backward();
    opt.step();
}

#[test]
#[should_panic(expected = "action_dims must have at least one element")]
fn rejects_empty_action_dims() {
    let _ = MultiDiscreteMlpPolicy::new(4, vec![], 16);
}

#[test]
#[should_panic(expected = "action_dims[1] = 0; must be >= 1")]
fn rejects_zero_action_dim() {
    let _ = MultiDiscreteMlpPolicy::new(4, vec![3, 0, 2], 16);
}
