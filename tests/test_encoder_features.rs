//! Tests for the public `MlpPolicy::encoder_features` tap.
//!
//! This is an integration test (separate test binary) rather than a unit test
//! in `src/policy/mlp.rs` because the in-crate test build currently fails to
//! compile in `src/multi_agent/simulator.rs` and `src/inference/weights.rs`
//! (pre-existing schema drift unrelated to this change). Once those are
//! fixed, the tests below can move into the module-level `mod tests` block.

use tch::{Kind, Tensor};
use thrust_rl::policy::mlp::MlpPolicy;

#[test]
fn encoder_features_returns_hidden_dim_shape() {
    let policy = MlpPolicy::new(4, 2, 64);
    let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));
    let features = policy.encoder_features(&obs);
    assert_eq!(features.size(), vec![8, 64]);
}

#[test]
fn encoder_features_carries_gradient() {
    // Auxiliary losses computed from the encoder tap must be able to backprop
    // into the trunk. Smoke-test by .backward()-ing a scalar function of the
    // features and checking that the optimizer step runs.
    let mut policy = MlpPolicy::new(4, 2, 64);
    let mut opt = policy.optimizer(3e-4);
    let obs = Tensor::randn([8, 4], (Kind::Float, policy.device()));

    let features = policy.encoder_features(&obs);
    let loss = features.sum(Kind::Float);
    opt.zero_grad();
    loss.backward();
    opt.step();
}

#[test]
fn encoder_features_and_forward_use_same_trunk() {
    // The tap should produce the *same* features that the heads see in
    // forward(). Verify by checking that an MSE-style training loop on the
    // features changes the forward() output (it would not, if encoder_features
    // were a parallel module).
    let mut policy = MlpPolicy::new(4, 2, 64);
    let mut opt = policy.optimizer(1e-1);
    let obs = Tensor::randn([16, 4], (Kind::Float, policy.device()));

    let (logits_before, _) = policy.forward(&obs);
    let logits_before = logits_before.detach().shallow_clone();

    // Train the trunk to drive features toward zero --- this should also
    // change forward()'s output, since the heads consume the trunk.
    for _ in 0..20 {
        let features = policy.encoder_features(&obs);
        let loss = features.pow_tensor_scalar(2).mean(Kind::Float);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let (logits_after, _) = policy.forward(&obs);
    let diff = (logits_after - logits_before).abs().sum(Kind::Float);
    let diff_val: f64 = f64::try_from(&diff).unwrap();
    assert!(
        diff_val > 1e-4,
        "encoder_features doesn't share the trunk with forward(): logits unchanged after \
         features-only training (diff = {})",
        diff_val
    );
}
