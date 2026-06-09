//! Numerical-parity tests between the tch and Burn loss
//! implementations (phase 3 of the Burn migration, #80).
//!
//! These tests run only when both `training` (tch) and `training-burn`
//! (Burn) features are compiled in. They enforce the DoD contract on
//! issue #80: identical inputs to the two backend implementations of
//! the PPO and DQN loss math must agree to **1e-4 absolute tolerance**.
//!
//! # What is and isn't compared
//!
//! - **In scope**: scalar values returned by the loss functions
//!   (`policy_loss`, `value_loss`, `entropy_loss`, `td_target`,
//!   `td_loss`). These are computed from synthetic inputs that exist
//!   on both backends and have known numerical answers.
//! - **Out of scope**: gradients. Burn and tch use different autograd
//!   implementations and their derivatives may differ in the last bit
//!   even when the forward pass matches; gradient-level parity is
//!   tested at the trainer-integration level in phases 4/5.

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Int, Tensor as BurnTensor, TensorData},
};
use tch::Tensor as TchTensor;

use crate::train::{dqn_burn::loss as dqn_loss_burn, ppo_burn::loss as ppo_loss_burn};

type B = Autodiff<NdArray<f32>>;

const PARITY_TOL: f64 = 1e-4;

fn tch1(data: &[f32]) -> TchTensor {
    TchTensor::from_slice(data)
}

fn tch2(data: &[f32], rows: usize, cols: usize) -> TchTensor {
    TchTensor::from_slice(data).reshape([rows as i64, cols as i64])
}

fn burn1(data: &[f32]) -> BurnTensor<B, 1> {
    let device = Default::default();
    BurnTensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
}

fn burn2(data: &[f32], rows: usize, cols: usize) -> BurnTensor<B, 2> {
    let device = Default::default();
    BurnTensor::<B, 2>::from_data(TensorData::new(data.to_vec(), [rows, cols]), &device)
}

fn burn_actions(data: &[i64]) -> BurnTensor<B, 1, Int> {
    let device = Default::default();
    BurnTensor::<B, 1, Int>::from_data(TensorData::new(data.to_vec(), [data.len()]), &device)
}

#[test]
fn parity_ppo_policy_loss() {
    let log_probs_f = [0.0_f32, 0.5, -0.5];
    let old_log_probs_f = [0.0_f32, 0.0, 0.0];
    let advantages_f = [1.0_f32, -1.0, 0.5];
    let clip_range = 0.2;

    let (loss_tch, clip_tch, kl_tch) = crate::train::ppo::compute_policy_loss(
        &tch1(&log_probs_f),
        &tch1(&old_log_probs_f),
        &tch1(&advantages_f),
        clip_range,
    );
    let loss_tch_val = f64::try_from(&loss_tch).unwrap();

    let (loss_burn, clip_burn, kl_burn) = ppo_loss_burn::compute_policy_loss(
        burn1(&log_probs_f),
        burn1(&old_log_probs_f),
        burn1(&advantages_f),
        clip_range,
    );
    let loss_burn_val = ppo_loss_burn::scalar_f64(loss_burn);

    assert!(
        (loss_tch_val - loss_burn_val).abs() < PARITY_TOL,
        "policy_loss mismatch: tch={loss_tch_val} burn={loss_burn_val}"
    );
    assert!(
        (clip_tch - clip_burn).abs() < PARITY_TOL,
        "clip_fraction mismatch: tch={clip_tch} burn={clip_burn}"
    );
    assert!(
        (kl_tch - kl_burn).abs() < PARITY_TOL,
        "approx_kl mismatch: tch={kl_tch} burn={kl_burn}"
    );
}

#[test]
fn parity_ppo_value_loss_with_clip() {
    let values_f = [5.0_f32, 5.0, 5.0];
    let old_values_f = [0.0_f32, 0.0, 0.0];
    let returns_f = [0.0_f32, 0.0, 0.0];
    let clip_range_vf = 0.2;

    let (loss_tch, ev_tch) = crate::train::ppo::compute_value_loss(
        &tch1(&values_f),
        &tch1(&old_values_f),
        &tch1(&returns_f),
        clip_range_vf,
    );
    let loss_tch_val = f64::try_from(&loss_tch).unwrap();

    let (loss_burn, ev_burn) = ppo_loss_burn::compute_value_loss(
        burn1(&values_f),
        burn1(&old_values_f),
        burn1(&returns_f),
        clip_range_vf,
    );
    let loss_burn_val = ppo_loss_burn::scalar_f64(loss_burn);

    assert!(
        (loss_tch_val - loss_burn_val).abs() < PARITY_TOL,
        "value_loss mismatch: tch={loss_tch_val} burn={loss_burn_val}"
    );
    assert!(
        (ev_tch - ev_burn).abs() < PARITY_TOL,
        "explained_var mismatch: tch={ev_tch} burn={ev_burn}"
    );
}

#[test]
fn parity_ppo_value_loss_infinite_clip() {
    let values_f = [1.0_f32, 2.0, 0.5];
    let old_values_f = [1.0_f32, 1.5, 0.8];
    let returns_f = [1.2_f32, 2.1, 0.6];
    let clip_range_vf = f64::INFINITY;

    let (loss_tch, _) = crate::train::ppo::compute_value_loss(
        &tch1(&values_f),
        &tch1(&old_values_f),
        &tch1(&returns_f),
        clip_range_vf,
    );
    let loss_tch_val = f64::try_from(&loss_tch).unwrap();

    let (loss_burn, _) = ppo_loss_burn::compute_value_loss(
        burn1(&values_f),
        burn1(&old_values_f),
        burn1(&returns_f),
        clip_range_vf,
    );
    let loss_burn_val = ppo_loss_burn::scalar_f64(loss_burn);

    assert!(
        (loss_tch_val - loss_burn_val).abs() < PARITY_TOL,
        "infinite-clip value_loss mismatch: tch={loss_tch_val} burn={loss_burn_val}"
    );
}

#[test]
fn parity_ppo_entropy_loss() {
    let entropy_f = [0.5_f32, 1.0, 0.1];
    let loss_tch = crate::train::ppo::compute_entropy_loss(&tch1(&entropy_f));
    let loss_tch_val = f64::try_from(&loss_tch).unwrap();

    let loss_burn = ppo_loss_burn::compute_entropy_loss(burn1(&entropy_f));
    let loss_burn_val = ppo_loss_burn::scalar_f64(loss_burn);

    assert!(
        (loss_tch_val - loss_burn_val).abs() < PARITY_TOL,
        "entropy_loss mismatch: tch={loss_tch_val} burn={loss_burn_val}"
    );
}

#[test]
fn parity_dqn_td_target() {
    let rewards_f = [1.0_f32, 2.0];
    let dones_f = [0.0_f32, 1.0];
    let next_q_f = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    let target_tch = crate::train::dqn::compute_td_target(
        &tch1(&rewards_f),
        &tch1(&dones_f),
        &tch2(&next_q_f, 2, 3),
        0.9,
    );
    let tch_vec: Vec<f32> = Vec::try_from(target_tch).unwrap();

    let target_burn = dqn_loss_burn::compute_td_target(
        burn1(&rewards_f),
        burn1(&dones_f),
        burn2(&next_q_f, 2, 3),
        0.9,
    );
    let burn_vec: Vec<f32> = target_burn.into_data().to_vec().unwrap();

    assert_eq!(tch_vec.len(), burn_vec.len());
    for (t, b) in tch_vec.iter().zip(burn_vec.iter()) {
        assert!((*t as f64 - *b as f64).abs() < PARITY_TOL, "td_target {} vs {}", t, b);
    }
}

#[test]
fn parity_dqn_td_target_double() {
    let rewards_f = [1.0_f32, 0.5];
    let dones_f = [0.0_f32, 1.0];
    let q_online_f = [0.5_f32, 2.0, 1.0, 0.5];
    let q_target_f = [10.0_f32, 3.0, 7.0, 4.0];

    let target_tch = crate::train::dqn::compute_td_target_double(
        &tch1(&rewards_f),
        &tch1(&dones_f),
        &tch2(&q_online_f, 2, 2),
        &tch2(&q_target_f, 2, 2),
        0.9,
    );
    let tch_vec: Vec<f32> = Vec::try_from(target_tch).unwrap();

    let target_burn = dqn_loss_burn::compute_td_target_double(
        burn1(&rewards_f),
        burn1(&dones_f),
        burn2(&q_online_f, 2, 2),
        burn2(&q_target_f, 2, 2),
        0.9,
    );
    let burn_vec: Vec<f32> = target_burn.into_data().to_vec().unwrap();

    for (t, b) in tch_vec.iter().zip(burn_vec.iter()) {
        assert!((*t as f64 - *b as f64).abs() < PARITY_TOL, "double td_target {} vs {}", t, b);
    }
}

#[test]
fn parity_dqn_gather_action_q() {
    let q_f = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let actions_f = [0_i64, 1, 0];

    // tch path: actions tensor.
    let actions_tch = TchTensor::from_slice(&actions_f);
    let gathered_tch = crate::train::dqn::gather_action_q(&tch2(&q_f, 3, 2), &actions_tch);
    let tch_vec: Vec<f32> = Vec::try_from(gathered_tch).unwrap();

    let gathered_burn = dqn_loss_burn::gather_action_q(burn2(&q_f, 3, 2), burn_actions(&actions_f));
    let burn_vec: Vec<f32> = gathered_burn.into_data().to_vec().unwrap();

    for (t, b) in tch_vec.iter().zip(burn_vec.iter()) {
        assert!((*t as f64 - *b as f64).abs() < PARITY_TOL, "gather_action_q {} vs {}", t, b);
    }
}

#[test]
fn parity_dqn_compute_loss() {
    let q_f = [1.0_f32, 2.0, 3.0];
    let target_f = [1.1_f32, 1.5, 4.0];

    let loss_tch = crate::train::dqn::compute_loss(&tch1(&q_f), &tch1(&target_f));
    let loss_tch_val: f64 = f64::try_from(&loss_tch).unwrap();

    let loss_burn = dqn_loss_burn::compute_loss(burn1(&q_f), burn1(&target_f));
    let loss_burn_val: f64 = loss_burn.into_scalar() as f64;

    assert!(
        (loss_tch_val - loss_burn_val).abs() < PARITY_TOL,
        "dqn compute_loss mismatch: tch={loss_tch_val} burn={loss_burn_val}"
    );
}
