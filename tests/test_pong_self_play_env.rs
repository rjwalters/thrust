//! Integration tests for the self-play additions to `Pong`
//! (`step_two` + `mirror_observation`).
//!
//! These mirror the in-file unit tests so the new API is exercised through
//! the public crate boundary. They are also runnable without compiling the
//! pre-existing in-lib `#[cfg(test)]` modules, which have unrelated breakage
//! at the time of writing (see also the corresponding inline tests in
//! `src/env/games/pong.rs`).

use thrust_rl::env::{
    Environment,
    pong::{Pong, mirror_observation},
};

#[test]
fn step_two_stay_stay_freezes_paddles_integration() {
    let mut p = Pong::new();
    let obs0 = p.get_observation();
    let left_y0 = obs0[4];
    let right_y0 = obs0[5];
    let ball_x0 = obs0[0];

    for _ in 0..10 {
        let _ = p.step_two(1, 1);
    }

    let obs1 = p.get_observation();
    assert_eq!(obs1[4], left_y0, "Left paddle moved despite stay action");
    assert_eq!(obs1[5], right_y0, "Right paddle moved — heuristic was not bypassed");
    assert_ne!(obs1[0], ball_x0, "Ball did not move during step_two");
}

#[test]
fn step_two_drives_right_paddle_independently_integration() {
    let mut p = Pong::new();
    let right_y0 = p.get_observation()[5];

    let _ = p.step_two(1, 0);
    let right_y1 = p.get_observation()[5];
    assert!(right_y1 < right_y0, "Right paddle did not move up with action=0");

    let _ = p.step_two(1, 2);
    let right_y2 = p.get_observation()[5];
    assert!(right_y2 > right_y1, "Right paddle did not move down with action=2");
}

#[test]
fn mirror_observation_round_trip_integration() {
    let p = Pong::new();
    let obs = p.get_observation();
    let mirrored = mirror_observation(&obs);
    let unmirrored = mirror_observation(&mirrored);
    for (a, b) in obs.iter().zip(unmirrored.iter()) {
        assert!((a - b).abs() < 1e-7, "round-trip mismatch: {a} vs {b}");
    }
}

#[test]
fn mirror_observation_semantics_integration() {
    let obs = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let m = mirror_observation(&obs);
    assert_eq!(m, vec![-0.3, 0.4, -0.5, 0.6, 0.8, 0.7]);
}
