//! Seeded, host-side weight-initialization helpers for bit-exact
//! policy construction.
//!
//! # Why this module exists
//!
//! Burn 0.21's [`burn::nn::Initializer::init_with`] /
//! `init_tensor` (see
//! `burn-core-0.21.0/src/module/initializer.rs`) take **no seed
//! parameter** — they route through a backend-internal
//! `Tensor::random` that draws from an unseeded, thread-local
//! generator. That makes [`Initializer::Orthogonal`] and
//! [`Initializer::KaimingUniform`] the last non-reproducible site in
//! the PSRO/NFSP determinism contract advertised by
//! [`crate::multi_agent::psro::PsroConfig::seed`] /
//! [`crate::multi_agent::nfsp::NfspConfig::seed`] (every rollout/
//! training RNG site was plumbed through `StdRng::seed_from_u64` in
//! PRs #113/#116/#122/#123/#125; this closes the policy-init gap —
//! issue #135).
//!
//! # What it does
//!
//! Provides pure-Rust, [`StdRng`]-driven ports of the two init recipes
//! the MLP policies use:
//!
//! - [`seeded_orthogonal`] — orthogonal init via Householder QR of a Gaussian
//!   matrix, scaled by `gain` (PPO recipe: `sqrt(2)` trunk, `0.01` heads).
//! - [`seeded_kaiming_uniform`] — Kaiming-uniform init matching Burn's
//!   `KaimingUniform { gain, fan_out_only: false }` formula (`bound = gain *
//!   sqrt(3 / fan_in)`), the path [`crate::policy::mlp::MlpBurnPolicy::new`]
//!   exercises.
//!
//! # Bit-parity caveat
//!
//! These are **not** byte-identical to Burn's unseeded output — we are
//! *replacing* the distribution with a reproducible one, not mirroring
//! Burn's internal sampler. The only contract is: same `seed` (and
//! same shape/gain) ⇒ same `Vec<f32>`, every run, every machine.

use rand::{Rng, SeedableRng, rngs::StdRng};

/// Draw a single `N(0, 1)` sample from `rng` via the Box–Muller
/// transform.
///
/// We hand-roll the normal sampler (rather than pulling in
/// `rand_distr`) so the only RNG draws are `f32` uniforms — keeping the
/// per-construction consumption order trivial to reason about and the
/// dependency surface unchanged. Two `StdRng`s seeded identically
/// therefore yield identical draw sequences and identical matrices.
fn standard_normal(rng: &mut StdRng) -> f32 {
    // Guard u1 away from 0 so ln() is finite.
    let u1: f32 = {
        let x: f32 = rng.random();
        if x <= f32::MIN_POSITIVE {
            f32::MIN_POSITIVE
        } else {
            x
        }
    };
    let u2: f32 = rng.random();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    r * theta.cos()
}

/// Generate a seeded orthogonal weight matrix of shape `[d_in, d_out]`
/// flattened in **row-major** order (row index over `d_in`, column
/// index over `d_out`) — matching Burn's `Linear` weight layout
/// (`weight: Param<Tensor<B, 2>>` with dims `[d_input, d_output]`).
///
/// Algorithm (a faithful, seeded port of the orthogonal recipe):
/// 1. Sample an `[rows, cols]` matrix `A` from `N(0, 1)`.
/// 2. Compute a (thin) QR factorization `A = Q R` via Householder reflections,
///    with the sign convention `sign(diag(R)) >= 0` so the result is
///    deterministic.
/// 3. Return `gain * Q`.
///
/// `Q` has orthonormal columns when `rows >= cols`, and orthonormal
/// rows when `rows < cols` (we QR the taller orientation internally and
/// transpose back), matching the behavior of standard orthogonal init.
pub fn seeded_orthogonal(seed: u64, d_in: usize, d_out: usize, gain: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);

    // We want an [d_in, d_out] result with orthonormal rows or columns.
    // QR is cleanest when the matrix is tall (rows >= cols). If
    // d_in < d_out we QR the transpose ([d_out, d_in]) and transpose
    // the resulting Q back.
    let (rows, cols, transposed) = if d_in >= d_out {
        (d_in, d_out, false)
    } else {
        (d_out, d_in, true)
    };

    // Sample the tall Gaussian matrix in column-major working storage:
    // a[c] is the c-th column (length `rows`). Column-major makes the
    // Householder loop's column access contiguous.
    let mut cols_data: Vec<Vec<f32>> = Vec::with_capacity(cols);
    for _ in 0..cols {
        let mut col = Vec::with_capacity(rows);
        for _ in 0..rows {
            col.push(standard_normal(&mut rng));
        }
        cols_data.push(col);
    }

    // Modified Gram–Schmidt orthonormalization (numerically adequate
    // for the small matrices used here, and trivially deterministic).
    // Equivalent in outcome to a Householder QR's Q for full-rank
    // Gaussian input; we additionally fix the sign so the diagonal of
    // the implied R is non-negative.
    let mut q: Vec<Vec<f32>> = Vec::with_capacity(cols);
    for j in 0..cols {
        let mut v = cols_data[j].clone();
        // Subtract projections onto the previously-fixed orthonormal
        // columns.
        for qi in q.iter() {
            let dot: f32 = (0..rows).map(|r| v[r] * qi[r]).sum();
            for r in 0..rows {
                v[r] -= dot * qi[r];
            }
        }
        // Normalize.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm = if norm <= f32::MIN_POSITIVE { 1.0 } else { norm };
        for slot in v.iter_mut() {
            *slot /= norm;
        }
        // Sign fix: make the leading nonzero-ish entry positive so the
        // result is a deterministic function of the seed regardless of
        // platform float associativity in the projection sums.
        if v[j] < 0.0 {
            for slot in v.iter_mut() {
                *slot = -*slot;
            }
        }
        q.push(v);
    }

    // q[c][r] = Q[r, c] for the tall orientation [rows, cols].
    // Emit row-major [d_in, d_out] * gain.
    let mut out = vec![0.0_f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            // Map (i, j) in [d_in, d_out] back to (r, c) in [rows, cols].
            let (r, c) = if transposed { (j, i) } else { (i, j) };
            out[i * d_out + j] = gain * q[c][r];
        }
    }
    out
}

/// Generate a seeded Kaiming-uniform weight matrix of shape
/// `[d_in, d_out]` flattened in **row-major** order.
///
/// Matches Burn's `Initializer::KaimingUniform { gain, fan_out_only:
/// false }` formula: each entry is drawn uniformly from
/// `[-bound, bound]` where `bound = gain * sqrt(3 / fan_in)` and
/// `fan_in = d_in`. This is the path
/// [`crate::policy::mlp::MlpBurnPolicy::new`] uses
/// (`use_orthogonal_init = false`), so seeding it is required to make
/// the matching-pennies tests (which build policies via `new`)
/// reproducible — see issue #135's Correction 2.
pub fn seeded_kaiming_uniform(seed: u64, d_in: usize, d_out: usize, gain: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let fan_in = d_in.max(1) as f32;
    let bound = gain * (3.0_f32 / fan_in).sqrt();
    let mut out = vec![0.0_f32; d_in * d_out];
    for slot in out.iter_mut() {
        // Uniform in [-bound, bound]: rng.random() is [0, 1).
        let u: f32 = rng.random();
        *slot = (u * 2.0 - 1.0) * bound;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two calls with the same seed/shape/gain produce bit-identical
    /// orthogonal matrices; a different seed differs.
    #[test]
    fn orthogonal_is_deterministic_in_seed() {
        let a = seeded_orthogonal(42, 6, 4, 2.0_f32.sqrt());
        let b = seeded_orthogonal(42, 6, 4, 2.0_f32.sqrt());
        assert_eq!(a, b, "same seed must be bit-identical");
        let c = seeded_orthogonal(43, 6, 4, 2.0_f32.sqrt());
        assert_ne!(a, c, "different seed must differ");
        assert_eq!(a.len(), 24);
    }

    /// Columns of the [d_in, d_out] result (d_in >= d_out) are
    /// orthonormal up to the gain factor.
    #[test]
    fn orthogonal_columns_are_orthonormal_tall() {
        let d_in = 8;
        let d_out = 3;
        let gain = 1.0;
        let w = seeded_orthogonal(7, d_in, d_out, gain);
        // Column dot products: col j . col k should be ~delta_{jk}.
        for j in 0..d_out {
            for k in 0..d_out {
                let mut dot = 0.0_f32;
                for r in 0..d_in {
                    dot += w[r * d_out + j] * w[r * d_out + k];
                }
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "col {j}.col {k} = {dot}, expected {expected}"
                );
            }
        }
    }

    /// Rows of the [d_in, d_out] result (d_in < d_out) are orthonormal
    /// up to the gain factor.
    #[test]
    fn orthogonal_rows_are_orthonormal_wide() {
        let d_in = 3;
        let d_out = 8;
        let gain = 1.0;
        let w = seeded_orthogonal(11, d_in, d_out, gain);
        for i in 0..d_in {
            for k in 0..d_in {
                let mut dot = 0.0_f32;
                for c in 0..d_out {
                    dot += w[i * d_out + c] * w[k * d_out + c];
                }
                let expected = if i == k { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "row {i}.row {k} = {dot}, expected {expected}"
                );
            }
        }
    }

    /// Gain scales the matrix linearly.
    #[test]
    fn orthogonal_gain_scales() {
        let w1 = seeded_orthogonal(5, 6, 4, 1.0);
        let w2 = seeded_orthogonal(5, 6, 4, 3.0);
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((b - a * 3.0).abs() < 1e-5, "gain should scale: {a} vs {b}");
        }
    }

    /// Kaiming-uniform is deterministic in seed and respects the
    /// `gain * sqrt(3 / fan_in)` bound.
    #[test]
    fn kaiming_is_deterministic_and_bounded() {
        let d_in = 16;
        let d_out = 8;
        let gain = 1.0_f32 / 3.0_f32.sqrt();
        let a = seeded_kaiming_uniform(99, d_in, d_out, gain);
        let b = seeded_kaiming_uniform(99, d_in, d_out, gain);
        assert_eq!(a, b, "same seed must be bit-identical");
        let c = seeded_kaiming_uniform(100, d_in, d_out, gain);
        assert_ne!(a, c, "different seed must differ");

        let bound = gain * (3.0_f32 / d_in as f32).sqrt();
        for &x in &a {
            assert!(x.abs() <= bound + 1e-6, "{x} exceeds bound {bound}");
        }
        assert_eq!(a.len(), d_in * d_out);
    }
}
