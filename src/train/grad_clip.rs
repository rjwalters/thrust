//! Global gradient-norm clipping shared by the PPO trainers and the joint
//! multi-agent trainer.
//!
//! Originally implemented inside the joint multi-agent trainer (issue #239)
//! and extracted here when the single-agent PPO trainers gained
//! `PPOConfig::max_grad_norm` support (issue #299), so all three call sites
//! share one definition of the clip.

use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{Tensor, backend::AutodiffBackend},
};

use crate::train::ppo::loss::scalar_f64;

/// Burn stores gradient tensors on the *inner* (non-autodiff) backend, so we
/// fetch / re-register them as `Tensor<B::InnerBackend, D>`.
type Inner<B> = <B as AutodiffBackend>::InnerBackend;

/// Global L2 norm of a gradient set over `module`'s parameters.
///
/// Computes `sqrt(Σ_p ‖g_p‖²)` across every float parameter `p` of `module`
/// that has a gradient registered in `grads`. Parameters without a gradient
/// contribute zero.
pub fn global_grad_norm<B, M>(module: &M, grads: &GradientsParams) -> f64
where
    B: AutodiffBackend,
    M: Module<B>,
{
    struct NormAccumulator<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f64,
        _marker: core::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormAccumulator<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(g) = self.grads.get::<Inner<B>, D>(param.id) {
                let sq = g.powf_scalar(2.0).sum();
                self.sum_sq += scalar_f64(sq);
            }
        }
    }
    let mut acc = NormAccumulator::<B> { grads, sum_sq: 0.0, _marker: core::marker::PhantomData };
    module.visit(&mut acc);
    acc.sum_sq.sqrt()
}

/// Clip a module's gradients to a global L2 norm (issues #239, #299).
///
/// This is the standard PPO/A2C "global gradient norm clip": compute the
/// L2 norm of the **concatenation** of every parameter's gradient, and if
/// it exceeds `max_norm`, scale *all* gradients by
/// `max_norm / (total_norm + eps)`. Unlike Burn's built-in
/// `GradientClipping::Norm` (which clips each parameter tensor
/// independently), this preserves the *direction* of the joint gradient —
/// which is exactly what we need so a large shared-trunk value-loss
/// gradient is tamed without distorting the relative scale of the policy
/// and value contributions.
///
/// Implemented with two module visitor passes:
/// 1. accumulate `Σ ‖g_p‖²` across all parameters `p` (see
///    [`global_grad_norm`]),
/// 2. if the resulting norm exceeds `max_norm`, multiply every gradient by the
///    clip coefficient and re-register it.
///
/// A non-finite or non-positive total norm leaves the gradients untouched.
pub fn clip_grads_by_global_norm<B, M>(
    module: &M,
    grads: GradientsParams,
    max_norm: f32,
) -> GradientsParams
where
    B: AutodiffBackend,
    M: Module<B>,
{
    let total_norm = global_grad_norm::<B, M>(module, &grads);
    if !total_norm.is_finite() || total_norm <= max_norm as f64 || max_norm <= 0.0 {
        // Already within the cap (or degenerate) — nothing to scale.
        return grads;
    }
    let clip_coef = (max_norm as f64 / (total_norm + 1e-6)) as f32;

    // Pass 2: scale every gradient by the clip coefficient.
    struct Scaler<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        coef: f32,
        _marker: core::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Scaler<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(g) = self.grads.remove::<Inner<B>, D>(param.id) {
                self.grads.register::<Inner<B>, D>(param.id, g.mul_scalar(self.coef));
            }
        }
    }
    let mut grads = grads;
    let mut scaler =
        Scaler::<B> { grads: &mut grads, coef: clip_coef, _marker: core::marker::PhantomData };
    module.visit(&mut scaler);
    grads
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
        optim::GradientsParams,
        tensor::Tensor,
    };

    use super::*;
    use crate::{multi_agent::JointPolicy as _, policy::mlp::MlpBurnPolicy};

    type B = Autodiff<NdArray<f32>>;

    /// Issues #239/#299: the global-norm clip must (a) leave an already-small
    /// gradient untouched and (b) scale an oversized gradient down so its
    /// global L2 norm equals the cap.
    #[test]
    fn test_clip_grads_by_global_norm() {
        let device: NdArrayDevice = Default::default();
        let policy = MlpBurnPolicy::<B>::new(4, 3, 16, &device);

        // Build a real gradient via a forward/backward so the param ids in
        // `GradientsParams` line up with the module's params.
        let obs = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(vec![0.1_f32; 4 * 8], [8, 4]),
            &device,
        );
        let logits = policy.encoder_features_joint(obs);
        let loss = logits.powf_scalar(2.0).sum();
        let grads = GradientsParams::from_grads(loss.backward(), &policy);

        let norm_before = global_grad_norm::<B, MlpBurnPolicy<B>>(&policy, &grads);
        assert!(norm_before.is_finite() && norm_before > 0.0);

        // (a) A cap well above the gradient norm is a no-op.
        let big_cap = norm_before * 10.0;
        let unclipped =
            clip_grads_by_global_norm::<B, MlpBurnPolicy<B>>(&policy, grads, big_cap as f32);
        let norm_unclipped = global_grad_norm::<B, MlpBurnPolicy<B>>(&policy, &unclipped);
        assert!(
            (norm_unclipped - norm_before).abs() < 1e-4,
            "cap above norm must not change gradients: {norm_before} -> {norm_unclipped}"
        );

        // (b) A small cap scales the global norm down to (approximately) the
        // cap.
        let small_cap = (norm_before / 4.0) as f32;
        let clipped =
            clip_grads_by_global_norm::<B, MlpBurnPolicy<B>>(&policy, unclipped, small_cap);
        let norm_clipped = global_grad_norm::<B, MlpBurnPolicy<B>>(&policy, &clipped);
        assert!(
            (norm_clipped - small_cap as f64).abs() < 1e-3 * small_cap as f64 + 1e-4,
            "clipped global norm {norm_clipped} should equal cap {small_cap}"
        );
    }
}
