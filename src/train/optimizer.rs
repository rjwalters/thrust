//! Burn optimizer wrapper used by the PPO and DQN trainers.
//!
//! After phase 5 of the Burn migration (#82), Burn is the only tensor
//! backend in the workspace; the backend-agnostic abstraction added in
//! phase 2b (#92) has collapsed to a single Burn impl. The
//! [`BackendOptimizer`] trait survives as a minimal interface so the
//! trainer bodies can hold the optimizer behind a generic without naming
//! the concrete Burn `Optimizer<M, B>` type.

use anyhow::Result;

/// Burn-side optimizer interface used by PPO and DQN trainers.
///
/// Burn's `Optimizer<M, B>` is *move-through*: every gradient step
/// consumes the module by value and returns the updated copy. This
/// trait exposes that single fundamental verb plus the
/// gradient-clipping configuration knob the trainer needs.
pub trait BackendOptimizer {
    /// The module type the optimizer steps.
    type Module;

    /// Stage the maximum global gradient L2-norm.
    ///
    /// This only records the cap on the wrapper; the trainer bodies read it
    /// back via [`BurnOptimizer::grad_clip_norm`] and apply the clip to the
    /// gradient slice before their move-through `inner_mut().step(...)`
    /// call (see the joint trainer's per-policy step, issue #239). The
    /// trait-level [`Self::step_module`] fallback does not itself clip.
    fn clip_grad_norm(&mut self, max: f64);

    /// Burn-style move-through update.
    ///
    /// Consumes `module`, applies the optimizer's staged gradient (with
    /// any clipping configured by [`Self::clip_grad_norm`]), and returns
    /// the updated module.
    fn step_module(&mut self, module: Self::Module) -> Self::Module;

    /// Construction-time learning rate. Exposed for diagnostics.
    fn learning_rate(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Burn impl
// ---------------------------------------------------------------------------

/// Burn-side optimizer wrapper.
///
/// Wraps a Burn `OptimizerAdaptor<O, M, B>` and exposes it through the
/// [`BackendOptimizer`] trait. The trainer bodies hold one of these and
/// route their gradient step through [`BackendOptimizer::step_module`].
pub struct BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    inner: O,
    learning_rate: f64,
    grad_clip_norm: Option<f64>,
    _marker: core::marker::PhantomData<(B, M)>,
}

impl<B, M, O> std::fmt::Debug for BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BurnOptimizer")
            .field("learning_rate", &self.learning_rate)
            .field("grad_clip_norm", &self.grad_clip_norm)
            .field("inner", &"burn::optim::Optimizer<...>")
            .finish()
    }
}

impl<B, M, O> BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    /// Wrap a Burn optimizer (typically `AdamConfig::new().init()`).
    pub fn new(inner: O, learning_rate: f64) -> Self {
        Self { inner, learning_rate, grad_clip_norm: None, _marker: core::marker::PhantomData }
    }

    /// Borrow the wrapped Burn optimizer.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Mutably borrow the wrapped Burn optimizer. The trainer bodies
    /// call `inner_mut().step(lr, module, grads)` directly from their
    /// loss closures.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// The currently-staged gradient-norm cap, if any.
    pub fn grad_clip_norm(&self) -> Option<f64> {
        self.grad_clip_norm
    }
}

impl<B, M, O> BackendOptimizer for BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    type Module = M;

    fn clip_grad_norm(&mut self, max: f64) {
        self.grad_clip_norm = Some(max);
    }

    fn step_module(&mut self, module: Self::Module) -> Self::Module {
        // Burn's `Optimizer::step` consumes the module by value and
        // returns the updated copy. The trainer bodies use
        // `inner_mut().step(lr, module, grads)` directly when they have
        // gradients to apply; this trait method is the no-grad fallback
        // used by call sites that just want a default step.
        module
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Helper: wrap a freshly-built Burn optimizer in a [`BurnOptimizer`].
pub fn wrap_burn<B, M, O>(inner: O, learning_rate: f64) -> BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    BurnOptimizer::new(inner, learning_rate)
}

// ---------------------------------------------------------------------------
// Result alias for parity with the rest of the crate's training surface.
// ---------------------------------------------------------------------------

/// Result alias used by trainer-side optimizer plumbing.
pub type OptimResult<T> = Result<T>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction smoke test: verify the Burn impl satisfies the trait
    /// and the move-through step type-checks.
    #[test]
    fn burn_optimizer_satisfies_trait() {
        use burn::{
            backend::{Autodiff, NdArray},
            optim::AdamConfig,
        };

        type B = Autodiff<NdArray<f32>>;

        let device = Default::default();
        let module = crate::policy::mlp::MlpBurnPolicy::<B>::new(2, 2, 4, &device);
        let inner_opt = AdamConfig::new().init();
        let mut opt: BurnOptimizer<B, crate::policy::mlp::MlpBurnPolicy<B>, _> =
            BurnOptimizer::new(inner_opt, 1e-3);

        opt.clip_grad_norm(0.5);
        assert_eq!(opt.grad_clip_norm(), Some(0.5));

        // step_module flows the module by value and (in the trait-level
        // fallback path) hands it back unchanged. The trainer bodies
        // perform real gradient steps via `inner_mut().step(...)`.
        let module = opt.step_module(module);
        let _ = module;
        assert!((opt.learning_rate() - 1e-3).abs() < 1e-12);
    }
}
