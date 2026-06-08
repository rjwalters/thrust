//! Backend-agnostic optimizer abstraction (phase 2b of the Burn migration,
//! #92).
//!
//! # Why this exists
//!
//! Phase 1 of the migration (#78, scout in
//! `docs/BURN_MIGRATION_PHASE1_REPORT.md`) confirmed that the **single largest
//! API friction** between tch and Burn is the optimizer-vs-module ownership
//! model:
//!
//! - **tch** (`tch::nn::Optimizer`) mutates the underlying `VarStore` *in
//!   place*. The optimizer holds a reference to the var-store at construction
//!   time and then the trainer calls `zero_grad()` / `backward()` on a loss
//!   tensor / `clip_grad_norm(max)` / `step()` as four independent
//!   side-effecting verbs. The policy struct is never handed back to anyone.
//! - **Burn** (`Optimizer<M, B>`) is move-through: `optim.step(lr, module,
//!   grads)` *consumes* the module and returns the updated copy. There is no
//!   in-place variant. The trainer must own the module and rebind it on every
//!   step.
//!
//! Phase 2b (this file) introduces a trait that hides this asymmetry so the
//! existing PPO and DQN trainers can hold the optimizer behind a generic type
//! without leaking either backend into the algorithm bodies. The actual loss
//! math stays on tch in this phase; phase 3 (#80) is where the algorithm
//! bodies themselves become backend-agnostic and start using the
//! `step_module` half of the trait.
//!
//! # Pattern choice
//!
//! Three patterns were on the table (see issue #92 for the full write-up):
//!
//! - **A: trait-based with an associated `Module` type** (what this file
//!   implements). One trainer struct, generic over `O: BackendOptimizer`, with
//!   a default of `TchOptimizer` so existing call sites compile unchanged.
//! - **B: PhantomData generic on the trainer struct.** Rejected: pushes a
//!   second type parameter onto every PPO/DQN call site, which compounds the
//!   type-inference cliffs called out in friction #3 of the scout report.
//! - **C: two sibling trainer structs (`PpoTrainerTch` / `PpoTrainerBurn`).**
//!   Rejected for this phase: PPO/DQN's non-optimizer state (counters, entropy
//!   guard, replay buffer book-keeping, KL early-stopping, etc.) is genuinely
//!   backend-agnostic. Duplicating ~700 LOC of trainer struct across two
//!   siblings just to vary the optimizer field would be more redundant than the
//!   asymmetry pattern A tolerates. If pattern A breaks down in phase 3 when
//!   the loss math goes generic, the issue calls out the fall-back to C
//!   explicitly — but as of phase 2b A holds.
//!
//! # The asymmetry, made concrete
//!
//! tch never flows the module through `step`, so on the tch impl the
//! associated `Module` type is `()` (a unit; the `step_module` call still
//! works, it just returns `()` and the trainer's own policy field is
//! unaffected — tch already mutated its `VarStore` in place during the
//! `step()` half). The Burn impl uses `Self::Module = M` and the trainer
//! does need to take its module field by value across the step.
//!
//! This is intentionally lopsided. Pattern A makes the trade-off "one
//! struct, two slightly-funny methods" instead of pattern C's "two structs,
//! two ergonomic methods each." The judgement is that this phase pays for
//! itself in the deletion phase (#5) — when the tch impl is dropped, this
//! whole file collapses to a single Burn-only struct with `Self::Module = M`
//! and the `()` arm vanishes.

use anyhow::Result;

/// Backend-agnostic optimizer interface used by PPO and DQN trainers.
///
/// Implementors expose two complementary halves:
///
/// 1. **The tch / side-effecting half**: `zero_grad`, `backward_tch`,
///    `clip_grad_norm`, `step`. Mirrors the four-verb ritual that the existing
///    tch trainers already run today. The Burn impl provides no-ops here (Burn
///    doesn't decompose the update that way).
/// 2. **The Burn / move-through half**: `step_module`. Consumes the current
///    module-by-value and returns the updated module. The tch impl returns `()`
///    (its `Module` associated type is `()`); the Burn impl actually flows `M`
///    through.
///
/// Phase 2b only wires the tch half into PPO/DQN. Phase 3 (#80) is where
/// the loss math goes generic and the Burn `step_module` path starts
/// getting called.
pub trait BackendOptimizer {
    /// The module type the optimizer steps. `()` for backends (tch) that
    /// mutate parameters in place via a held `VarStore`; the actual
    /// Burn `Module` type for Burn.
    type Module;

    /// Zero gradients accumulated in the optimizer's parameter buffers.
    ///
    /// On tch this calls `tch::nn::Optimizer::zero_grad`. On Burn this is
    /// a no-op: Burn's `OptimizerAdaptor::step` recomputes gradients from
    /// `GradientsParams` every call, so there's nothing to zero.
    fn zero_grad(&mut self);

    /// Run `loss.backward()` for backends where the loss tensor is a
    /// `tch::Tensor`. No-op for Burn (Burn produces gradients via
    /// `loss.backward()` returning a `Gradients`, not by mutating
    /// thread-local autograd state).
    ///
    /// This method exists so the existing tch trainer bodies can call
    /// `optim.backward_tch(&loss)` without naming `tch::Tensor` at the
    /// call site. Phase 3 will retire it in favour of a backend-agnostic
    /// `step_module(lr, module, loss_fn)` once the loss math goes generic.
    #[cfg(feature = "training")]
    fn backward_tch(&mut self, loss: &tch::Tensor);

    /// Clip the global gradient L2-norm to `max`.
    ///
    /// On tch this calls `tch::nn::Optimizer::clip_grad_norm`. On Burn,
    /// clipping happens via `GradientsParams` before the move-through
    /// step; this method records the cap and the Burn impl applies it
    /// inside `step_module`.
    fn clip_grad_norm(&mut self, max: f64);

    /// Apply the staged gradient update.
    ///
    /// tch: calls `tch::nn::Optimizer::step` (which reads the gradients
    /// that the previous `backward()` accumulated on the held `VarStore`
    /// and writes new parameter values back in place).
    ///
    /// Burn: no-op. The Burn path uses [`Self::step_module`] instead,
    /// which takes the module by value and returns the updated copy.
    fn step(&mut self);

    /// Burn-style move-through update.
    ///
    /// Consumes `module`, applies the optimizer's staged gradient (with
    /// any clipping configured by [`Self::clip_grad_norm`]), and returns
    /// the updated module. For tch (`Self::Module = ()`) this is a
    /// degenerate no-op that simply returns `()` — tch's actual update
    /// happens inside [`Self::step`] because tch mutates the `VarStore`
    /// in place.
    ///
    /// Phase 2b does not call this method from PPO/DQN — it exists so
    /// the trait surface is closed and so phase 3 can plug Burn into the
    /// loss math without re-shaping the trait.
    fn step_module(&mut self, module: Self::Module) -> Self::Module;

    /// Construction-time learning rate. tch records this on the held
    /// `tch::nn::Optimizer`; Burn records it on the impl struct for use
    /// inside `step_module`. Exposed for diagnostics only.
    fn learning_rate(&self) -> f64;
}

// ---------------------------------------------------------------------------
// tch impl
// ---------------------------------------------------------------------------

/// Wraps a [`tch::nn::Optimizer`] so it satisfies [`BackendOptimizer`].
///
/// All four side-effecting verbs delegate directly to the underlying tch
/// optimizer; the algorithm bodies in PPO/DQN look exactly like the
/// pre-phase-2b code modulo the indirection through this wrapper. The
/// associated `Module` type is `()` because tch mutates its `VarStore` in
/// place — there is no module to flow back to the caller.
///
/// # Construction
///
/// Build the inner `tch::nn::Optimizer` exactly as before (e.g.
/// `policy.optimizer(lr)`), then wrap it with [`TchOptimizer::new`]. The
/// existing `PPOTrainer::set_optimizer` / `DQNTrainer::new` constructors
/// retain a `tch::nn::Optimizer` argument for source-compat and wrap it
/// internally; callers do not need to change.
#[cfg(feature = "training")]
pub struct TchOptimizer {
    inner: tch::nn::Optimizer,
    learning_rate: f64,
}

#[cfg(feature = "training")]
impl std::fmt::Debug for TchOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TchOptimizer")
            .field("learning_rate", &self.learning_rate)
            .field("inner", &"tch::nn::Optimizer")
            .finish()
    }
}

#[cfg(feature = "training")]
impl TchOptimizer {
    /// Wrap an existing `tch::nn::Optimizer`.
    ///
    /// `learning_rate` should match whatever was passed to the tch
    /// optimizer constructor; it is stored for diagnostics only.
    pub fn new(inner: tch::nn::Optimizer, learning_rate: f64) -> Self {
        Self { inner, learning_rate }
    }

    /// Borrow the wrapped tch optimizer. Escape hatch for code that
    /// genuinely needs the concrete type (rare; the trainer bodies don't).
    pub fn inner(&self) -> &tch::nn::Optimizer {
        &self.inner
    }

    /// Mutably borrow the wrapped tch optimizer.
    pub fn inner_mut(&mut self) -> &mut tch::nn::Optimizer {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner `tch::nn::Optimizer`.
    pub fn into_inner(self) -> tch::nn::Optimizer {
        self.inner
    }
}

#[cfg(feature = "training")]
impl BackendOptimizer for TchOptimizer {
    type Module = ();

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn backward_tch(&mut self, loss: &tch::Tensor) {
        loss.backward();
    }

    fn clip_grad_norm(&mut self, max: f64) {
        self.inner.clip_grad_norm(max);
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn step_module(&mut self, _module: Self::Module) -> Self::Module {
        // tch already mutated its VarStore in `step()`. There's nothing
        // to flow through — `Self::Module = ()` makes that explicit.
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

// ---------------------------------------------------------------------------
// Burn impl
// ---------------------------------------------------------------------------

/// Burn-side optimizer wrapper.
///
/// Wraps a Burn `OptimizerAdaptor<O, M, B>` and exposes it through the
/// [`BackendOptimizer`] trait. The tch-side side-effecting verbs are all
/// no-ops; the real update happens inside [`BackendOptimizer::step_module`],
/// which consumes the module and returns the updated copy.
///
/// # Phase 2b status
///
/// This impl exists so the trait is provably closed over both backends
/// and so the unit tests called out in issue #92 can construct it. The
/// PPO / DQN trainer bodies do not call into it yet — that wiring lands
/// in phase 3 (#80) along with the backend-agnostic loss math.
///
/// # Gradient flow
///
/// Phase 3 will use `step_module` as follows:
///
/// ```text
/// let grads = loss.backward();
/// let grads_params = burn::optim::GradientsParams::from_grads(grads, &module);
/// // (optional) clip grads_params here using stored `clip_grad_norm`.
/// let module = optimizer.step_module_with_grads(lr, module, grads_params);
/// ```
///
/// The current `step_module` signature on the trait only takes the
/// module; phase 3 will widen it to thread the `GradientsParams` through
/// (or evolve `BurnOptimizer` to take the loss closure directly). The
/// scaffolding here is the seam.
#[cfg(feature = "training-burn")]
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

#[cfg(feature = "training-burn")]
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

#[cfg(feature = "training-burn")]
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

    /// Mutably borrow the wrapped Burn optimizer. Phase 3 calls
    /// `inner_mut().step(lr, module, grads)` directly from the loss
    /// closure.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// The currently-staged gradient-norm cap, if any.
    pub fn grad_clip_norm(&self) -> Option<f64> {
        self.grad_clip_norm
    }
}

#[cfg(feature = "training-burn")]
impl<B, M, O> BackendOptimizer for BurnOptimizer<B, M, O>
where
    B: burn::tensor::backend::AutodiffBackend,
    M: burn::module::AutodiffModule<B>,
    O: burn::optim::Optimizer<M, B>,
{
    type Module = M;

    fn zero_grad(&mut self) {
        // Burn rebuilds `Gradients` per step from `loss.backward()`;
        // there is no separate zero step. Intentionally a no-op.
    }

    #[cfg(feature = "training")]
    fn backward_tch(&mut self, _loss: &tch::Tensor) {
        // Burn doesn't operate on tch tensors. Phase 2b never calls this
        // through the Burn impl; it exists to keep the trait surface
        // single. Intentionally a no-op; phase 3 will retire it.
    }

    fn clip_grad_norm(&mut self, max: f64) {
        self.grad_clip_norm = Some(max);
    }

    fn step(&mut self) {
        // No in-place step in Burn — see `step_module`.
    }

    fn step_module(&mut self, module: Self::Module) -> Self::Module {
        // Phase 2b stub: phase 3 widens this signature (or grows a
        // companion method) to thread `GradientsParams` through. Today
        // there is nothing to step against, so we hand the module back
        // unchanged.
        module
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Helper: wrap a freshly-built `tch::nn::Optimizer` in a [`TchOptimizer`].
///
/// Saves call sites from importing `TchOptimizer` directly when they only
/// need to feed a trainer constructor.
#[cfg(feature = "training")]
pub fn wrap_tch(inner: tch::nn::Optimizer, learning_rate: f64) -> TchOptimizer {
    TchOptimizer::new(inner, learning_rate)
}

/// Helper: wrap a freshly-built Burn optimizer in a [`BurnOptimizer`].
#[cfg(feature = "training-burn")]
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

    /// Verify the tch impl satisfies the trait and the four-verb ritual
    /// is callable without panicking. We don't run a real gradient step
    /// here — the trainer-level tests in `dqn::trainer::tests` and
    /// `ppo::trainer::tests` exercise the end-to-end path.
    #[cfg(feature = "training")]
    #[test]
    fn tch_optimizer_satisfies_trait() {
        use tch::{Tensor, nn::OptimizerConfig};

        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        // Need at least one trainable variable for tch to be willing to
        // build an optimizer.
        let w = vs.root().var("w", &[1], tch::nn::Init::Const(0.0));
        let tch_opt = tch::nn::Adam::default().build(&vs, 1e-3).unwrap();
        let mut opt = TchOptimizer::new(tch_opt, 1e-3);

        // Run a minimal forward / backward cycle so the optimizer has
        // gradients to clip and step against. The math doesn't matter —
        // we only need a finite loss tensor with `requires_grad`.
        opt.zero_grad();
        let target = Tensor::from_slice(&[1.0_f32]);
        let loss = (&w - &target).pow_tensor_scalar(2.0).sum(tch::Kind::Float);
        opt.backward_tch(&loss);
        opt.clip_grad_norm(0.5);
        opt.step();
        // step_module on tch is a no-op returning ().
        let unit: () = opt.step_module(());
        assert_eq!(unit, ());
        assert!((opt.learning_rate() - 1e-3).abs() < 1e-12);
    }

    /// Verify the Burn impl satisfies the trait — construction-only
    /// smoke test, per the DoD on issue #92. Phase 3 will add the
    /// end-to-end gradient flow.
    #[cfg(feature = "training-burn")]
    #[test]
    fn burn_optimizer_satisfies_trait() {
        use burn::{
            backend::{Autodiff, NdArray},
            optim::AdamConfig,
        };

        type B = Autodiff<NdArray<f32>>;

        let device = Default::default();
        let module = crate::policy::mlp_burn::MlpBurnPolicy::<B>::new(2, 2, 4, &device);
        let inner_opt = AdamConfig::new().init();
        let mut opt: BurnOptimizer<B, crate::policy::mlp_burn::MlpBurnPolicy<B>, _> =
            BurnOptimizer::new(inner_opt, 1e-3);

        opt.zero_grad();
        opt.clip_grad_norm(0.5);
        assert_eq!(opt.grad_clip_norm(), Some(0.5));
        opt.step();

        // step_module flows the module by value and (in phase 2b) hands
        // it back unchanged. We only need to prove the call type-checks
        // and round-trips; the no-op semantics are intentional.
        let module = opt.step_module(module);
        // Use the returned module so the compiler doesn't optimize the
        // call away — read a field.
        let _ = module;
        assert!((opt.learning_rate() - 1e-3).abs() < 1e-12);
    }
}
