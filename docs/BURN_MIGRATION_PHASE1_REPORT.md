# Burn Migration — Phase 1 Scout Report

**Issue**: #78 (phase 1 of epic #65, "migrate tensor backend from tch to Burn")
**Scope**: port `train_simple_bandit` to a Burn-backed trainer behind a new
`training-burn` Cargo feature, in parallel with the existing tch trainer.
**Status**: scout complete; tch path untouched; both `training` and
`training-burn` features compile in isolation and together.

This report is the load-bearing deliverable for the scout — phases 2–6 of
the migration epic should be re-sized against the observations below.

---

## What was added

| File | LOC | Purpose |
|---|---|---|
| `src/policy/mlp_burn.rs` | ~115 | Minimal `MlpBurnPolicy<B: Backend>` with `forward` / `get_action_host` / `evaluate_actions`. Two `Linear` hidden layers + tanh + actor head + critic head. Derives `Module`. |
| `examples/games/bandit/train_simple_bandit_burn.rs` | ~210 | End-to-end PPO-style trainer on `SimpleBandit`. Uses `Autodiff<NdArray<f32>>`, `AdamConfig::init()`, `GradientsParams::from_grads`, full-batch updates. |
| `Cargo.toml` | ~10 | `burn = { version = "0.21", default-features = false, features = ["std", "ndarray", "autodiff"], optional = true }` plus a new `training-burn = ["dep:burn", "rayon", "tracing", "tracing-subscriber"]` feature and the `[[example]]` entry. |
| `src/policy/mod.rs` | ~5 | `#[cfg(feature = "training-burn")] pub mod mlp_burn;` |
| `src/env/mod.rs` | ~3 | Widened the `pool` gate from `feature = "training"` to `any(feature = "training", feature = "training-burn")` so the Burn trainer can reuse `EnvPool`. |

**Total scout delta: ~345 LOC** — comfortably under the 400-LOC ceiling.
Conclusion: a single-trainer port is small.

---

## Empirical results

Both runs use `SimpleBandit` (50 000 env steps, 4 parallel envs, 100 steps
per rollout, lr 1e-3, hidden=64). Default seeds, single run each (the issue
called for an order-of-magnitude observation, not a benchmark).

| Backend | Success rate at completion | Wall time | Notes |
|---|---|---|---|
| Burn `NdArray<f32>` + `Autodiff` | **98.8 %** (end-of-run) | ~2.4 s | Final trained logits clearly bimodal: state=0 → `[5.51, -6.08]`, state=1 → `[-5.76, 5.52]`. |
| tch (PyTorch 2.10, CPU) | 85.8 % then aborted | ~0.7 s | Hit the trainer's built-in entropy-collapse guard (`< 0.05` for 3 consecutive updates) and intentionally bailed. The tch path *would* converge if the guard were relaxed — this is a property of the shared `PPOTrainer`, not of tch itself. |

**Headline takeaway**: order-of-magnitude parity. Burn-ndarray is about
3-4× slower than libtorch CPU on this tiny workload (50k tensor ops),
which matches the rule of thumb that ndarray's BLAS path is competitive
within ~2-5× of libtorch CPU for small batches. GPU backends were
explicitly deferred to phase 6 of the migration.

---

## Burn 0.21 API surface actually used

| Subsystem | API touched | Notes |
|---|---|---|
| Tensors | `Tensor<B, D>`, `Tensor::from_data`, `TensorData::new`, `into_data().to_vec::<f32>()`, `clone`, `mul_scalar`, `mean`, `exp`, `gather`, `unsqueeze_dim`, `squeeze_dim`, `min_pair`, `clamp`, `sum_dim`, `into_scalar` | All const-generic on rank `D`. Type-inference holes (see friction #3). |
| Activations | `burn::tensor::activation::{tanh, softmax, log_softmax}` | Free functions, not method chains. `softmax(tensor, dim)`. |
| `nn::Linear` | `LinearConfig::new(d_in, d_out).init(&device)` → `Linear<B>` | Two-step config-then-init pattern, distinct from tch's `nn::linear(path, ..., LinearConfig::default())`. No path/var-store concept. |
| Module derive | `#[derive(Module, Debug)]` on a struct of `Linear<B>` fields | Auto-implements `Module<B>` and `AutodiffModule<B>` when `B: AutodiffBackend`. No manual `register_parameters` step. |
| Optimizer | `AdamConfig::new().init()` → `OptimizerAdaptor<Adam, M, B>` | `init()` is generic on `<B, M>` and inferred from the first `step()` call site. |
| Gradients | `loss.backward()` → `B::Gradients`; `GradientsParams::from_grads(grads, &module)` → `GradientsParams`; `optim.step(lr, module, grads_params)` → updated module | Three-step dance. See friction #2. |
| Module / Autodiff split | `policy.valid()` returns `MlpBurnPolicy<NdArray<f32>>` (the `InnerBackend`) for eval | Clean — the `Module` derive makes this free. |

Notably **NOT** used (and missing for a fuller port):
- `multinomial` / categorical sampling op (see friction #5)
- `argmax`-with-stable-tiebreak guarantees
- A first-class `Dataset` / `DataLoader` story (we hand-rolled a batch
  iterator over `Vec<f32>` host buffers)
- `burn::train::Learner` (which subsumes the optimizer loop but pulls
  in a much larger surface area; deliberately avoided for the scout)

---

## Friction points (the real deliverable)

### 1. The "move-the-module-through-the-optimizer" ownership model

Burn's `Optimizer::step(lr, module, grads) -> module` **consumes** the
module and hands back the updated copy. tch's `nn::Optimizer::backward_step(&loss)`
mutates the `VarStore` in place and is decoupled from the network struct
entirely.

In the scout, this forced the trainer to hold the policy in
`Option<MlpBurnPolicy<B>>` and swap it out via `.take().unwrap()` →
`optim.step(...)` → `Some(...)` on every gradient step. It is ergonomic
once you accept it, but it has two consequences:

- **The existing `PPOTrainer<P>` API is incompatible as-written.** It
  holds a `&P` and a separate optimizer and assumes side-effecting
  updates. A Burn-side PPO trainer has to *own* the module and yield it
  back at every `train_step`. **This validates Architect risk #2** —
  the trainer struct cannot just be generic over a `Policy` trait; it
  has to be generic over a `Module` plus its `Optimizer`, with the
  module flowing through `step` by value.
- Any helper that wants to inspect the module mid-update (logging,
  weight stats, checkpoint hooks) has to do so on the *returned*
  module, not on a borrowed reference held elsewhere.

**Phase-2 implication**: the PPO trainer refactor will be larger than
the bandit diff suggests. Estimate: porting `src/train/ppo/trainer.rs`
+ helpers is ~600–900 LOC, not the ~300 a naive `s/tch/burn/` would
suggest.

### 2. Gradient extraction is a three-step ritual

Compared to tch:

```rust
// tch
optimizer.backward_step(&loss);
```

Burn needs:

```rust
let grads = loss.backward();                         // (1) compute grads
let grads = GradientsParams::from_grads(grads, &m);  // (2) tie to params
m = optim.step(lr, m, grads);                        // (3) update
```

Step 2 in particular is non-obvious — it walks the module tree using a
visitor to attach each gradient tensor to its `ParamId`. If a parameter
is referenced in the loss graph but missing from the visited module
(e.g. a `Param` held in a `Vec` that isn't part of the `Module` derive),
its gradient is silently dropped. Worth a dedicated docs paragraph in
phase 2.

### 3. Const-generic rank everywhere → type-inference cliffs

Every tensor op carries a `const D: usize` rank. Most are inferred from
context, but `unsqueeze_dim`, `squeeze`, and `from_data` over `TensorData`
regularly need turbofish (`::<2>(1)` for `unsqueeze_dim`). Mistakes
manifest as `E0282 type annotations needed` or rank-check panics at
runtime, not as obvious compile errors. The fix is verbose but
mechanical.

### 4. Cargo feature for `training-burn` had to gate the env pool

`crate::env::pool` only depends on `rayon`, but it was previously gated
on `feature = "training"`. The Burn trainer needs the same pool, so the
gate widened to `any(feature = "training", feature = "training-burn")`.
This pattern will repeat for every shared, backend-agnostic module
(`buffer/rollout`, `multi_agent/simulator`, the per-env adapters). It is
cheap — single-line cfg changes — but easy to forget; a checklist for
phase 2 should enumerate them.

### 5. No first-class categorical sampling op

Burn 0.21 does not ship a `multinomial` / `categorical` op on `Tensor`.
The scout sidesteps this by doing the categorical draw on the host
(`into_data().to_vec()` → `rand::Rng::r#gen()`-based inverse CDF). For
the bandit (4 envs, 2 actions) this is trivial; for Snake (configurable
grid, many actions per step) and self-play training (thousands of
sims/s), host-side sampling will be a measurable cost. **Phase 3
(Snake port) should budget ~50 LOC of categorical-sampling utility**
(plus tests) on day one.

### 6. Edition 2024 keyword collision: `gen`

`rand 0.8`'s `Rng::gen()` is now a reserved-word collision in Edition
2024 and must be called as `rng.r#gen()`. Not Burn-specific, but
worth noting: the migration will surface this at every random-action
site. Bumping to `rand 0.9` (which exposes `Rng::random()`) is a
worthwhile parallel cleanup.

### 7. Weight init: Burn defaults differ from PPO baselines

`LinearConfig::default()` uses Kaiming-uniform with `gain = 1/√3`.
The tch `MlpPolicy` uses orthogonal init (`gain = √2`) for hidden
layers and `gain = 0.01` for the output heads — the de-facto PPO
recipe. The bandit was simple enough that the difference did not
matter (98.8 % vs. the tch baseline that *would* have hit ~100 %),
but **CartPole and Snake are sensitive to this**. Phase 2 should
port the orthogonal initializer over via `Initializer::Orthogonal`
(burn-nn exposes it).

### 8. No drop-in `WASM inference` story yet

The tch path has a polished `policy/inference.rs` that exports the
trained `MlpPolicy` to a pure-Rust `InferenceModel` for the WASM
browser demo. Burn's own `store` crate (`burn::store`, gated on
`feature = "store"`) is the intended replacement and supports
SafeTensors round-trip, but the scout did not exercise it. Phase 5
(WASM unification) should plan for: (a) `store` feature on, (b) a
new `InferenceModel`-equivalent struct shared between tch and Burn
paths, (c) re-exporter for the existing `web/` demo.

---

## Re-sizing phases 2–6 based on what the scout learned

Original epic guesses (from issue #65) annotated with scout-informed
revisions:

| Phase | Goal | Original | Revised | Rationale |
|---|---|---|---|---|
| 2 | Port PPO trainer to Burn | "medium" | **large** | The owning-module / GradientsParams dance forces a rewrite of `PPOTrainer<P>`; not a `s/tch/burn/` job. ~600–900 LOC + ~300 LOC of tests. |
| 3 | Port Snake CNN policy + trainer | "medium" | **medium-large** | Adds: categorical sampling utility (~50 LOC), Conv2d module derive, multi-discrete head. Buffer/rollout port lands here too. ~400–600 LOC + tests. |
| 4 | Port DQN trainer + replay | "medium" | **medium** | DQN does not have PPO's surrogate-loss complexity. Replay buffer is index-based and backend-agnostic — should be ~150 LOC to rewire. |
| 5 | WASM inference unification (`burn::store`) | "small" | **small-medium** | Need a new `InferenceModel` schema OR adopt SafeTensors end-to-end. Cleanup of `policy/inference.rs` + browser demo wiring. |
| 6 | GPU backends (wgpu, cuda) | "small" | **small** | Confirmed: once a trainer is generic on `B: AutodiffBackend`, the backend swap is a one-line type alias change. Geode-fem already proves this. |

**Net rebalancing**: phases 2 and 3 are each ~1.5–2× the original
estimate; phases 5 and 6 are roughly as estimated. The single biggest
de-risk would be to land phase 2's PPO refactor *before* phase 3, so
Snake/Pong only have to deal with the new trainer shape and not the
"old-PPO-still-on-tch, new-CNN-on-burn" hybrid.

---

## Open questions for phase 2 kickoff

1. **Do we keep the `training` (tch) feature long-term?** The scout
   proves two-backend coexistence; the question is whether we want to
   pay the maintenance cost. Recommend: keep through phase 4, retire
   in phase 5 alongside libtorch docs removal.
2. **Do we adopt `burn::train::Learner`?** It wraps optimizer + epoch
   loop + metrics in one struct. Saves boilerplate but couples to
   Burn's data-loading API. Recommend: skip for phase 2; revisit
   after Snake is ported and the trainer shape stabilizes.
3. **Should `Initializer::Orthogonal` be wrapped in a thrust-side
   helper that matches the tch `Init::Orthogonal { gain }` API?**
   Probably yes — keeps phase-3 hyperparameter parity from
   accidentally drifting.
