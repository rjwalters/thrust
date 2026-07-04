# Mixed-precision (FP16) training — feasibility spike

Feasibility note for epic [#267](https://github.com/rjwalters/thrust/issues/267)
("Mixed-precision (FP16) training"). This is the **spike deliverable**, not an
implementation: it answers the epic's three open design questions with empirical
evidence and issues a go/no-go verdict.

- **Burn version:** 0.21.0 (`Cargo.toml`), `half` 2.7.1 (transitive).
- **Spike host:** macOS (Darwin 25.5, Apple M-series), Rust nightly 1.94.
- **Method:** a throwaway `examples/fp16_spike.rs` (not merged — see
  [Reproducing](#reproducing)) exercised a one-layer `Autodiff<…>` forward +
  backward on each backend/dtype available on this host. NdArray (CPU) and
  wgpu (Metal) were tested at runtime; cuda was assessed from Burn's source
  because no CUDA host was available.

---

## Verdict: viable on the GPU (cubecl) backends, **but defer adoption**

**f16 and bf16 forward + autodiff are fully supported in Burn 0.21 on the
cubecl GPU backends** (wgpu — verified at runtime; cuda — confirmed by the
shared source path). They are **not** supported on the default `NdArray` CPU
backend (hard compile error).

**Recommendation: do not implement FP16 training yet.** The mechanism works,
but the payoff is nil in Thrust's current regime:

- The default/recommended backend for every current workload is **NdArray
  (CPU)**, which cannot use f16 at all.
- Per [`BURN_BACKENDS.md`](./BURN_BACKENDS.md), at today's net sizes (4-input,
  64-unit MLPs) **CPU already beats every GPU backend 4.4–9.5×** because kernel
  launch + host↔device transfer dominate. FP16 only helps once a workload is
  large enough to be memory- or compute-bound on the GPU — which is also the
  precondition for using a GPU backend *at all*.
- This spike's own timing (below) reproduces the crossover: at a large
  1024×1024 net the GPU beats CPU ~7×, but f16 buys only ~4% over f32 there on
  Apple Metal (no tensor cores). On NVIDIA the f16 win would be larger, but
  Thrust has no workload in that size class today.

**Re-evaluation trigger:** revisit when a large-net / CNN / high-parallelism
workload lands (e.g. image-observation envs, Snake CNN — the same trigger
`BURN_BACKENDS.md` already names for GPU adoption). At that point the opt-in
path described below becomes worthwhile. Child implementation issues have been
filed against the epic and should be sequenced behind that trigger.

---

## Q1 — Does Burn 0.21 support f16/bf16 tensors + autodiff, per backend?

| Backend | f16 forward | f16 autodiff | bf16 | Evidence |
| --- | --- | --- | --- | --- |
| **NdArray** (CPU, default) | ❌ no | ❌ no | ❌ no | **Compile error** (below) |
| **wgpu** (Metal / Vulkan / DX12 / WebGPU) | ✅ yes | ✅ yes | ✅ yes | Compiles **and runs**; loss + grad finite and numerically correct |
| **cuda** (Linux + NVIDIA) | ✅ yes (compiles) | ✅ yes (compiles) | ✅ yes | Same `CubeBackend` / `FloatElement` path as wgpu; runtime **unverified** (no CUDA host) |

### NdArray — not supported (compile error)

`Autodiff<NdArray<f16>>` and `Autodiff<NdArray<bf16>>` do **not compile**:

```
error[E0277]: the trait bound `burn::tensor::f16: FloatNdArrayElement` is not satisfied
   the trait `FloatNdArrayElement` is not implemented for `burn::tensor::f16`
   help: the following other types implement trait `FloatNdArrayElement`:
     impl FloatNdArrayElement for f64 {}
     impl FloatNdArrayElement for f32 {}
```

`burn-ndarray 0.21` implements `FloatNdArrayElement` only for `f32` and `f64`
(`burn-ndarray-0.21.0/src/element.rs:79-80`). The NdArray backend delegates
float math to `ndarray` / `matrixmultiply`, which have no f16 arithmetic path,
so there is nothing to wire up. This is a hard, type-level "no" — not a runtime
panic.

### wgpu — supported and verified at runtime

`Autodiff<Wgpu<f16, i32>>` and `Autodiff<Wgpu<bf16, i32>>` both compile and run.
The spike's one-layer linear grad step produced identical, correct results
across dtypes on Metal:

```
[OK]   NdArray<f32>  (baseline): loss=2.500000  grad_sum=28.000000
[OK]   Wgpu<f32>  (baseline):    loss=2.500000  grad_sum=28.000000
[OK]   Wgpu<f16>:                loss=2.500000  grad_sum=28.000000
[OK]   Wgpu<bf16>:               loss=2.500000  grad_sum=28.000000
```

Forward pass, backward pass, and gradient read-back all work; the f16/bf16
gradients match the f32 reference for this small case.

### cuda — supported at the type level (runtime not tested here)

No CUDA host was available on the spike machine, so cuda was assessed from
Burn's source. The result is a confident "compiles" and a very-likely "runs":

- `Cuda<F, I> = CubeBackend<CudaRuntime, F, I, u8>` and
  `Wgpu<F, I, B> = CubeBackend<WgpuRuntime, F, I, B>` are the **same
  `CubeBackend` generic** over different cubecl runtimes
  (`burn-cuda-0.21.0/src/lib.rs`, `burn-wgpu-0.21.0/src/lib.rs`).
- The float-element bound they require, `FloatElement`, is implemented **once,
  runtime-agnostically**, for `f16`, `bf16`, `flex32`, `f32`, `f64`
  (`burn-cubecl-0.21.0/src/element.rs:66-68`).
- Since `Autodiff<Wgpu<f16>>` compiles and runs, `Autodiff<Cuda<f16>>` uses the
  identical type machinery and the same `burn-autodiff` decorator, so it
  compiles identically. Only NVIDIA-side kernel execution is unverified — and
  CUDA natively supports `__half` + tensor cores, which cubecl targets.

**Note — `flex32`:** cubecl also exposes `flex32`, a relaxed-precision f32
(computes in reduced precision, stores as f32). It is a lower-risk middle option
than true f16 for a first mixed-precision experiment on the cubecl backends.

---

## Q2 — Is loss scaling needed, or does Burn handle it?

**Burn 0.21 ships no automatic mixed-precision (AMP) utility** — there is no
`GradScaler` analogue and no autocast. Any FP16 training path must handle
numerical range manually:

- **f16** has a ~5-bit exponent (min normal ≈ 6.1e-5). Small gradients underflow
  to zero. Real training (many layers, small learning-rate-scaled grads) would
  need **manual loss scaling** — multiply the loss before `.backward()`, unscale
  the grads before the optimizer step, and skip/adjust on overflow. The spike's
  toy loss did not underflow, but that is not evidence a real trainer wouldn't.
- **bf16** has the full 8-bit exponent (f32 dynamic range, reduced mantissa) and
  usually trains **without loss scaling** — at the cost of precision, not range.
  It is the safer dtype where hardware supports it.

So: loss scaling is **required for f16, generally not for bf16**, and Burn
provides neither automatically — it would be part of the implementation work.

---

## Q3 — Expected payoff in the current small-net regime

Low to none today. Wall-clock from the spike (release build, Apple M-series
Metal; 2-layer MLP forward+backward, one leaf per step):

| Workload | Wgpu&lt;f32&gt; | Wgpu&lt;f16&gt; | f16 speedup | NdArray&lt;f32&gt; (CPU) |
| --- | --- | --- | --- | --- |
| batch 256, hidden 256 | 6.76 ms/step | 3.56 ms/step | **1.9×** | — |
| batch 1024, hidden 1024 | 9.28 ms/step | 8.90 ms/step | ~1.04× | 65.4 ms/step |

Reading these:

- f16 gives a real speedup on Metal (1.9× at 256², shrinking to ~4% at 1024²).
  The gain is memory-bandwidth-bound, not tensor-core-bound (Apple GPUs have no
  FP16 tensor cores); on NVIDIA cuda the large-net f16 win would be bigger.
- But **both** GPU dtypes only beat CPU once the net is large. At 1024² the GPU
  is ~7× faster than CPU NdArray — the crossover `BURN_BACKENDS.md` predicts.
  Thrust's actual workloads (4-input, 64-unit) sit far **below** that crossover,
  where CPU wins 4.4–9.5× and f16 is unavailable anyway.

The payoff therefore materializes only when Thrust grows a large-net / CNN /
high-parallelism workload. Until then, FP16 is a mechanism with no target.

---

## Reproducing

The spike used a throwaway `examples/fp16_spike.rs` that is **intentionally not
merged** (per the epic's scope: "spike prototyping scratch, not merged"). To
reproduce, recreate a small example that instantiates
`Autodiff<NdArray<f16>>`, `Autodiff<Wgpu<f16, i32>>`, etc. and run:

```bash
cargo run --example fp16_spike --features training           # NdArray (CPU): f16 fails to compile
cargo run --release --example fp16_spike --features "training,wgpu"   # wgpu/Metal: f16 + bf16 run
cargo run --release --example fp16_spike --features "training,cuda"   # cuda: run on a Linux+NVIDIA host
```

The one-line adoption change, if/when implemented, is at the `type InnerBackend`
alias in each example/trainer (e.g. `NdArray<f32>` → `Wgpu<f16, i32>`), gated
behind an opt-in Cargo feature — no trainer-logic changes are required for the
forward/backward path itself (loss scaling would be additive).

---

## Child issues

Because at least one backend (wgpu) supports f16 autodiff end-to-end, the epic's
candidate implementation phases have been filed as child issues referencing
#267. They are deliberately sequenced behind the re-evaluation trigger above —
implement only once a large-net workload exists to benefit from them.

- **#270** — Opt-in mixed-precision (FP16) training path behind a Cargo feature
  flag (epic phase 2).
- **#272** — Benchmark FP16 vs FP32 training on a large-net workload (epic
  phase 3).
