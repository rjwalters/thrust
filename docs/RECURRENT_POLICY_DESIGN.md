# LSTM / recurrent policy support — design note

Design note for [#262](https://github.com/rjwalters/thrust/issues/262)
("LSTM / recurrent policy support"). This is the **design/ADR deliverable**, not
an implementation: it answers the two load-bearing design questions the curator
flagged (buffer strategy, minibatch strategy), decides the surrounding trainer
and environment questions, grounds every claim in Thrust's own source and the
Burn 0.21 API, and decomposes the work into scoped child issues.

- **Burn version:** 0.21.0 (`Cargo.toml:63`).
- **Method:** answers are grounded in (a) Thrust's policy / buffer / trainer code
  on `feature/issue-262` (branched from `origin/main`), and (b) the Burn 0.21
  `burn-nn` crate as cached in the local cargo registry
  (`burn-nn-0.21.0/src/modules/rnn/`).
- **Template:** mirrors [`DISTRIBUTED_TRAINING_DESIGN.md`](./DISTRIBUTED_TRAINING_DESIGN.md)
  (epic #265, PR #282) and [`COMMS_DESIGN.md`](./COMMS_DESIGN.md) (epic #266,
  PR #277) — verdict first, one section per open question with source citations,
  then a child-issue decomposition.

---

## Verdict: new `RecurrentRolloutBuffer` + full-sequence minibatches, behind a separate `RecurrentPPOTrainer`

Two load-bearing decisions, plus the two subordinate ones the curator raised:

1. **Buffer strategy → new `RecurrentRolloutBuffer`, not extended `RolloutBuffer`.**
   Keep the recurrence-specific storage (per-step `(h_t, c_t)`, sequence layout,
   episode boundaries) in an isolated type so the shipped feedforward
   `RolloutBuffer` and its ~dozen consumers stay byte-for-byte untouched.
2. **Minibatch strategy → full-sequence (Strategy A), one env-trajectory = one
   sequence.** Minibatch over *environments*, keep each env's `T`-step trajectory
   temporally intact as a `[N_env, T, obs_dim]` rank-3 tensor, and reset the LSTM
   state at **episode boundaries** (`terminated || truncated`) *inside* the
   forward pass. Note the asymmetry with GAE: value bootstrapping resets on
   `terminated` alone (truncation keeps bootstrapping), but hidden-state reset
   follows episode *starts* — `terminated || truncated` — because a truncated
   step is still the end of an episode in the buffer (see the GAE-vs-state-reset
   note under Q2). No hidden states need to be stored for the training pass; only
   the rollout-time states are recorded (for warm-starting and debugging).
   Fixed-length subsequences (SB3 `RecurrentPPO` Strategy B) are a deferred
   optimization, not a v1 requirement.
3. **Trainer API → new `RecurrentPPOTrainer<B, P, O>`, not a parameterized
   existing trainer.** The feedforward trainer's evaluate closure is hard-typed
   to rank-2 observations (`Tensor<B, 2>`, `trainer.rs:168`); recurrence needs
   rank-3. Adding a parallel trainer avoids a contract-breaking change to the
   existing `train_step` signature and its callers.
4. **POMDP env → MaskedCartPole first.** Smallest possible env change that
   produces a learning signal a feedforward policy provably cannot match; T-Maze
   is a stretch follow-up.

Rationale for each below.

---

## Q1 — Buffer strategy: new `RecurrentRolloutBuffer` vs. extend `RolloutBuffer`?

**Recommendation: a new `RecurrentRolloutBuffer` type; leave `RolloutBuffer`
untouched.**

### What the existing buffer is, and why extending it is costly

`RolloutBuffer` (`src/buffer/rollout/storage.rs:26-63`) stores nine parallel
`[num_steps][num_envs]` host arrays and nothing else — there is no hidden-state
slot. Its whole contract is *flatten time and env together*:

- `RolloutBatch::from_buffer_partial` (`storage.rs:303-334`) collapses
  `[step][env]` into one flat `[batch_size]` row list in a double loop
  (`for step … for env …`, `storage.rs:322-331`), and `to_burn_tensors`
  materializes observations as a **rank-2** `[batch, obs_dim]` tensor
  (`storage.rs:379-382`, `RolloutBurnTensors.observations: BurnTensor<B, 2>` at
  `storage.rs:419`).
- `sample_minibatch` (`src/buffer/rollout/sampling.rs:37-84`) further flattens
  every field with `.iter().flatten()` (`sampling.rs:48-58`) and indexes samples
  by a single global integer, discarding which `(step, env)` each row came from.

Temporal ordering is *structurally erased* by the time data leaves this type.
Recurrence needs the exact opposite: `(step, env)` identity preserved, plus two
new per-step tensors (`hidden`, `cell`) that no existing field carries. Bolting
optional `Option<Vec<Vec<Vec<f32>>>>` hidden/cell fields onto `RolloutBuffer`
would:

- Force every `RolloutBatch` / `Minibatch` consumer to reason about a
  sometimes-present rank-3 layout (the batch types are `pub` and materialize
  rank-2 tensors by construction — `storage.rs:263-281`, `417-438`).
- Leave the flatten-based `from_buffer_partial` / `sample_minibatch` paths as
  dead weight for the recurrent case (they cannot preserve sequences), so the
  recurrent code would need a parallel path *anyway*.

A separate type pays that cost once, in isolation, and the shipped feedforward
PPO/A2C tests (`src/buffer/rollout/tests.rs`, `src/train/ppo/`,
`src/train/a2c/`) keep compiling and passing with zero edits — satisfying the
"no regressions on feedforward consumers" acceptance criterion by construction.

### Sketch (design-stage; not compiled)

```rust
/// Recurrent rollout storage. Preserves per-env temporal order and records
/// the policy's recurrent state at each step. Mirrors `RolloutBuffer`'s
/// backend-agnostic `Vec<f32>` host layout (storage.rs:7-9).
pub struct RecurrentRolloutBuffer {
    num_steps: usize,
    num_envs: usize,
    obs_dim: usize,
    hidden_dim: usize,

    // Same nine fields as RolloutBuffer, [num_steps][num_envs] ...
    observations: Vec<Vec<Vec<f32>>>,
    // ... actions, rewards, values, log_probs, terminated, truncated,
    //     advantages, returns ...

    /// LSTM hidden state entering each step, [num_steps][num_envs][hidden_dim].
    hidden: Vec<Vec<Vec<f32>>>,
    /// LSTM cell state entering each step, [num_steps][num_envs][hidden_dim].
    cell: Vec<Vec<Vec<f32>>>,
}

impl RecurrentRolloutBuffer {
    /// Record the recurrent state that produced this step's action.
    pub fn add_recurrent_state(&mut self, step: usize, env_id: usize,
                               h: &[f32], c: &[f32]) { /* ... */ }

    /// Materialize a full-sequence batch: obs as [N_env, T, obs_dim], plus
    /// per-(step,env) episode-start flags (terminated || truncated) so the
    /// trainer can reset state mid-forward at episode boundaries.
    pub fn to_sequence_batch<B: Backend>(&self, device: &B::Device)
        -> RecurrentRolloutBatch<B> { /* ... */ }
}
```

`add`, GAE (`src/buffer/rollout/gae.rs`), and the getter surface can be reused
essentially verbatim — GAE already operates on the `[step][env]` advantage/return
grids (`storage.rs:253-255`), and that layout is unchanged.

**Why store `(h_t, c_t)` at all if full-sequence training re-derives them?**
Two reasons: (a) warm-starting — the recurrent state carried across rollout
*iterations* (the state at the last step of iteration `k` seeds step 0 of
iteration `k+1` for envs that did not reset — i.e. envs that neither
`terminated` **nor** `truncated` on that last step; a time-limit truncation
ends the episode and must seed step 0 with a zeroed state, not the carried
one), and (b) it is the storage hook
Strategy B needs if it is ever adopted. v1 uses the stored states only for
warm-start; the training forward recomputes states from the initial state.

---

## Q2 — Minibatch strategy: full-sequence (A) vs. fixed-length subsequences (B)?

**Recommendation: full-sequence (Strategy A) for v1. Defer Strategy B.**

### Why the current sampler is disqualifying, and what A changes

Both minibatch samplers in the tree shuffle the *global* timestep index and
chunk it:

- `generate_minibatch_indices` (`sampling.rs:20-27`) does
  `indices.shuffle(&mut rand::rng()); indices.chunks(batch_size)` over
  `0..buffer_size`.
- The PPO trainer's seedable variant `generate_minibatch_indices_with_rng`
  (`src/train/ppo/loss.rs:214-222`) does the same shuffle, and the trainer then
  gathers those scattered rows with `select_rows_2d` / `select_rows_int`
  (`trainer.rs:192-193`).

Shuffling `num_steps * num_envs` indices and chunking **splits episodes across
minibatches and reorders timesteps** — fatal for an LSTM, whose forward pass is
defined only over an ordered sequence.

**Strategy A** keeps each environment's rollout as one ordered sequence of length
`T = num_steps`. A minibatch is a subset of the `N_env` environments; within a
minibatch the tensor is `[n_env_in_batch, T, obs_dim]` — exactly the rank Burn's
LSTM forward consumes (see Q4). Episode boundaries *inside* a sequence are
handled by resetting `(h, c) → 0` at steps where the episode ends —
`terminated[step][env] || truncated[step][env]` — during the forward pass, not
by cutting the sequence. This is the standard "reset state at episode starts"
recurrent-PPO forward.

Concretely the recurrent trainer replaces the shuffle-and-gather path with an
env-major shuffle:

```rust
// Instead of shuffling num_steps*num_envs timesteps, shuffle env indices only.
let mut env_ids: Vec<usize> = (0..num_envs).collect();
env_ids.shuffle(rng);
for env_chunk in env_ids.chunks(envs_per_minibatch) {
    // obs_seq: [chunk_len, T, obs_dim]; episode_end_seq: [chunk_len, T]
    //   where episode_end = terminated || truncated
    // evaluate_sequences runs the LSTM over T, resetting (h,c) at episode ends
}
```

`envs_per_minibatch` is the recurrent analogue of `batch_size` (a count of whole
trajectories, not loose timesteps).

### GAE bootstrapping vs. hidden-state reset — a deliberate asymmetry

The reset condition for the LSTM state is **not** the same as the reset
condition GAE uses, and implementers must not "fix" one to match the other:

- **GAE / value bootstrapping** resets on `terminated` **only**
  (`src/buffer/rollout/gae.rs`). A time-limit `truncated` step is *not* a true
  terminal — the value target should continue to bootstrap from the next state's
  value, so truncation must **not** zero the return. This is correct and stays
  as-is.
- **Hidden-state reset** follows **episode boundaries**, i.e.
  `terminated || truncated`. Thrust's rollout loop resets the environment on
  either flag (`examples/games/cartpole/train_cartpole_modern.rs:204` computes
  `done = terminated || truncated`; line 214 resets on it), and CartPole /
  MaskedCartPole truncate at `max_steps = 500` (`src/env/games/cartpole.rs:105`).
  After a truncation the next observation in the buffer belongs to a **fresh
  episode**, so carrying the LSTM state across that boundary leaks stale memory
  into a new episode — precisely the corruption this design exists to prevent.
  This bites hardest late in training, when a good policy hits the 500-step
  truncation routinely.

This is exactly the distinction SB3's `RecurrentPPO` encodes with its
`episode_starts` array (episode-boundary semantics) kept separate from the
value-bootstrap terminal flag. The buffer therefore exposes an
episode-start/done flag (`terminated || truncated`) for state reset while GAE
keeps consuming `terminated` alone.

### Why not Strategy B (SB3 `RecurrentPPO` fixed-length subsequences) in v1

Strategy B chops each trajectory into fixed `seq_len` chunks and stores the LSTM
state at each chunk boundary to re-initialize from it during training. It is more
flexible (decouples truncation-window from `num_steps`, gives more minibatches
for long rollouts) but strictly more machinery:

- It **requires** boundary-state storage as a first-class training input (not
  just warm-start), so the buffer, sampler, and trainer all grow a state-plumbing
  path v1 does not otherwise need.
- Its win only materializes when `T` is large enough that one-env-per-sequence
  gives too few / too large minibatches. Thrust's PPO rollouts are modest
  (`num_steps` typically 128–2048 per the `storage.rs:22` doc-comment) and
  MaskedCartPole episodes are short, so A's granularity is adequate.

A's `(h_t, c_t)` storage (Q1) is deliberately the same hook B would reuse, so
adopting B later is additive: a new sampler + a trainer flag, no buffer redesign.
This is the same "reserve the design space, ship the simpler thing" move
`COMMS_DESIGN.md` makes for Gumbel-softmax comms (Phase 3 there).

---

## Q3 — Trainer API: new `RecurrentPPOTrainer` vs. parameterize the existing trainer?

**Recommendation: a new `RecurrentPPOTrainer<B, P, O>`; do not touch the existing
`PPOTrainerBurn::train_step` signature.**

The feedforward trainer is closure-dispatched, and the closure's observation type
is rank-2 and load-bearing:

```rust
// src/train/ppo/trainer.rs:157-168
pub fn train_step<F>(
    &mut self,
    observations: Tensor<B, 2>,          // <-- rank-2, [batch, obs_dim]
    // ...
    mut evaluate_fn: F,
) where
    F: FnMut(&P, Tensor<B, 2>, Tensor<B, 1, Int>)
        -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>),
```

Recurrent evaluation needs rank-3 observations (`[n_env, T, obs_dim]`) plus an
initial-state argument and per-step episode-boundary flags
(`terminated || truncated`, for state reset) — an incompatible signature. Changing `train_step` in place is a contract break for every caller
(e.g. the doctest-style call `|p, o, a| p.evaluate_actions(o, a)` at
`trainer.rs:425`). A sibling trainer keeps the blast radius at zero, exactly as
`DISTRIBUTED_TRAINING_DESIGN.md` chose an actor-learner *around* the existing
trainer rather than restructuring `train_step`.

The recurrent policy exposes two entry points (Phase 2 below): `forward_step`
for single-step rollout collection (threads `(h, c)` in and out), and
`evaluate_sequences` for the rank-3 training forward. `RecurrentPPOTrainer`
mirrors `PPOTrainerBurn`'s loss/optimizer core (`GradientsParams::from_grads`
→ `optimizer.step`, `trainer.rs:240-245`) and reuses the loss functions in
`src/train/ppo/loss.rs` unchanged — only the batch assembly (rank-3, sequence
sampler) differs.

---

## Q4 — Burn 0.21 LSTM API realities

Confirmed against `burn-nn-0.21.0/src/modules/rnn/` in the local registry:

- **`Lstm<B>` forward is rank-3, stateful, and returns the full output
  sequence + final state** (`lstm.rs:172-176`):

  ```rust
  pub fn forward(
      &self,
      batched_input: Tensor<B, 3>,     // [batch, seq_len, input_size], batch_first=true
      state: Option<LstmState<B, 2>>,
  ) -> (Tensor<B, 3>, LstmState<B, 2>) // [batch, seq_len, hidden], final state
  ```

  This maps *directly* onto Strategy A: feed `[n_env, T, obs_dim]`, take the
  `[n_env, T, hidden]` output into the policy/value heads, and reset state at
  episode boundaries (`terminated || truncated`) between sub-segments.

- **`LstmState<B, 2>` is `{ cell, hidden }`, each `[batch, hidden_size]`**
  (`lstm.rs:11-15`) — the two tensors `RecurrentRolloutBuffer` records per step.
- **`LstmConfig` supports `batch_first` (default true, `lstm.rs:40`) and an
  `initializer` (default `XavierNormal{gain:1.0}`, `lstm.rs:35-36`).** The
  default init is **unseeded**, which conflicts with the issue's reproducibility
  criterion.
- **GRU** (`gru.rs`) is the same shape with a single state tensor; it is a drop-in
  variant and can be an optional config flag, not a v1 requirement.

### Seeded init for the LSTM gates

The reproducibility criterion ("two constructions with the same seed produce
bit-identical weights") is met the same way `MlpBurnConfig::seed` already is.
`MlpBurnPolicy` bypasses Burn's unseedable `Initializer` by building each
`Linear` from a pre-computed, seeded weight buffer via `seeded_layer_weights`
(`src/policy/mlp.rs:100-115`) with a distinct per-layer seed from
`derive_layer_seed(base_seed, layer_index)` (`mlp.rs:124-125`), drawing from
`seeded_orthogonal` / `seeded_kaiming_uniform` (`src/policy/seeded_init.rs:82,164`).

The LSTM's weights live in four `GateController`s (input/forget/cell/output),
each holding two `Linear` layers — `input_transform` `[input_size, hidden_size]`
and `hidden_transform` `[hidden_size, hidden_size]`
(`gate_controller.rs:16-20`). So the seeded path overrides **eight** `Linear`
layers post-`init` using eight derived seeds, reusing `seeded_layer_weights`
verbatim — `GateController`'s fields are `pub`, so an `LstmBurnConfig { seed }`
can swap each `Linear.weight`/`bias` `Param` exactly as `mlp.rs` does. No new
seeded-init primitive is required; `seeded_init.rs` extends naturally.

---

## Reproducing the source claims

All checkable from the repo root + local cargo registry, no GPU needed:

```bash
# 1. Feedforward buffer erases temporal order (flatten in from_buffer_partial):
sed -n '303,334p' src/buffer/rollout/storage.rs

# 2. Both minibatch samplers shuffle global timesteps (incompatible with RNNs):
sed -n '20,27p'   src/buffer/rollout/sampling.rs
sed -n '214,222p' src/train/ppo/loss.rs

# 3. PPO evaluate closure is rank-2 (the contract that would break):
sed -n '157,168p' src/train/ppo/trainer.rs

# 4. Burn 0.21 LSTM: rank-3 forward, Option<LstmState>, unseeded default init:
REG=~/.cargo/registry/src/index.crates.io-*/burn-nn-0.21.0/src/modules/rnn
sed -n '11,15p;35,40p;172,176p' $REG/lstm.rs
sed -n '16,20p'                  $REG/gate_controller.rs

# 5. Seeded-init pattern the LSTM path reuses:
sed -n '100,125p' src/policy/mlp.rs
grep -n 'pub fn seeded_' src/policy/seeded_init.rs
```

---

## Child issues

Following the epic phasing the curator proposed, refined against the decisions
above. Each is filed with `loom:triage` and references #262.

| Phase | Scope | Key deliverables | Depends on |
| --- | --- | --- | --- |
| 1 | `LstmBurnPolicy<B>` + seeded LSTM init | `src/policy/lstm.rs`: LSTM trunk + policy/value heads; `forward_step(obs, state)`; `evaluate_sequences(obs_seq, actions, init_state, episode_starts)` (episode_starts = per-step `terminated \|\| truncated`, for state reset); `LstmBurnConfig { seed }` overriding all 8 gate `Linear`s via `seeded_layer_weights`; bit-identical-seed test | none |
| 2 | `RecurrentRolloutBuffer` + full-sequence sampler | New buffer type with per-step `(h, c)` storage + `add_recurrent_state`; `to_sequence_batch` → `[N_env, T, obs_dim]`; env-major sequence sampler with episode-boundary (`terminated \|\| truncated`) state reset; GAE reused (bootstraps on `terminated` only — see the GAE-vs-state-reset note under Q2); existing `RolloutBuffer` untouched | Phase 1 (for state shapes) |
| 3 | `RecurrentPPOTrainer` + MaskedCartPole + end-to-end example | New `src/train/ppo/recurrent_trainer.rs` (rank-3 forward, reuses `loss.rs`); `src/env/games/masked_cartpole.rs` (drop velocity dims); `examples/recurrent_ppo_masked_cartpole.rs` showing the recurrent policy learns where a feedforward baseline cannot | Phases 1 + 2 |

Deferred (not filed, reserved by the `(h, c)` storage hook): Strategy B
fixed-length subsequences; GRU variant; T-Maze memory env; V-trace + recurrent
integration (tracked separately under the distributed-training epic).
