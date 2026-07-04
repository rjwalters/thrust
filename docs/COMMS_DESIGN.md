# Multi-Agent Communication Channels — Design Note

**Status**: Design proposal (Epic #266). No implementation lands with this note;
the deliverable is this design + a child-issue decomposition.

**Goal**: Give thrust agents a first-class, trainable way to communicate during
a rollout — from a 1-bit broadcast up to a learned message protocol — without
breaking the existing Burn-agnostic multi-agent environment surface.

This note follows the format convention of
[`docs/MULTI_AGENT_DESIGN.md`](./MULTI_AGENT_DESIGN.md).

---

## 1. Current State

Three things exist today that touch "communication," and it is important to keep
them distinct because the epic body conflates them.

### 1.1 Embedded comms in the Bucket Brigade env (the working PoC)

`src/env/games/bucket_brigade/env.rs` already ships a working, trainable comms
channel — but it is baked into one env's action and observation layout, not a
general framework.

- Each agent emits a length-3 action vector `[house_index, mode, signal]`
  (`ACTION_DIMS = 3`; see the module header at lines 12–37 and the
  `agent_action_space` layout `[num_houses, 2, 2]` at line 140).
- The `signal` dim (0 = REST, 1 = WORK) is effectively a **1-bit broadcast
  message** from each agent.
- Received signals from all agents reappear as `round1_signals` in every agent's
  observation vector (obs layout documented at env.rs lines 12–13 and
  366–367; `round1_signals` contributes `num_agents` dims at line 163).

This proves the **action-space-extension model** end-to-end: a message is just
extra action dims on the sender and extra observation dims on the receiver, and
the existing multi-discrete policy trains it with no framework changes. The cost
is that it requires env-specific action/obs layout surgery and does not
generalize to other environments or richer message types.

### 1.2 `messages.rs` — training infrastructure, not agent-to-agent comms

`src/multi_agent/messages.rs` defines `Experience`, `PolicyUpdate`,
`TrainingStats`, and `ControlMessage`. These are **simulator-thread →
learner-thread** carriers (experience out, policy weights back). They are plain
`Vec<f32>` / `Vec<i64>` host payloads by design so producer and consumer can pick
different Burn backends (module header, lines 1–16). There is **no
`AgentMessage` type** and nothing here addresses agent-to-agent communication
during a rollout.

### 1.3 `joint.rs` — the "Slepian-Wolf adapter" is a training coupling

`src/multi_agent/joint.rs` implements the cross-agent representational
redundancy penalty
`L_red = λ * Σ_{i<j} || corr(Z_i, Z_j) ||_F² / d²` (module header, lines 14–18),
surfaced as the shared scalar `JointStats::aux_loss` (line 526–529). This is a
**loss-function coupling** applied at train time on a shared minibatch — it is
not a runtime message-passing API. It is relevant to this epic only as the
**hook where a comms-regularization term would attach** (see Phase 2).

### 1.4 The trait gap

`src/multi_agent/environment.rs` defines the shipped, Burn-agnostic
`MultiAgentEnvironment` trait (lines 39–78):

```rust
pub trait MultiAgentEnvironment: Environment {
    fn num_agents(&self) -> usize;
    fn get_agent_observation(&self, agent_id: usize) -> Vec<f32>;
    fn agent_action_space(&self, agent_id: usize) -> Vec<usize>;
    fn step_multi(&mut self, actions: &[Vec<i64>]) -> MultiAgentResult;
    fn active_agents(&self) -> Vec<bool>;
}
```

`MultiAgentResult` (lines 84–105) carries per-agent `observations`, `rewards`,
`terminated`, `truncated`, and a **shared** `info: HashMap<String, String>`.
There is **no first-class message slot**. Today, agent-to-agent comms must
either be embedded in the observation/action space (as BB does) or smuggled
through `info` (stringly-typed, not trainable).

### 1.5 `crossbeam-channel`

`Cargo.toml` line 99 declares `crossbeam-channel = { version = "0.5", optional
= true }`, enabled only by the `training` feature (line 144). It is the
cross-thread primitive for the learner/simulator infrastructure. It is **not**
needed for synchronous in-step comms (agents share the env's mutable state
within a single `step_multi` call); it becomes relevant only for async,
deferred-delivery comms (Phase 3+).

---

## 2. Recommended Comms Model

Adopt the **action-space-extension model as the default**, formalized behind a
**non-breaking supertrait** so that the general case (variable vocab, typed
messages, partial delivery) has a home without disturbing existing implementors.

Three layered pieces, phased:

1. **Phase 1 — Action-space extension for discrete fixed-vocab comms.**
   Messages are an additional group of action dims on the sender; received
   messages are additional observation dims on the receiver. This is exactly the
   proven BB pattern, lifted into a reusable helper + a small reference env.
2. **Phase 2 — `CommunicatingEnvironment` supertrait + trainer hook.** A thin
   supertrait with defaulted methods makes the message channel introspectable
   (sizes, routing) without changing `MultiAgentEnvironment`, and exposes an
   optional comms-regularization term through the existing `joint.rs` aux-loss
   slot.
3. **Phase 3 — Differentiable (learned) comms.** Gumbel-softmax relaxation over
   message logits so gradients flow sender → receiver. Deferred: it requires
   continuous/relaxed policy output support that thrust does not yet have.

### Why this model

- **Proven**: BB already trains a 1-bit channel this way end-to-end.
- **Framework-free for the common case**: the existing `MultiDiscreteMlpBurnPolicy`
  already emits multi-discrete actions and consumes flat observation vectors, so
  fixed-vocab comms needs no new policy or trainer code.
- **Gym/Gymnasium-compatible**: no new trait methods are required for the simple
  case; a message is indistinguishable from any other action/obs dim.
- **Non-breaking**: the supertrait carries default impls, so snake, matching
  pennies, bucket brigade, and the joint env keep compiling untouched.

### Why NOT a mandatory new channel in `MultiAgentResult`

Adding a required `messages` field to `MultiAgentResult` or a required method to
`MultiAgentEnvironment` is a **breaking change** for every current implementor
and forces a message concept onto envs that do not communicate. The supertrait
keeps comms opt-in.

---

## 3. API Sketch

All syntax below is illustrative (design-stage; not compiled).

### 3.1 `AgentMessage` (addition to `src/multi_agent/messages.rs`)

A typed, backend-agnostic message carrier that mirrors the existing
`Vec<f32>`/`Vec<i64>` host-payload discipline of the file:

```rust
/// A message emitted by one agent for delivery to others within a rollout.
///
/// Host-side payload only (no tensor types), consistent with the rest of
/// `messages.rs`. Discrete tokens live in `tokens`; an optional continuous
/// payload supports the differentiable path (Phase 3).
#[derive(Debug, Clone, Default)]
pub struct AgentMessage {
    /// Sending agent.
    pub sender: AgentId,
    /// Discrete message tokens (multi-discrete, one entry per message dim).
    /// Cardinalities are published by `CommunicatingEnvironment::message_vocab`.
    pub tokens: Vec<i64>,
    /// Optional continuous payload (relaxed/learned comms, Phase 3).
    /// Empty for fixed-vocab discrete comms.
    pub embedding: Vec<f32>,
}

/// Delivery routing for a single step.
#[derive(Debug, Clone)]
pub enum Delivery {
    /// Delivered to every other agent (the BB `signal` model).
    Broadcast,
    /// Delivered only to the listed recipients.
    Targeted(Vec<AgentId>),
}
```

### 3.2 `CommunicatingEnvironment` supertrait (`src/multi_agent/environment.rs`)

Non-breaking: every method has a default that reports "no comms," so existing
`MultiAgentEnvironment` implementors satisfy it for free.

```rust
/// Opt-in extension of `MultiAgentEnvironment` for environments that expose a
/// first-class agent-to-agent message channel.
///
/// Default impls describe a non-communicating environment, so this is a
/// non-breaking supertrait: existing implementors get a zero-width channel.
pub trait CommunicatingEnvironment: MultiAgentEnvironment {
    /// Per-agent message action-space layout (one bin count per message dim),
    /// analogous to `agent_action_space`. Empty vec = agent sends nothing.
    fn message_vocab(&self, _agent_id: usize) -> Vec<usize> {
        Vec::new()
    }

    /// Number of observation dims that carry *received* messages for this
    /// agent. Zero = agent receives nothing. The env is responsible for
    /// placing received messages into the observation vector returned by
    /// `get_agent_observation` / `step_multi`.
    fn message_obs_size(&self, _agent_id: usize) -> usize {
        0
    }

    /// Delivery policy for messages emitted this step. Defaults to broadcast,
    /// matching the Bucket Brigade `signal` semantics.
    fn delivery(&self) -> Delivery {
        Delivery::Broadcast
    }
}
```

### 3.3 Where messages ride in `step_multi`

Phase 1/2 deliberately keep the wire format flat: message tokens are the trailing
dims of each agent's action vector, and received messages are trailing dims of
each agent's observation vector. Concretely, an agent's action vector becomes

```
[ ...task action dims (agent_action_space) ..., ...message dims (message_vocab) ... ]
```

The env slices off the message dims inside `step_multi`, applies `delivery()`,
and writes received tokens into the observation layout it already controls. No
change to the `MultiAgentResult` struct is required for Phase 1/2. A reusable
helper (e.g. `comms::split_action(action, task_len)` → `(task, message)`)
factors the BB-style slicing so reference and downstream envs share one
implementation.

A first-class `messages: Vec<AgentMessage>` field on `MultiAgentResult` is
**explicitly deferred to Phase 3**, when async/targeted delivery and
introspection (logging what was said) justify the added surface.

---

## 4. Design Questions Answered

The epic poses four questions. Each is answered with rationale grounded in the
code above.

### Q1 — Explicit message actions appended to the action space, vs. a separate comms channel in the env API?

**Recommendation: action-space extension as the default; a separate channel only
for the general case (Phase 3).**

The action-space model is already proven by the BB `signal` dim
(`env.rs:33–37`) and trains with the existing multi-discrete policy with zero
framework changes. It is Gym-compatible and requires no new trait method for
simple cases. A distinct comms channel (a `messages` field / dedicated method)
only earns its keep when messages become typed, targeted, or deferred — that is
Phase 3, and the `AgentMessage`/`Delivery` types above reserve the design space
for it.

### Q2 — Extend `MultiAgentEnvironment`, or a new adapter?

**Recommendation: a new supertrait `CommunicatingEnvironment: MultiAgentEnvironment`
with defaulted methods.**

Extending the base trait breaks every current implementor (snake, matching
pennies, bucket brigade, joint env) and forces a comms concept onto envs that do
not need it. A supertrait with default impls
(`message_obs_size(..) -> 0`, `message_vocab(..) -> vec![]`) is non-breaking and
mirrors the codebase's existing adapter layering. Envs opt in by implementing the
supertrait; the trainer probes for it.

### Q3 — Differentiable (learned) comms vs. fixed protocol?

**Recommendation: phased — fixed discrete vocab first, differentiable later.**

- **Phase 1**: fixed discrete vocab via extra action dims. Immediately
  achievable with `MultiDiscreteMlpBurnPolicy`; this is what BB already does.
- **Phase 3**: differentiable relaxation via **Gumbel-softmax** over message
  logits, so sender gradients flow into the receiver's loss. This requires
  continuous/relaxed policy output support that thrust does **not** have yet, so
  it is deferred. The `AgentMessage::embedding` field and this section are the
  hooks left for it.

Rationale: fixed-vocab comms delivers a usable, trainable channel now; learned
comms is a research feature gated on a prerequisite (continuous policy support)
that should be tackled on its own.

### Q4 — `crossbeam-channel` for the runtime plumbing?

**Recommendation: not for Phase 1/2; yes for Phase 3.**

For synchronous, step-level comms the env's own mutable state is the message bus:
within a single `step_multi` call the env reads every agent's emitted message and
writes each receiver's observation before returning. **No channel is needed**
(and none is added). `crossbeam-channel` (`Cargo.toml:99`, `training` feature)
becomes the right primitive only for **async deferred delivery / partial
observability** in Phase 3, where a message sent at step *t* is delivered at step
*t+k* across the simulator/learner thread boundary. The design note draws this
boundary explicitly so Phase 1/2 do not prematurely pull in channel machinery.

---

## 5. Phase Breakdown

| Phase | Issue | Scope | Est. | Prereq |
|-------|-------|-------|------|--------|
| **1** | #274 | `CommunicatingEnvironment` supertrait + `AgentMessage`/`Delivery` types + action/obs slicing helper + a minimal reference comms env + tests | 2–3 days | none |
| **2** | #275 | Joint-trainer integration: probe `CommunicatingEnvironment`, route messages through the rollout, and expose an optional comms-regularization term via the existing `JointStats::aux_loss` hook | 1–2 days | Phase 1 |
| **3** | #276 | Differentiable comms: Gumbel-softmax message head + gradient flow sender→receiver; optional async delivery over `crossbeam-channel` | deferred | continuous/relaxed policy support (not yet in thrust) |

### Phase 1 — Trait + reference env

- Add `AgentMessage` and `Delivery` to `src/multi_agent/messages.rs`.
- Add `CommunicatingEnvironment` supertrait (defaulted methods) to
  `src/multi_agent/environment.rs`.
- Add a `comms` helper for splitting task vs. message action dims and for laying
  received messages into observation vectors (factor out the BB pattern).
- Ship one small reference env implementing `CommunicatingEnvironment` (e.g. a
  2-agent "referential signaling game": one agent observes a hidden token and
  must transmit it so the other can act on it).
- Unit tests: default impls report a zero-width channel; the reference env
  round-trips a message; slicing helper is correct.

### Phase 2 — Joint trainer integration + comms loss hook

- In `joint.rs`, detect `CommunicatingEnvironment` and route emitted messages to
  receivers across the rollout collection loop.
- Add an **optional** comms-regularization term (e.g. a message-entropy or
  bandwidth penalty) computed on the shared minibatch and folded into
  `JointStats::aux_loss` — reusing the existing aux-loss plumbing rather than
  adding a new one.
- Tests: end-to-end train step on the reference env with comms enabled produces
  finite loss and a populated `aux_loss` when the comms term is on.

### Phase 3 — Differentiable comms (deferred / stretch)

- Gumbel-softmax message head so message logits are differentiable; gradients
  flow from receiver loss back into the sender.
- Optional async/targeted delivery using `crossbeam-channel`.
- **Explicitly gated** on continuous/relaxed policy output support landing first;
  file as deferred.

---

## 6. Out of Scope

- **`crossbeam-channel` for in-episode comms** (Phase 1/2 use the env's mutable
  state as the message bus).
- **Async / partial-observability deferred delivery** (Phase 3+).
- **Continuous-action / relaxed policy support** — a prerequisite for Phase 3
  differentiable comms, tracked separately.
- **A mandatory `messages` field on `MultiAgentResult`** — deferred to Phase 3;
  Phase 1/2 keep messages inline in the action/observation vectors.
- **Rewriting the Bucket Brigade env** onto the new supertrait — BB's embedded
  `signal` channel keeps working as-is; migrating it is optional follow-up, not
  part of this epic.
