# Tutorial 6 — Writing your own environment

> **You will:** implement Thrust's `Environment` trait from scratch — a tiny
> 3×3 gridworld — and learn the seeding/determinism contract that makes
> snapshot-and-restore (`clone_state` / `restore_state`) safe for MCTS-style
> lookahead.
> **Prerequisites:** [Tutorial 1](01-your-first-agent.md) — you only need to
> recognize the `Environment` trait and the rollout loop. Nothing from the PPO,
> DQN, or SAC tutorials is required.
> **Time:** ~15 minutes.

Every tutorial so far *used* a built-in environment — `SimpleBandit`,
`CartPole`, `GridWorld`. Those are just structs that implement one trait,
`Environment`, and once you implement it for a struct of your own, every trainer
and tool in Thrust (the env pool, the PPO/DQN loops, the snapshot machinery)
works with your environment for free. There is no registration step and no base
class: **implementing the trait is the whole contract.**

This tutorial builds the smallest environment that is still interesting: a 3×3
grid where an agent walks from the top-left corner to the bottom-right goal. It
is deliberately tiny so the prose can focus on *what each trait method is for*
and *why the determinism contract matters*, rather than on env design.

## The trait, in full

`Environment` has two associated types and ten methods. There are no default
implementations — you provide all ten — but several are one-liners.

```text
trait Environment {
    type Action;   // what step() accepts: i64 for discrete, Vec<f32> for continuous
    type State;    // an opaque snapshot for clone_state / restore_state

    fn reset(&mut self);                         // start a new episode
    fn get_observation(&self) -> Vec<f32>;       // the agent's view of "now"
    fn step(&mut self, a: Self::Action) -> StepResult;  // advance one tick
    fn observation_space(&self) -> SpaceInfo;    // shape + kind of observations
    fn action_space(&self) -> SpaceInfo;         // shape + kind of actions
    fn render(&self) -> Vec<u8>;                 // optional visualization bytes
    fn close(&mut self);                         // optional teardown
    fn clone_state(&self) -> Self::State;        // snapshot for lookahead
    fn restore_state(&mut self, s: &Self::State);// rewind to a snapshot
}
```

That is the entire surface. The rest of this section is *why* each method exists
— because the point of a trait is the promises it makes to callers, and you
cannot implement it well without knowing what those callers expect.

### `Action` and `State`: the two type knobs

- **`type Action`** is what `step` accepts. Discrete environments — anything
  where the agent picks one of *n* buttons — set `type Action = i64;` and treat
  the integer as an index. Continuous-control environments set `type Action =
  Vec<f32>;`. Our gridworld has four moves, so it is discrete: `i64`.
- **`type State`** is an opaque snapshot type *you* define. It holds exactly the
  fields needed to rewind the simulation — no more. Keeping it separate from the
  environment struct is what lets `clone_state` hand out a cheap, `Clone`-able
  value without copying things like the RNG or scratch buffers.

### Why `reset` is separate from construction

`new()` builds the struct; `reset()` starts a fresh episode. They are separate
on purpose. A trainer holds **one** environment and plays **thousands** of
episodes through it — every time an episode ends (the pole falls, the agent
reaches the goal), the trainer calls `reset()` to begin the next one *without*
reallocating. Construction is where you set up fixed, episode-independent things
(the seed, the grid layout, the step cap); `reset` is where you return the
per-episode state (agent position, step counter) to its starting values. If you
folded reset into the constructor, the pool would have to drop and rebuild an
env on every episode boundary.

### `get_observation`: the agent's view

`get_observation` returns what the policy network actually sees — always a
`Vec<f32>`, because networks eat float vectors. It is deliberately a *view*, not
the full internal state: the observation can hide information (that is exactly
what makes a task a POMDP). Our grid returns a **one-hot** vector of length 9 —
all zeros except a single `1.0` at the agent's cell. One-hot keeps the network
input fixed-width and makes "which cell am I in" trivially linearly separable.

### `step`: the one method that moves time

`step` is the heart of the environment. It takes an action, advances the
simulation one tick, and returns a `StepResult`:

```text
StepResult {
    observation: Vec<f32>,  // the NEW observation, after the action
    reward: f32,            // the scalar the agent is trying to maximize
    terminated: bool,       // episode ended for a "real" reason (goal/failure)
    truncated: bool,        // episode cut off by a time/step limit
    info: StepInfo,         // extra diagnostics; StepInfo::default() is fine
}
```

The `terminated` / `truncated` split matters and is easy to get wrong.
**Termination** means the episode reached a genuine end state — the agent hit the
goal or died — and the value of that state is fully captured by its reward.
**Truncation** means you cut the episode off artificially, usually at a step cap,
while the task was still "in progress." Bootstrapping value estimates depends on
the difference: a truncated final state still has future value the agent never
got to collect, so trainers bootstrap through truncation but not through
termination. Our rule is the standard one — termination takes priority on the
cap step:

```text
truncated = !terminated && steps >= MAX_STEPS
```

### `observation_space` / `action_space`: the contract a generic trainer reads

A trainer that works on *any* environment cannot hard-code "observations are 9
floats, there are 4 actions." It asks. `observation_space` and `action_space`
each return a `SpaceInfo { shape, space_type }`, and that is how a generic
trainer sizes the input and output layers of its network without knowing
anything else about your env. `SpaceType` is either:

- `Discrete(n)` — a choice among *n* options (our action space: `Discrete(4)`).
- `Box` — a continuous vector (our observation space: nine `f32` cells, reported
  as `Box` even though the underlying position is discrete — the *observation
  vector* is real-valued, which is what the network cares about).

This is the same convention the built-in envs use: `CartPole` reports a `Box`
observation with `Discrete` actions. Get these two methods wrong and a generic
trainer silently builds a mis-sized network — so they are worth a test.

### `render` and `close`: present but optional

Not every environment needs a visualizer or teardown, but the trait still
requires the methods so callers can rely on them existing. `render` returns
`Vec<u8>` (image bytes, an ASCII frame, whatever — returning `Vec::new()` means
"nothing to draw"), and `close` releases any resources (an empty body is fine
for a pure-Rust sim like ours). They must be *present*; they may be trivial.

### `clone_state` / `restore_state`: snapshot-and-restore for lookahead

These two are the reason this tutorial exists. `clone_state` returns a snapshot;
`restore_state` rewinds the env to a snapshot. Together they let a caller **try a
move, look at where it leads, then take it back** — which is exactly what
tree-search planners like MCTS do: from the current state, roll a hypothetical
trajectory forward, record the return, restore the snapshot, and try a different
first move. Without snapshot/restore the only way to "undo" a step is to
`reset()` and replay the entire episode from the start, which is both slow and,
for a stochastic env, not even reproducible.

The promise `restore_state` makes is the **determinism contract**, and it has two
flavors:

- **Fully deterministic env (no RNG in `step`)** — the strong guarantee:
  `restore_state(&snap)` followed by `step(a)` produces a **bit-identical**
  `StepResult` every time. Our gridworld is in this class.
- **Env with RNG in `step`** (random spawns, dice rolls) — the snapshot captures
  the *simulation* state but usually **not** the RNG, so results reproduce only
  across steps that do not draw from it. Built-in `Snake` and `Pong` document
  this weaker contract.

Our env holds a seeded `StdRng` — to mirror the built-in envs' surface and to
reserve the field for a future stochastic variant — but **`step` never draws from
it**, so we are in the strong-guarantee class. Because the RNG never influences a
step, the snapshot deliberately *excludes* it: `TinyGridState` carries only the
agent position and step counter. Spelling out which contract you provide is part
of documenting your env.

## The code

Below is the complete environment and a short demonstration. It is a doc-test, so
it compiles and runs against the current API every time CI runs — the code here
cannot silently rot. Copy it into a `fn main()` (and `cargo add thrust-rl rand`)
and it runs unchanged.

```rust
use rand::{SeedableRng, rngs::StdRng};
use thrust_rl::env::{Environment, SpaceInfo, SpaceType, StepInfo, StepResult};

// --- Fixed, episode-independent parameters --------------------------------
const GRID_SIZE: usize = 3; // a 3x3 grid
const NUM_CELLS: usize = GRID_SIZE * GRID_SIZE; // 9 one-hot observation slots
const NUM_ACTIONS: usize = 4; // Up, Right, Down, Left
const GOAL_ROW: usize = GRID_SIZE - 1; // bottom-right corner
const GOAL_COL: usize = GRID_SIZE - 1;
const GOAL_REWARD: f32 = 1.0; // reaching the goal
const STEP_PENALTY: f32 = -0.01; // every other step (encourages short paths)
const MAX_STEPS: usize = 50; // truncation cap

// --- The snapshot type ----------------------------------------------------
// Exactly the fields needed to rewind the simulation — no RNG, because `step`
// never consults it, so restoring position + step count reproduces the future
// bit-for-bit.
#[derive(Debug, Clone)]
struct TinyGridState {
    row: usize,
    col: usize,
    steps: usize,
}

// --- The environment ------------------------------------------------------
struct TinyGrid {
    row: usize,
    col: usize,
    steps: usize,
    // Seeded RNG held to match the built-in envs' surface and reserved for a
    // future stochastic variant. `step` never draws from it, so this env is
    // fully deterministic and the snapshot omits it.
    #[allow(dead_code)]
    rng: StdRng,
}

impl TinyGrid {
    fn new(seed: u64) -> Self {
        let mut env = Self { row: 0, col: 0, steps: 0, rng: StdRng::seed_from_u64(seed) };
        env.reset();
        env
    }

    // Flattened index of the agent's cell, used as the hot one-hot slot.
    fn cell_index(&self) -> usize {
        self.row * GRID_SIZE + self.col
    }

    // Decode an action into a (drow, dcol) delta. Out-of-range actions decode
    // to (0, 0): a no-op that still counts as a (penalized) step.
    fn action_delta(action: i64) -> (i64, i64) {
        match action {
            0 => (-1, 0), // Up
            1 => (0, 1),  // Right
            2 => (1, 0),  // Down
            3 => (0, -1), // Left
            _ => (0, 0),  // invalid -> stay put
        }
    }
}

impl Environment for TinyGrid {
    type Action = i64;
    type State = TinyGridState;

    fn reset(&mut self) {
        // Return the agent to the start cell and zero the step counter. Note
        // we do NOT rebuild the RNG or re-read the seed: episode-independent
        // setup stays in `new`.
        self.row = 0;
        self.col = 0;
        self.steps = 0;
    }

    fn get_observation(&self) -> Vec<f32> {
        // One-hot over the 9 cells: 1.0 at the agent's cell, 0.0 elsewhere.
        let mut obs = vec![0.0; NUM_CELLS];
        obs[self.cell_index()] = 1.0;
        obs
    }

    fn step(&mut self, action: i64) -> StepResult {
        // Move, clamping into bounds: bumping a wall keeps the agent in place.
        let (drow, dcol) = Self::action_delta(action);
        let new_row = self.row as i64 + drow;
        let new_col = self.col as i64 + dcol;
        if new_row >= 0
            && new_row < GRID_SIZE as i64
            && new_col >= 0
            && new_col < GRID_SIZE as i64
        {
            self.row = new_row as usize;
            self.col = new_col as usize;
        }
        self.steps += 1;

        // Reward + termination: reaching the goal ends the episode with the
        // goal reward; any other move costs the step penalty.
        let at_goal = self.row == GOAL_ROW && self.col == GOAL_COL;
        let (reward, terminated) =
            if at_goal { (GOAL_REWARD, true) } else { (STEP_PENALTY, false) };

        // Truncation only fires if we did NOT already terminate — termination
        // takes priority even on the cap step.
        let truncated = !terminated && self.steps >= MAX_STEPS;

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated,
            truncated,
            info: StepInfo::default(),
        }
    }

    fn observation_space(&self) -> SpaceInfo {
        // Nine real-valued one-hot slots: a Box, even though position is
        // discrete (the observation VECTOR is what the network sees).
        SpaceInfo { shape: vec![NUM_CELLS], space_type: SpaceType::Box }
    }

    fn action_space(&self) -> SpaceInfo {
        // Four discrete moves.
        SpaceInfo { shape: vec![NUM_ACTIONS], space_type: SpaceType::Discrete(NUM_ACTIONS) }
    }

    fn render(&self) -> Vec<u8> {
        // Nothing to draw for this pure-Rust sim.
        Vec::new()
    }

    fn close(&mut self) {
        // No resources to release.
    }

    fn clone_state(&self) -> TinyGridState {
        // Capture only the simulation state — no RNG (see the determinism note).
        TinyGridState { row: self.row, col: self.col, steps: self.steps }
    }

    fn restore_state(&mut self, state: &TinyGridState) {
        self.row = state.row;
        self.col = state.col;
        self.steps = state.steps;
    }
}

// === Using the environment ================================================

// 1. Construct once, then inspect the spaces exactly as a generic trainer
//    would to size its network's input and output layers.
let mut env = TinyGrid::new(0);
assert_eq!(env.observation_space().shape, vec![NUM_CELLS]);
assert!(matches!(env.action_space().space_type, SpaceType::Discrete(NUM_ACTIONS)));

// 2. reset() starts a fresh episode: the agent is at cell 0 (one-hot slot 0).
env.reset();
assert_eq!(env.get_observation()[0], 1.0);

// 3. A wall bump: Up from the top row is a no-op that still costs a step and
//    never terminates.
let bump = env.step(0); // Up, but already on row 0
assert_eq!(bump.reward, STEP_PENALTY);
assert!(!bump.terminated);
assert_eq!(env.get_observation()[0], 1.0, "agent stayed at the start cell");

// 4. Walk to the goal: Right, Right, Down, Down -> (2, 2).
env.reset();
env.step(1); // Right -> (0, 1)
env.step(1); // Right -> (0, 2)
env.step(2); // Down  -> (1, 2)
let arrival = env.step(2); // Down -> (2, 2) == goal
assert_eq!(arrival.reward, GOAL_REWARD);
assert!(arrival.terminated, "reaching the goal terminates the episode");
assert!(!arrival.truncated, "termination is not truncation");

// 5. The determinism contract, demonstrated: snapshot, roll a trajectory
//    forward, rewind, and replay the first move. Because `step` is fully
//    deterministic, the replayed StepResult is bit-identical to the original.
env.reset();
env.step(1); // advance to a non-initial state (0, 1)
let snapshot = env.clone_state();

let first = env.step(2); // Down -> (1, 1)
// Continue the rollout to prove the snapshot is independent of later steps.
env.step(1);
env.step(2);

// Rewind to the snapshot and replay the same action.
env.restore_state(&snapshot);
let replay = env.step(2);

assert_eq!(first.observation, replay.observation, "observation reproduces bit-for-bit");
assert_eq!(first.reward, replay.reward, "reward reproduces bit-for-bit");
assert_eq!(first.terminated, replay.terminated);
assert_eq!(first.truncated, replay.truncated);
```

That is a complete, working Thrust environment in well under a hundred lines. It
plugs into the same machinery as the built-in envs — you could hand a `TinyGrid`
to the DQN loop from [Tutorial 3](03-dqn.md) unchanged (nine-dim observation,
four discrete actions) and watch it learn the shortest path.

## The determinism contract, restated

The step-5 block above is the contract in miniature, and it is worth being
explicit about *why* it holds:

1. `step` reads only `self.row`, `self.col`, and `self.steps` — never the RNG.
2. `clone_state` captures exactly those three fields.
3. Therefore `restore_state(&snap)` puts the env into a state from which the next
   `step(a)` is a pure function of `(row, col, steps, a)` — no hidden inputs, so
   the result is bit-identical no matter how many other trajectories you rolled
   in between.

If you later add randomness to `step` (a slippery floor, a random goal), you have
a choice to make, and you must document it: either capture the RNG in
`TinyGridState` too (restoring the *exact* stochastic future, at the cost of a
bigger, `Clone`-heavy snapshot) or leave it out (the weaker contract the built-in
`Snake` and `Pong` document). The trait does not decide for you — the per-env
documentation is where that promise lives.

## Try it yourself

- **Add a hazard.** Introduce a hole cell that terminates with a `-1.0` reward,
  mirroring the built-in
  [`GridWorld`](../../src/env/games/grid_world.rs). Watch the reward/termination
  branch in `step` grow one arm.
- **Randomize the start.** Give `reset` a random start cell drawn from the held
  `rng`. Now `reset` is stochastic — but note `step` still is not, so the
  snapshot contract is unchanged. This is the cleanest way to see why the RNG
  lives outside the snapshot.
- **Grow the grid.** Bump `GRID_SIZE` to 5. The observation space
  (`NUM_CELLS = 25`) and the network a generic trainer builds resize
  automatically, because everything reads `observation_space()` — proof that the
  space methods are the real contract, not the hard-coded `9`.
- **Train it.** Feed `TinyGrid` to the DQN loop from
  [Tutorial 3](03-dqn.md): swap `CartPole` for `TinyGrid`, drop `gamma` to
  ~`0.95` for the sparse reward, and confirm the agent finds the four-step path.

## Next

You can now write any environment Thrust can train on. See the
[tutorial index](README.md) for the full series; the remaining tutorials take a
*trained* policy the rest of the way — including exporting it to run in the
browser.
