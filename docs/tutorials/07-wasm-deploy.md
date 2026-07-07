# Tutorial 7 — Train in Rust, run in the browser

> **You will:** take a policy trained by any earlier tutorial and export it to
> the JSON *inference format* that Thrust's WebAssembly demo consumes — including
> the one weight-layout gotcha that silently breaks a naive export — then wire
> the JSON into the running browser demo without touching a line of TypeScript.
> **Prerequisites:** any tutorial that produces a trained `MlpBurnPolicy` —
> [Tutorial 2](02-cartpole-ppo.md) (CartPole + PPO) is the natural pairing,
> since the demo ships a CartPole page. You need to recognize `MlpBurnPolicy`
> and the `MlpBurnConfig` surface; nothing about the browser stack is assumed.
> **Time:** ~15 minutes.

This is the last tutorial in the series, and it closes the loop the first six
opened: you have *trained* policies — now you *ship* one. The destination is the
live WebAssembly demo under [`web/`](../../web), where a `CartPole`, `Pong`, or
`Snake` agent runs its forward pass entirely in the browser, at 60 FPS, with no
server round-trip.

The thing to understand before any code is *why deployment needs a second model
format at all.*

## Why a separate inference format

Training and inference in Thrust deliberately run on **two different stacks**:

- **Training** uses [Burn](https://burn.dev), a full tensor framework with
  autodiff, optimizers, and backend-swappable kernels (NdArray on CPU, WGPU on
  GPU). That machinery is exactly what you want while learning — and exactly what
  you do *not* want to compile to WebAssembly, because it drags a large,
  GPU-flavored dependency tree into a bundle that has to download over the wire.
- **Inference** in the browser uses a tiny, pure-Rust struct —
  [`InferenceModel`](../../src/policy/inference.rs) — that holds the weights as
  plain `Vec<Vec<f32>>` matrices and does the forward pass with nested `for`
  loops. No Burn, no tensors, no autodiff: it compiles to a small WASM module and
  runs a CartPole forward pass in well under a millisecond.

The bridge between the two stacks is a **JSON file**. You train with Burn, export
the learned weights into an `InferenceModel`, serialize it to JSON, and the WASM
demo deserializes that JSON at page load. The JSON is the contract: as long as it
has the right field names and shapes, the browser never needs to know Burn
existed.

> **A note on `InferenceModel` vs `ExportedModel`.** The crate also contains a
> `crate::inference::ExportedModel` type — a simpler, single-layer format that is
> **not** what the WASM demo loads. The bindings in `src/wasm.rs`
> (`WasmCartPole`, `WasmPong`, `WasmSnake`) all deserialize into
> `thrust_rl::policy::inference::InferenceModel`. That is the type this tutorial
> targets, and the type whose field names you will see in
> [`web/public/cartpole_model_best.json`](../../web/public/cartpole_model_best.json).

## The inference format, field by field

`InferenceModel` mirrors a 2-layer shared-trunk actor-critic — the architecture
`MlpBurnConfig { num_layers: 2, .. }` builds:

```text
InferenceModel {
    obs_dim:    usize,   // network input width  (CartPole: 4)
    action_dim: usize,   // policy-head output   (CartPole: 2)
    hidden_dim: usize,   // trunk width          (e.g. 64)
    activation: InferenceActivation,  // "Tanh" or "ReLU", applied on the trunk

    // Trunk (shared between policy and value heads)
    shared_fc1_weight: Vec<Vec<f32>>,  // [hidden_dim × obs_dim]
    shared_fc1_bias:   Vec<f32>,       // [hidden_dim]
    shared_fc2_weight: Vec<Vec<f32>>,  // [hidden_dim × hidden_dim]
    shared_fc2_bias:   Vec<f32>,       // [hidden_dim]

    // Heads
    policy_weight: Vec<Vec<f32>>,      // [action_dim × hidden_dim]  -> action logits
    policy_bias:   Vec<f32>,           // [action_dim]
    value_weight:  Vec<Vec<f32>>,      // [1 × hidden_dim]           -> scalar value
    value_bias:    Vec<f32>,           // [1]

    metadata: Option<TrainingMetadata>, // provenance; optional
}
```

Every weight matrix is stored **out-major**: `weight[o][i]` is the connection
from input neuron `i` to output neuron `o`, so each *row* is one output neuron's
full set of incoming weights. That row-major convention is what `InferenceModel::forward`
iterates over, and it is the crux of the one gotcha in this tutorial.

### The weight-layout gotcha

Burn's `Linear` layer stores its weight tensor with shape `[in_features,
out_features]` — the transpose of what `InferenceModel` wants. If you copy the
flat Burn buffer straight into an `InferenceModel` row, every weight lands in the
wrong cell and the exported policy plays like noise, even though the JSON is
"valid" and loads without error. This is the single most common way a WASM export
goes silently wrong.

The fix is a transpose during the copy. Burn's flat buffer indexes as
`flat[i * out_features + o]` (input-major); `InferenceModel` wants `rows[o][i]`
(output-major). So:

```text
for o in 0..out_features {
    for i in 0..in_features {
        rows[o][i] = flat[i * out_features + o];
    }
}
```

The reference implementation lives in `extract_linear()` in
[`examples/games/pong/eval_pong.rs`](../../examples/games/pong/eval_pong.rs),
which is what actually produced the shipped Pong weights. The doc-test below is
the same transpose, distilled to the essentials.

## The code

The snippet below is the complete export path, and it is a doc-test — it compiles
and runs against the current API on every CI run, so the transpose can never
silently rot out of sync with Burn's tensor layout.

One deliberate shortcut: **it does not train.** In production you train first
(that is what Tutorials 2–5 are for) and *then* export; a real training run takes
minutes to hours, which is far too slow for a doc-test. Since the export
machinery — layer extraction, transpose, JSON round-trip — is completely
independent of *what* the weights are, we exercise it on a freshly initialized
(random) policy. The structure is identical; only the numbers would differ after
training. Copy it into a `fn main()` (and `cargo add thrust-rl`, plus `burn` and
`tempfile`) and it runs unchanged.

```rust
use burn::{
    backend::{Autodiff, NdArray},
    nn::Linear,
};
use thrust_rl::policy::inference::{InferenceActivation, InferenceModel};
use thrust_rl::prelude::*;

type Backend = Autodiff<NdArray<f32>>;

// --- Architecture (must match the trained policy you are exporting) --------
const OBS_DIM: usize = 4; // CartPole observation width
const ACTION_DIM: usize = 2; // CartPole: push left / right
const HIDDEN_DIM: usize = 64; // trunk width

let device = Default::default();

// --- The policy ------------------------------------------------------------
// In a real workflow this is the policy you trained in Tutorial 2 and loaded
// back from a Burn `.bin` record. Here we build a fresh, randomly initialized
// one: the export path is identical either way, and a doc-test cannot afford a
// training run. `Tanh` matches the activation the shipped CartPole model uses.
let config = MlpBurnConfig {
    num_layers: 2,
    hidden_dim: HIDDEN_DIM,
    use_orthogonal_init: true,
    activation: BurnActivation::Tanh,
    seed: Some(0),
};
let policy = MlpBurnPolicy::<Backend>::with_config(OBS_DIM, ACTION_DIM, config, &device);

// --- Extract one Burn Linear layer as an out-major (weight, bias) pair -----
// Burn stores the weight tensor as [in_features, out_features] in a flat,
// input-major buffer. InferenceModel wants out-major rows, weight[out][in], so
// we transpose during the copy. Get this wrong and the JSON still "loads" — it
// just plays like noise.
fn extract_linear(layer: &Linear<Backend>) -> (Vec<Vec<f32>>, Vec<f32>) {
    let weight = layer.weight.val();
    let [in_features, out_features] = weight.dims();
    let flat: Vec<f32> = weight.into_data().to_vec().expect("weight to_vec");

    let mut rows = vec![vec![0.0_f32; in_features]; out_features];
    for o in 0..out_features {
        for i in 0..in_features {
            // Burn flat index: flat[i * out_features + o]  (in = i, out = o).
            rows[o][i] = flat[i * out_features + o];
        }
    }

    // A Linear may or may not carry a bias; default to zeros when it doesn't.
    let bias = layer
        .bias
        .as_ref()
        .map(|b| b.val().into_data().to_vec::<f32>().expect("bias to_vec"))
        .unwrap_or_else(|| vec![0.0_f32; out_features]);

    (rows, bias)
}

// --- Build the InferenceModel from the four extracted layers ---------------
// The layer accessors (fc1/fc2/policy_head/value_head) expose exactly the four
// Linear layers of the shared-trunk actor-critic.
let (shared_fc1_weight, shared_fc1_bias) = extract_linear(policy.fc1());
let (shared_fc2_weight, shared_fc2_bias) = extract_linear(policy.fc2());
let (policy_weight, policy_bias) = extract_linear(policy.policy_head());
let (value_weight, value_bias) = extract_linear(policy.value_head());

let model = InferenceModel {
    obs_dim: OBS_DIM,
    action_dim: ACTION_DIM,
    hidden_dim: HIDDEN_DIM,
    activation: InferenceActivation::Tanh,
    metadata: None, // set TrainingMetadata here to record provenance
    shared_fc1_weight,
    shared_fc1_bias,
    shared_fc2_weight,
    shared_fc2_bias,
    policy_weight,
    policy_bias,
    value_weight,
    value_bias,
};

// Sanity-check the shapes before serializing: each weight is out-major, so its
// row count is the output width and each row's length is the input width.
assert_eq!(model.shared_fc1_weight.len(), HIDDEN_DIM); // [hidden × obs]
assert_eq!(model.shared_fc1_weight[0].len(), OBS_DIM);
assert_eq!(model.policy_weight.len(), ACTION_DIM); // [action × hidden]
assert_eq!(model.policy_weight[0].len(), HIDDEN_DIM);
assert_eq!(model.value_weight.len(), 1); // [1 × hidden]

// --- Round-trip through JSON, exactly as the browser will ------------------
// `save_json` writes the same wire format the WASM demo fetches; `load_json`
// parses it back. We use a temp file so the doc-test leaves nothing behind; in
// practice you write straight to web/public/<name>.json (see below).
let file = tempfile::NamedTempFile::new().expect("temp file");
model.save_json(file.path()).expect("save_json");
let loaded = InferenceModel::load_json(file.path()).expect("load_json");

// The four fields the demo keys on survive the round-trip intact.
assert_eq!(loaded.obs_dim, OBS_DIM);
assert_eq!(loaded.action_dim, ACTION_DIM);
assert_eq!(loaded.hidden_dim, HIDDEN_DIM);
assert!(matches!(loaded.activation, InferenceActivation::Tanh));

// And the loaded model actually runs a forward pass: obs -> (logits, value).
let (logits, value) = loaded.forward(&[0.0, 0.0, 0.0, 0.0]);
assert_eq!(logits.len(), ACTION_DIM);
assert!(value.is_finite());
```

That is the entire native side of deployment: extract, transpose, build,
serialize. The JSON `save_json` produces has the top-level keys `obs_dim`,
`action_dim`, `hidden_dim`, `activation`, the six weight/bias fields, and an
optional `metadata` block — byte-for-byte the structure of the shipped
[`web/public/cartpole_model_best.json`](../../web/public/cartpole_model_best.json).

## Wiring it into the browser

The WASM demo is already fully wired — there is **no TypeScript to write.** The
CartPole page fetches its weights by a fixed filename at startup and hands the
raw JSON to the WASM binding:

```js
// web/src/components/CartPole/useCartPole.ts (already in the repo)
const response = await fetch(`${import.meta.env.BASE_URL}cartpole_model_best.json`);
const modelJson = await response.text();
envRef.current.load_policy_json(modelJson); // WasmCartPole::load_policy_json
```

`load_policy_json` on the Rust side (`src/wasm.rs`) is just
`InferenceModel::load_json` over a string — the same deserialize the doc-test
round-trips. So "deploying your policy" means putting your JSON where that
`fetch` looks for it. End to end:

```bash
# 1. Export your trained policy to the filename the demo already expects.
#    (In your own binary, call model.save_json("web/public/cartpole_model_best.json").)

# 2. Build the WASM bundle from the crate root. --no-default-features drops the
#    Burn training stack; the `wasm` feature pulls in only the inference path.
wasm-pack build --target web --features wasm --no-default-features

# 3. Start the demo's dev server.
cd web && pnpm install && pnpm dev
```

Then open the printed local URL. On load the CartPole page fetches
`cartpole_model_best.json`, calls `load_policy_json`, and your exported policy
starts balancing the pole — running its forward pass in the browser, no server
involved. To swap in a different policy, overwrite that JSON and reload; the
filename is the only contract. Pong and Snake follow the same pattern with their
own fixed filenames (`pong_model.json`, `snake_model.json`) and bindings
(`WasmPong`, `WasmSnake`).

See [`web/README.md`](../../web/README.md) for the full demo setup and
[`docs/WASM_ROADMAP.md`](../WASM_ROADMAP.md) for the broader
train-export-deploy pipeline.

## Try it yourself

- **Export a *trained* policy.** Run [Tutorial 2](02-cartpole-ppo.md) to
  convergence, save the Burn record, then load it into a blank
  `MlpBurnPolicy::<Backend>::with_config(..)` with `load_file` (see
  `eval_pong.rs`) before the extraction step. The export code above does not
  change — only the weights it reads.
- **Attach provenance.** Fill in `metadata: Some(TrainingMetadata { .. })` with
  the environment name, algorithm, and step count. It rides along in the JSON and
  shows up when you inspect a shipped model.
- **Match the activation.** If you trained with `BurnActivation::ReLU`, export
  with `InferenceActivation::ReLU` — the field controls which nonlinearity
  `InferenceModel::forward` applies. A mismatch here is another silent
  correctness bug, exactly like the transpose.
- **Verify parity.** Before and after export, feed the same observation through
  the Burn policy and the `InferenceModel` and compare the logits. Agreement (to
  float tolerance) proves your transpose and activation are right — the surest
  check that the browser will behave like training did.

## Next

That closes the series: you can install Thrust, train a policy with PPO, DQN,
SAC, or recurrent PPO, write your own environment, and now take a trained policy
all the way to a browser tab. From here, the [Example Gallery](../EXAMPLES.md)
has full runnable trainers for every algorithm, and
[`docs/WASM_ROADMAP.md`](../WASM_ROADMAP.md) covers the deployment pipeline in
depth. See the [tutorial index](README.md) for the whole path at a glance.
