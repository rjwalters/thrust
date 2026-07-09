# Pong DQN Training Runbook (Epic #306, Phase 4 — issue #329)

Operator runbook for the reference-game training run: **Nature-DQN on ALE
Pong**, executed on `alc-2` (RTX 4090 24 GB, CUDA 12.0). The training binary is
`examples/games/atari/train_pong_dqn.rs` (`--example train_pong_dqn`).

This run is **operator-gated** (like #134): the long GPU run needs operator
hardware access and should not be auto-claimed by a builder. The committed
binary + this runbook are the reviewable builder artifacts; the learning-curve
report is committed as a post-run follow-up per the #298 / #306 protocol.

## Success criterion

Pong's random floor is ≈ −21. **Any positive mean episode score beats random**
and is the decisive success signal. Published baselines: DQN ≈ 18.9 (Mnih 2015,
50M **raw frames**) / PPO ≈ 20.7. Under the **sticky-action** protocol (p=0.25,
Machado et al. 2018) crossing zero is typically seen within **10–20M raw
frames** — later than the classic no-sticky DQN curves; the 5M wrapper step
(20M raw frame) budget is sized to cover this range.

**Units — frames vs. wrapper steps.** `AtariPreprocess` applies **frame-skip 4**,
so one wrapper `step()` = 4 raw ALE frames. The loop counts **wrapper steps**
(`TOTAL_TIMESTEPS` is a wrapper-step budget), while the literature above counts
**raw frames**. The default `TOTAL_TIMESTEPS=5_000_000` is therefore
**5M wrapper steps = 20M raw frames**, which covers the 10–20M raw frame band in
which sticky-action Pong typically crosses zero — budgeted for a *positive*
score, not a ceiling score.

## Replay-buffer memory (honest f32 math)

The in-tree `ReplayBuffer` stores both `obs` and `next_obs` as `Vec<f32>`
(4 bytes/value). One Pong observation is `4 * 84 * 84 = 28_224` values, so one
transition's frame storage is `2 * 28_224 * 4 = 225_792` bytes ≈ 221 KiB:

```text
buffer at 1M   : 2 * 28224 * 4 * 1_000_000 = ~210 GiB   (Mnih 2015 — does NOT fit)
buffer at 100k : 2 * 28224 * 4 *   100_000 = ~21.0 GiB  (host RAM — DEFAULT)
buffer at 50k  : 2 * 28224 * 4 *    50_000 = ~10.5 GiB  (host RAM — fallback)
```

The buffer lives in **host RAM**, not VRAM. Confirm host RAM before launching.
If 100k (~21 GiB) is too large, set `BUFFER_CAPACITY=50000` (~10.5 GiB). A future
u8 frame store (before the 1/255 scale) would cut this 4× (~5.3 GiB at 100k) but
requires a new buffer type — out of scope for this issue.

## Hyperparameters (Mnih 2015 adapted; budget = 5M wrapper steps ≈ 20M raw frames)

| Parameter | Mnih 2015 (50M raw frames) | This run (5M wrapper steps ≈ 20M raw frames) |
|---|---|---|
| `learning_rate` | 2.5e-4 (RMSProp) | 6.25e-5 (Adam; Rainbow/Dopamine) |
| `batch_size` | 32 | 32 |
| `buffer_capacity` | 1M | 100_000 (f32 budget) |
| `min_buffer_size` | 50_000 | 10_000 |
| `target_update_interval` | 10_000 | 10_000 (hard copy) |
| `gamma` | 0.99 | 0.99 |
| `epsilon_start` / `epsilon_end` | 1.0 / 0.1 | 1.0 / 0.1 |
| `epsilon_decay_steps` | 1M | 1_000_000 |
| `max_grad_norm` | 10.0 | 10.0 |
| `soft_update_tau` | None (hard sync) | None (hard sync) |
| reward clipping | ±1 | n/a for Pong (rewards already ±1) |

Budget default: `TOTAL_TIMESTEPS=5_000_000` **wrapper steps** (= 20M raw frames;
override via env var).

## Env-var knobs

| Var | Default | Meaning |
|---|---|---|
| `TOTAL_TIMESTEPS` | `5_000_000` | Total **wrapper steps** (frame-skip 4 → ≈ 4× raw frames). |
| `BUFFER_CAPACITY` | `100_000` | Replay capacity (see RAM math). |
| `MIN_BUFFER_SIZE` | `10_000` | Warmup transitions before updates. |
| `LOG_INTERVAL` | `10_000` | Env-step period for stdout logs + CSV rows. |
| `LEARNING_RATE` | `6.25e-5` | Adam learning rate (Atari standard; Rainbow/Dopamine). Must be finite and positive. |
| `CURVE_CSV` | *(unset)* | Path for the `env_steps,mean_episode_reward` CSV. |
| `CHECKPOINT_INTERVAL` | *(unset)* | Env-step period for weight snapshots. |
| `CHECKPOINT_DIR` | `checkpoints` | Directory for `.bin` snapshots. |
| `ATARI_PYTHON` | `python3` | Interpreter with `ale-py` installed. |
| `ATARI_WORKER_SCRIPT` | `envs/atari/ale_worker.py` | Worker path (relative → run from repo root). |
| `ALE_ROM_PATH` | *(unset)* | Override ROM lookup (dir or `.bin` file). |

## Experiment log

### Run 1 — Adam 2.5e-4 (negative result, 2026-07-08)

Stopped at the pre-declared no-learning decision point: **9.56M raw frames
(2.39M wrapper steps, ~6.2 h on alc-2 RTX 4090)**. `avg(last≤100)` never
exceeded −20.27 and sat at −20.8 ± 0.1 from ~1M wrapper steps onward (ε
floor). Greedy policy marginally worse than random — flat curve with zero
upward trend over 1.4M post-floor steps.

**Diagnosis:** Adam 2.5e-4 is the Mnih 2015 RMSProp rate; the Atari-standard
Adam rate is 6.25e-5 (Rainbow, Dopamine). The hyperparameter table's Adam
column was copied from the RMSProp column without adaptation (see issue #342).

**Resolution:** Default changed to 6.25e-5 in this PR. See run 2 below for
the corrected-LR result. Artifacts preserved on alc-2 under
`~/pong_dqn_run1_lr2.5e-4/` (curve CSV, full log, 4 checkpoints at 500k
intervals).

### Run 2 arm A — Adam 6.25e-5 (corrected LR, alc-2, 2026-07-08)

Single-variable rerun isolating the LR change (6.25e-5 vs. run 1's 2.5e-4); no
other hyperparameter changes. **Textbook learning curve.** Liftoff ~200k wrapper
steps (first >−19 at 590k), −15 at the ε floor (~1M steps), plateau near −8
(~1.9–2.6M), then a climb toward −5.5. **Final `avg(last≤100) = −5.14 at exactly
5M wrapper steps (20M raw frames)** — STILL IMPROVING at budget exhaustion; the
last 400k window oscillates −6.3 → −3.9 → −5.1 with a rising envelope, best
strict 400k-step trailing mean (40 logged points at 10k spacing) **−4.13**,
single best logged 10k-step point −3.44 at 4.67M steps. **Did not cross zero**
within budget (expected — see gap analysis). Stats: 2,289 episodes, 14.0 h wall
clock, ~99 wrapper steps/s. Artifacts: `alc-2:~/pong_dqn_run2_lr6.25e-5/` (curve
CSV, full log, 9 checkpoints).

### Run 2 arm B — Adam 1e-4 (parallel LR-stability arm, alc-8, 2026-07-09)

Parallel arm with `LEARNING_RATE=1e-4` (all else identical) to probe LR
stability. Tracks arm A closely to ~2M steps, then shows **repeated instability
dips** arm A never had: a −8.9 to −10.9 trough at ~2.0–2.6M, then two more
−8.5 to −9.1 excursions at ~3.1M and ~4.0M. Late rally (best point −5.91 at
3.77M) before settling to **final `avg(last≤100) = −7.21` at 5M wrapper steps**
(best 400k window −6.73). Ends **worse than arm A** despite an extra hour of wall
clock. Stats: 2,411 episodes, 15.0 h wall clock (53,968 s), ~93 wrapper steps/s.
Artifacts: `alc-8:~/pong_dqn_run2b_lr1e-4/`.

**LR-stability verdict:** 6.25e-5 is the more stable default — arm A climbs with
a monotone rising envelope and ends higher (−5.14, still rising) while arm B's
1e-4 shows repeated multi-point regressions and settles lower (−7.21).

Full report, gap analysis, and the concrete path to a zero-crossing (u8 frame
buffer → 1M capacity; longer budget):
[`research/2026-07-pong-dqn-learning-curve.md`](research/2026-07-pong-dqn-learning-curve.md).

## Phase 0 — pre-flight (once, before the run)

```bash
# Confirm ale-py importable by the chosen interpreter
/usr/bin/python3 -c "import ale_py; print(ale_py.__version__)"

# Confirm the Pong ROM resolves (ale-py 0.12 ships ROMs via get_rom_path)
/usr/bin/python3 -c "from ale_py import roms; print(roms.get_rom_path('pong'))"

# Confirm CUDA visible
nvidia-smi
```

If `ale-py` is not present: `pip install ale-py` (it bundles ROMs; otherwise
`AutoROM --accept-license`). If the ROM does not resolve, set `ALE_ROM_PATH` to
a directory containing `pong.bin`.

### CUDA on non-alc-2 nodes (nvrtc pip-wheel recipe)

Only `alc-2` has the apt CUDA toolkit (`nvidia-cuda-toolkit`) installed. Other
alc-* nodes have the NVIDIA driver but no toolkit, so `--features cuda` fails to
find `libnvrtc` / CUDA headers at build/run time. Run 2 arm B was executed on
`alc-8` after provisioning it this way (proven working):

```bash
# 1. nvrtc runtime via pip wheel (no apt/root needed)
pip install nvidia-cuda-nvrtc-cu12

# 2. Stage CUDA toolkit headers into a user-writable root
mkdir -p ~/cuda-root/include
# copy/extract the CUDA headers (cuda.h, nvrtc.h, etc.) into ~/cuda-root/include
# (from a wheel, a tarball, or rsync'd from alc-2's /usr/local/cuda/include)

# 3. Point Burn/cubecl at the staged root and the wheel's shared libs
export CUDA_PATH=~/cuda-root
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cuda_nvrtc, os; print(os.path.join(os.path.dirname(nvidia.cuda_nvrtc.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

With those three exported, `cargo run --features "training,env-atari,cuda"`
builds and runs on the node's RTX 4090 exactly as on alc-2. See also the
node-provisioning note in [`BURN_BACKENDS.md`](BURN_BACKENDS.md).

## Phase 1 — start the training run (in a tmux session)

```bash
# On alc-2, from the repo root
tmux new-session -s pong_dqn

# cargo may not be on PATH in a fresh tmux on alc-* nodes:
source ~/.cargo/env    # or: source ~/.bashrc

# ale-py lives at /usr/bin/python3 on alc-2, not the default python3
export ATARI_PYTHON=/usr/bin/python3

# Learning-curve CSV output
export CURVE_CSV="$HOME/pong_dqn_curve.csv"

# Checkpoint every 500k steps
export CHECKPOINT_INTERVAL=500000
export CHECKPOINT_DIR="$HOME/pong_dqn_checkpoints"

# Run the training binary (cuda feature selects the RTX 4090).
# Run from the repo root so the default relative worker path resolves.
cargo run --release \
  --features "training,env-atari,cuda" \
  --example train_pong_dqn \
  2>&1 | tee "$HOME/pong_dqn_run.log"
```

### Throughput expectation

**Convention.** The code's `fps=` log field measures **wrapper steps/sec**
(`total_env_steps / elapsed`), not raw frames/sec. Everything below is stated in
that unit first, then converted; with frame-skip 4, raw frame/s = 4 × wrapper
steps/s, and wall-clock = `5M wrapper_steps / (wrapper_steps_per_sec)`.

CPU (NdArray) throughput on this stack is ~1 wrapper-step/sec after warmup — a
full 1.6M-param CNN forward+backward per step (validated in the builder's smoke
run), so the CPU path is smoke-test-only and its rate cannot predict the GPU
rate. On the RTX 4090 the plausible band spans two cases:

| Case | `fps=` (wrapper steps/s) | Raw frames/s (×4) | Wall-clock for 5M wrapper steps |
|---|---|---|---|
| Optimistic (GPU-bound) | 300–600 | 1_200–2_400 | ~2.5–5 h |
| Conservative (IPC-bound) | 75–150 | 300–600 | ~9–18 h |

The subprocess IPC overhead is partly amortized by the frame-skip, but whether
the run lands GPU-bound (optimistic) or IPC-bound (conservative) is not something
the CPU smoke run can tell you in advance. **Read the actual `fps=` value from
the first few log lines and derive the wall-clock from the table above:** e.g.
`fps=100` ⇒ ~14 h, `fps=400` ⇒ ~3.5 h. If `fps=` is far below the 75/s floor,
the subprocess IPC is the suspect — profile before letting the full run continue.

**Measured (runs 1 & 2):** the actual runs landed in the **conservative
(IPC-bound)** band — ~99 wrapper steps/s on arm A, ~93 on arm B, with GPU
utilization sampled at only **~60 %**. The bottleneck is the subprocess `ale-py`
adapter's per-step IPC, not the CNN on the 4090. A single host is therefore **not
GPU-saturated** by this workload (see the
[#281](https://github.com/rjwalters/thrust/issues/281) distributed-training
re-triage). Wall clock was ~14 h (arm A) / ~15 h (arm B) for the 5M-step budget.

### Where results land

- Stdout / tmux pane — progress every `LOG_INTERVAL` steps (with `fps=`).
- `$HOME/pong_dqn_run.log` — full log.
- `$HOME/pong_dqn_curve.csv` — learning-curve CSV.
- `$HOME/pong_dqn_checkpoints/pong_dqn_<step>.bin` — weight snapshots.

## Phase 2 — commit results (post-run, builder/operator follow-up)

Per the #298 / epic #306 report protocol, after the run commit:

- `docs/research/2026-07-pong-dqn-learning-curve.md` — honest report: hardware,
  wall-clock, env steps, final mean score, learning-curve discussion, and a gap
  analysis vs Mnih 2015. If Pong goes positive, report the step at which it
  crossed zero and the final score. If not, report an honest gap analysis
  (current score, convergence shape, likely blockers).
- `docs/research/data/2026-07-pong_dqn_curve.csv` — the raw learning-curve CSV.

## Local smoke test (reproduces the artifact acceptance gate)

Requires a local `ale-py`. Tiny budget, CPU backend, small buffer/warmup so it
finishes in minutes and exercises the full lifecycle (worker connect → buffer
fill → DQN updates → CSV → checkpoint → clean exit):

```bash
TOTAL_TIMESTEPS=400 MIN_BUFFER_SIZE=100 BUFFER_CAPACITY=400 \
LOG_INTERVAL=50 CHECKPOINT_INTERVAL=200 CHECKPOINT_DIR=/tmp/pong_ckpt \
CURVE_CSV=/tmp/pong_smoke.csv ATARI_PYTHON=/path/to/venv/bin/python3 \
cargo run --release --features "training,env-atari" --example train_pong_dqn
```

Verify `/tmp/pong_smoke.csv` has a header row plus at least one data row, and
that `/tmp/pong_ckpt/` contains `pong_dqn_*.bin` snapshots.
