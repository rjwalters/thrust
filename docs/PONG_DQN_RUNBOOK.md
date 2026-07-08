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

### Run 2 — Adam 6.25e-5 (corrected LR)

_Placeholder — to be filled in post-run. This is the single-variable rerun
isolating the LR change (6.25e-5 vs. run 1's 2.5e-4); no other hyperparameter
changes. Append the crossing-zero step and final mean score here._

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
