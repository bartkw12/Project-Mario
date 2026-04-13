# Project Mario — Gameplan (Stable Reference)

> **Purpose**: High-level project reference. Rarely changes.
> For current status, blockers, experiments, and session notes → see `progress_log.md`.

---

## 1. Vision

**Primary Goal**: Train a PPO agent that reliably completes Super Mario Bros Level 1-1.

**Future Goals** (stretch, post-completion):
1. Generalized agent across multiple levels (1-1 → 1-4+)
2. Beat the entire game (Worlds 1–8)
3. Rival human speedrun times on a single level

---

## 2. Stack

| Component          | Library                        | Version    |
|--------------------|--------------------------------|------------|
| Environment        | `gym-super-mario-bros`         | 7.4.0      |
| Gym API            | `gymnasium`                    | >= 0.29    |
| Compat shim        | `shimmy[gym-v21]`              | >= 1.0     |
| NES emulator       | `nes-py`                       | 8.2.1      |
| RL framework       | `stable-baselines3`            | >= 2.3     |
| Deep learning      | `torch`                        | >= 2.1     |
| Config             | `dataclasses` + YAML           | —          |
| Experiment track   | `tensorboard`                  | —          |
| Video recording    | `gymnasium.wrappers.RecordVideo` | —        |
| Python             | 3.10–3.12                      | —          |

**Rejected**: `gym` (dead), CleanRL as primary (ref only for Phase 4), `torch 1.x`.

---

## 3. Project Structure

```
project-mario/
├── README.md
├── gameplan.md              ← this file (stable)
├── progress_log.md          ← volatile session state
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── experiments/
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── wrappers.py      # SkipFrame, reward shaping
│   │   └── mario_env.py     # Environment factory
│   ├── train.py
│   ├── evaluate.py
│   ├── callbacks.py
│   └── config.py            # YAML → dataclass
├── results/
│   ├── models/
│   ├── logs/
│   └── videos/
├── notebooks/
│   └── analysis.ipynb
└── legacy/
```

---

## 4. Phases & Success Criteria

### Phase 1 — Foundation & Environment Stack
**Hardware**: CPU only

Build clean project structure, modern deps, Gymnasium-compatible Mario env with full
wrapper stack (JoypadSpace → SkipFrame → Grayscale → Resize → FrameStack), shimmy
compat, seeding, YAML config + dataclass loader.

**Done when**: `train.py --dry-run` runs 100 steps, prints obs shape `(n_envs, 4, 84, 84)`.
Random-agent video saved to `results/videos/`.

### Phase 2 — Baseline PPO Training (1-1)
**Hardware**: 3080 Ti

PPO via SB3, 8 parallel envs, TensorBoard, EvalCallback. Simple reward shaping
(+1 forward, -15 death, +50 flag). 5–10M timesteps.

**Baseline config** (see `configs/default.yaml` once created):
- `CnnPolicy`, `lr=2.5e-4`, `n_steps=512`, `batch=256`, `epochs=4`
- `gamma=0.99`, `gae_lambda=0.95`, `clip=0.2`, `ent_coef=0.01`
- `num_envs=8`, `frame_skip=4`, `frame_stack=4`, `obs=84×84 gray`

**Progress benchmarks (x_pos by timesteps):**

| Steps | x_pos avg | Milestone                        |
|-------|-----------|----------------------------------|
| 500K  | 300–500   | Runs right, may clear first pipe |
| 1M    | 500–800   | Clearing pipes consistently      |
| 2M    | 800–1200  | Approaching first gap            |
| 5M    | 2000–3160 | Late-level or flag               |
| 10M   | 3160+     | Reliable flag capture            |

**Solved = ≥80% flag capture over 50 eval episodes.**

**Key metrics**: `ep_rew_mean`, `ep_len_mean`, `entropy_loss`, `clip_fraction`,
`approx_kl`, `value_loss`, custom `mean_x_pos`, `flag_capture_rate`, `fps`.

### Phase 3 — Ablations & Iteration
**Hardware**: 3080 Ti

One-variable-at-a-time: reward shaping, action space (SIMPLE/RIGHT_ONLY/COMPLEX),
ent_coef, num_envs, n_steps, VecNormalize. Target ≥90% success.

### Phase 4 — PPO from Scratch *(optional)*
**Hardware**: CPU → GPU

Standalone PPO in PyTorch (CleanRL as ref). Rollout buffer, GAE, clipped surrogate,
entropy bonus, minibatch updates. Benchmark vs SB3. Goal = understanding.

### Phase 5 — Polish & Resume Packaging
**Hardware**: CPU

README with results/GIFs, training curves, reproducibility, failure analysis.
`notebooks/analysis.ipynb`. A stranger can clone → install → train → evaluate.

### Future Phases (post-completion)
- **A**: Multi-level generalization / curriculum (3080 Ti or cloud)
- **B**: Speedrunning optimization (cloud GPU: A100/H100)
- **C**: Full game (cloud GPU, multi-day)

---

## 5. Hardware Map

| Phase       | Hardware   | Where          | Time Estimate        |
|-------------|------------|----------------|----------------------|
| 1 Foundation| CPU        | Any dev machine| Dev days             |
| 2 Baseline  | 3080 Ti    | Personal PC    | 1–4 hrs / 5M steps  |
| 3 Ablations | 3080 Ti    | Personal PC    | ~1 day total         |
| 4 PPO Scratch| CPU → GPU | Dev → PC       | Variable             |
| 5 Polish    | CPU        | Any dev machine| Writing time         |
| Future A–C  | A100/H100  | Cloud          | Hours to days        |

3080 Ti: ~800–1500 FPS w/ 8 envs, ~2–3 GB VRAM.
Cloud trigger: 50M+ steps or parallel experiment sweeps.

---

## 6. Environment Quick Reference

**1-1 layout**: Start x≈40 → Goomba x≈200 → Pipe x≈368 → Pipe x≈618 →
Tall pipe x≈734 → Gap x≈1110 → Staircase x≈2840 → **Flag x≈3160**

**Obs** (preprocessed): `(4, 84, 84)` — 4 grayscale frames
**Actions** (SIMPLE_MOVEMENT, 7): NOOP, Right, Right+Jump, Right+Sprint,
Sprint+Jump, Jump, Left

**Info dict**: `x_pos`, `y_pos`, `time`, `life`, `flag_get`, `score`,
`coins`, `stage`, `world`, `status`
