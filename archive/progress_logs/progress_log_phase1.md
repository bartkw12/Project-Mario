# Project Mario — Progress Log (Phase 1 Archive)

> **Archived**: April 14, 2026
> **Phase**: 1 — Foundation & Environment Stack
> **Status**: COMPLETE — both exit criteria met

---

## Phase 1 Checklist

- [x] Set up clean project structure (see gameplan.md §3)
- [x] Create `pyproject.toml` with modern dependencies
- [x] Build `src/envs/mario_env.py` — env factory, Gymnasium-compatible
- [x] Implement wrapper stack: JoypadSpace → shimmy → SkipFrame → SimpleRewardShaping → Resize → Grayscale → VecTransposeImage → VecFrameStack
- [x] Shimmy compat shim for gym-super-mario-bros → Gymnasium (shimmy 2.0.1 works)
- [x] Deterministic seeding (env, torch, numpy) via `set_global_seed()`
- [x] `configs/default.yaml` + `src/config.py` (dataclass loader)
- [x] Sanity check: `python -m src.train --dry-run` → obs shape `(8, 4, 84, 84)` ✓
- [x] Record random-agent video → `results/videos/` ✓
- [x] Move old v1 code to `legacy/`
- [x] Create `src/evaluate.py` (random agent eval with per-episode stats)
- [x] Create `src/callbacks.py` (stub for Phase 2)
- [x] Smoke tests A–F all passing

### Exit Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `python -m src.train --dry-run` prints `obs.shape = (8, 4, 84, 84)` | **PASSED** | Step 13 |
| Random-agent video saved to `results/videos/` | **PASSED** | Step 10 (smoke_d.py) |

---

## Files Created / Modified in Phase 1

```
Project-Mario/
├── pyproject.toml                  (new — modern deps, CPU torch)
├── .gitignore                      (updated — legacy paths, .venv, egg-info)
├── configs/
│   └── default.yaml                (new — training + reward + eval config)
├── src/
│   ├── __init__.py                 (new)
│   ├── config.py                   (new — dataclass config + CLI + seeding)
│   ├── train.py                    (new — --dry-run mode, Phase 2 stub)
│   ├── evaluate.py                 (new — random agent eval + video)
│   ├── callbacks.py                (new — stub)
│   └── envs/
│       ├── __init__.py             (new — exports make_single_env, make_vec_env)
│       ├── wrappers.py             (new — SkipFrame, SimpleRewardShaping)
│       └── mario_env.py            (new — env factory, two creation paths)
├── tests/
│   ├── smoke_a.py                  (bare mario + JoypadSpace)
│   ├── smoke_b.py                  (shimmy bridge)
│   ├── smoke_c.py                  (full single-env wrapper stack)
│   ├── smoke_d.py                  (RecordVideo)
│   ├── smoke_e.py                  (DummyVecEnv shape verification)
│   └── smoke_f.py                  (SubprocVecEnv Windows compat)
├── legacy/
│   ├── Super_Mario_V1/             (moved from src/)
│   ├── Testing/                    (moved from src/)
│   ├── requirements.txt            (moved from root)
│   └── requirements-cpu.txt        (moved from root)
└── results/videos/                 (random agent video)
```

---

## Key Technical Decisions Made During Phase 1

| Date       | Decision                                                | Rationale                                                     |
|------------|---------------------------------------------------------|---------------------------------------------------------------|
| 2026-04-14 | Python 3.12, not 3.13                                   | nes-py + numpy failed to build on 3.13 (no prebuilt wheels)   |
| 2026-04-14 | Visual Studio C++ Build Tools required                   | nes-py has C extension, no prebuilt wheel for 3.12 either     |
| 2026-04-14 | `SimpleRewardShaping(gymnasium.Wrapper)`, NOT `RewardWrapper` | Needs info dict + internal state; RewardWrapper too limited |
| 2026-04-14 | `ResizeObservation` before `GrayscaleObservation`        | cv2.resize drops trailing dim=1; resize while still 3-channel |
| 2026-04-14 | `VecTransposeImage` IS needed                            | Converts (H,W,C) → (C,H,W) before VecFrameStack              |
| 2026-04-14 | Two factory paths: `make_single_env` + `make_vec_env`    | Separate debug/eval/video from training vectorized stack       |
| 2026-04-14 | DummyVecEnv default, SubprocVecEnv opt-in                | Safer for debugging; both verified working on Windows          |
| 2026-04-14 | No CUDA extra in pyproject.toml                          | PyTorch CUDA needs custom index URL; documented as comment     |
| 2026-04-14 | `moviepy` added as dependency                            | Required by gymnasium.wrappers.RecordVideo                     |
| 2026-04-14 | Reward values in config marked PROVISIONAL               | Tuning is Phase 2/3; forward_scale=0.1 conservative default   |

---

## Wrapper Stack (Verified Shape at Each Stage)

| Stage | Wrapper | obs.shape |
|-------|---------|-----------|
| 1 | gym_super_mario_bros.make() | (240, 256, 3) |
| 2 | JoypadSpace(SIMPLE_MOVEMENT) | (240, 256, 3), Discrete(7) |
| 3 | shimmy GymV21CompatibilityV0 | (240, 256, 3), 5-tuple API |
| 4 | SkipFrame(skip=4) | (240, 256, 3) |
| 5 | SimpleRewardShaping | (240, 256, 3) |
| 6 | ResizeObservation(84, 84) | (84, 84, 3) |
| 7 | GrayscaleObservation(keep_dim=True) | (84, 84, 1) |
| 8 | VecTransposeImage | (1, 84, 84) |
| 9 | VecFrameStack(n_stack=4, channels_order='first') | **(4, 84, 84)** |

Vectorized with 8 envs: **(8, 4, 84, 84)** ✓

---

## V1 Bugs — Status in V2

| # | V1 Bug | V2 Status |
|---|--------|-----------|
| 1 | `CustomReward.last_time` never initialized | Fixed — `SimpleRewardShaping` initializes `_last_x_pos` in `reset()` |
| 2 | `VecNormalize(clip_reward=1.0)` kills signal | Fixed — no VecNormalize used |
| 3 | Only 2 envs | Fixed — default 8 envs |
| 4 | 1M steps too few | Fixed — default 5M in config |
| 5 | `ent_coef=0.005` + `clip=0.1` → premature convergence | Fixed — `0.01` + `0.2` in config |
| 6 | No seeding | Fixed — `set_global_seed()` + per-env seeds |
| 7 | `channels_order` mismatch | Fixed — VecTransposeImage + channels_order='first' verified |
| 8 | Model filenames scattered | Deferred — Phase 2 will use structured results/ paths |

---

## Session Notes

### Session — April 13, 2026
- Reviewed old v1 codebase, identified 8 bugs/issues
- Created gameplan.md (stable) and progress_log.md (this file)
- Agreed on 5-phase plan with optional Phase 4 (PPO from scratch)
- Reviewed external milestone suggestions — adopted structure,
  adjusted CleanRL role and marked generalization as future phase
- Repo is on `v2-dev` branch, pushed to origin

### Session — April 13-14, 2026 (Phase 1 Implementation)
- Formulated 14-step implementation plan across 6 sub-phases (A–F)
- External review incorporated 7 suggestions (accepted 6 fully, 1 partially)
- Restructured repo: V1 code → legacy/, new directory skeleton
- Created pyproject.toml with pinned modern deps
- Installed deps: Python 3.13 failed (nes-py C ext), fell back to 3.12
- Installed Visual Studio C++ Build Tools for nes-py compilation
- Smoke tests A–F all passed:
  - A: bare mario env works on 3.12
  - B: shimmy 2.0.1 bridge works perfectly (no fallback needed)
  - C: full wrapper stack verified, discovered ResizeObs drops channel dim
  - D: RecordVideo works (needed render_mode fix + moviepy install)
  - E: DummyVecEnv shape (8, 4, 84, 84) confirmed (needed wrapper reorder + VecTransposeImage)
  - F: SubprocVecEnv works on Windows (no pickling issues)
- train.py --dry-run passes — Phase 1 exit criterion #1
- Random agent video recorded — Phase 1 exit criterion #2
- evaluate.py + callbacks.py stubs created
