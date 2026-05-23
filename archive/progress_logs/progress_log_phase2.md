# Project Mario — Progress Log

> **Purpose**: Volatile, session-to-session tracking. Current state, blockers,
> decisions, experiment results, and next actions. Update every session.
> For stable project reference → see `gameplan.md`.

---

## Current State

**Phase**: 2 — Baseline PPO Training (1-1) — COMPLETE (not solved)
**Branch**: `v2-dev`
**Hardware**: CPU (dev VM) + 3080 Ti (personal PC)
**Last session**: April 17, 2026
**Previous phase**: Phase 1 complete — archived to `progress_log_archive/progress_log_phase1.md`

---

## Phase 1 Checklist — COMPLETE

- [x] Set up clean project structure (see gameplan.md §3)
- [x] Create `pyproject.toml` with modern dependencies
- [x] Build `src/envs/mario_env.py` — env factory, Gymnasium-compatible
- [x] Implement wrapper stack: JoypadSpace → shimmy → SkipFrame → Resize → Grayscale → VecTransposeImage → VecFrameStack
- [x] Shimmy compat shim — shimmy 2.0.1 works (no fallback needed)
- [x] Deterministic seeding (env, torch, numpy)
- [x] `configs/default.yaml` + `src/config.py` (dataclass loader)
- [x] Sanity check: `python -m src.train --dry-run` → obs shape `(8, 4, 84, 84)` ✓
- [x] Record random-agent video → `results/videos/` ✓
- [x] Move old v1 code to `legacy/`

---

## Phase 2 Checklist — COMPLETE (not solved)

- [x] Implement PPO training loop in `src/train.py` (SB3, CnnPolicy)
- [x] EvalCallback with best model saving to `results/models/`
- [x] TensorBoard logging to `results/logs/`
- [x] Custom callbacks for x_pos tracking, flag capture rate (`src/callbacks.py`)
- [x] ProgressBarCallback with live metrics and ETA
- [x] Periodic checkpointing every ~500K steps
- [x] `--resume` CLI for interrupted training
- [x] Video recording: 3x upscale (768×720) with stats overlay
- [x] `evaluate.py` updated: model loading, flag_capture_rate summary
- [x] Baseline run: 5M timesteps, 8 envs, default config — **completed**
- [x] Monitor progress against benchmarks — **x_pos tracked, see results below**
- [ ] ~~Achieve ≥80% flag capture over 50 eval episodes~~ — **NOT ACHIEVED (peak 6%)**

### Phase 2 Baseline Run Results

| Metric | Value |
|--------|-------|
| Total timesteps | 5M (3.5M initial + 1.5M resumed) |
| Total training time | ~12 hours (8h initial + 4h resumed) |
| Hardware | 3080 Ti, DummyVecEnv, 8 envs |
| FPS | ~185 step/s |
| Peak mean_x_pos | ~2086 (at ~4.75M) |
| Peak max_x_pos | 3161 (flag reached, but rare) |
| Peak flag_capture_rate | 6% (rolling window) |
| Final flag_capture_rate | 0% (entropy collapsed) |
| Peak eval reward | 2134.4 |
| Final eval reward | 260.6 (policy degraded post-5M) |
| Entropy (start) | -1.93 (healthy, 7 actions) |
| Entropy (end) | -0.000 (fully collapsed) |
| Episodes hitting time limit | 51.2% of eval episodes |

### Diagnosis

1. **Entropy collapsed to zero** — `ent_coef=0.01` insufficient for 5M steps.
   Policy became fully deterministic; all 5 eval episodes produce identical results.
   V1 bug #5 noted `ent_coef=0.005` was too low; `0.01` is better but still
   not enough for this training length.

2. **Agent runs out of time, not out of level** — Episode lengths consistently
   hit 2005 (NES time limit). Mean x_pos ~1800–1900 at peak, but Mario is too
   slow/cautious. `forward_scale=0.1` may not incentivize speed strongly enough.

3. **Policy degraded after ~5.07M** — Eval reward collapsed from ~1993 to ~260
   (x_pos 296, dying in 40 steps). Entropy-collapsed policy is brittle;
   a few bad updates pushed it off a cliff. Best model was saved before collapse.

---

## V1 Bugs to Avoid in V2

1. `CustomReward.last_time` never initialized → crash. **Don't port this wrapper.**
2. `VecNormalize(clip_reward=1.0)` squashes +100 flag bonus to ±1. **Skip VecNormalize initially.**
3. Only 2 envs → use 8.
4. 1M steps too few → start at 5M.
5. `ent_coef=0.005` + `clip=0.1` → premature convergence. Use `0.01` + `0.2`.
6. No seeding → add deterministic seeds.
7. Legacy runner uses `channels_order='last'`, training uses `'first'` → mismatch.
8. Model version filenames scattered with no tracking.

---

## Decisions Log

| Date       | Decision                                  | Rationale                                 |
|------------|-------------------------------------------|-------------------------------------------|
| 2026-04-13 | SB3 as primary, not CleanRL               | Polish + reliability for resume           |
| 2026-04-13 | SIMPLE_MOVEMENT (7 actions)               | Standard for Mario RL                     |
| 2026-04-13 | No VecNormalize on rewards initially      | V1 reward clipping killed signal          |
| 2026-04-13 | 8 parallel envs default                   | Balance speed/stability on 3080 Ti        |
| 2026-04-13 | Phase 4 (PPO scratch) optional            | Learning exercise, not critical path      |
| 2026-04-13 | Old code → legacy/, not deleted            | Reference for what was tried              |
| 2026-04-13 | Split gameplan.md / progress_log.md       | Separate stable ref from volatile state   |
| 2026-04-14 | Python 3.12, not 3.13                     | nes-py + numpy failed on 3.13            |
| 2026-04-14 | SimpleRewardShaping as gymnasium.Wrapper  | Needs info dict + state; RewardWrapper too limited |
| 2026-04-14 | ResizeObs before GrayscaleObs             | cv2.resize drops trailing channel dim=1   |
| 2026-04-14 | VecTransposeImage IS needed               | (H,W,C)→(C,H,W) before VecFrameStack     |
| 2026-04-14 | Two factory paths (single + vec)          | Separate debug/eval/video from training   |
| 2026-04-14 | DummyVecEnv default, SubprocVecEnv opt-in | Safer debugging; both work on Windows     |
| 2026-04-14 | No CUDA extra in pyproject.toml           | PyTorch CUDA needs custom index URL       |
| 2026-04-14 | moviepy added as dependency               | Required by RecordVideo                   |
| 2026-04-15 | Callback freqs adjusted for n_envs        | SB3 callbacks fire per env.step(), not per timestep |
| 2026-04-15 | Best model = mean reward (proxy)          | Flag capture rate judged separately via evaluate.py |
| 2026-04-15 | DummyVecEnv for baseline                  | Safer on Windows; revisit if FPS disappoints |
| 2026-04-15 | tqdm progress bar for training            | UX: live x_pos, flag%, ETA during long runs |
| 2026-04-17 | --resume CLI for interrupted training     | SB3 PPO.load + reset_num_timesteps=False |
| 2026-04-17 | Video: 3x upscale + stats overlay         | Raw NES 240×256 → 768×720, cv2 text overlay |
| 2026-04-17 | Hard stop callback at total_timesteps     | SB3 overshoots due to rollout completion |

---

## Blockers

*Phase 2 complete. Moving to Phase 3 — see new progress_log.md.*

---

## Experiment Results

### Experiment: Phase 2 Baseline
- **Date**: 2026-04-16 to 2026-04-17
- **Config**: default.yaml (ent_coef=0.01, forward_scale=0.1, clip=0.2, 8 envs, SIMPLE_MOVEMENT)
- **Timesteps**: 5M (3.5M + 1.5M resumed)
- **Result**: mean_x_pos=1824 (final), peak=2086. flag_rate=0% (final), peak=6%. Peak eval reward=2134.4.
- **Verdict**: Not solved. Entropy collapsed. Agent too slow (time-limited, not death-limited). Forward progress good but insufficient speed to reach flag.

---

## Next Actions

1. Archive this log to `progress_log_archive/progress_log_phase2.md`
2. Begin Phase 3: Ablations & Iteration
3. Priority ablations based on Phase 2 diagnosis:
   - Increase `ent_coef` (0.01 → 0.02–0.03) to prevent entropy collapse
   - Increase `forward_scale` (0.1 → 0.5–1.0) to incentivize speed
   - Consider reduced action space (RIGHT_ONLY)
   - Consider longer training (10M) with healthy entropy

---

## Session Notes

### Session — April 13, 2026
- Reviewed old v1 codebase, identified 8 bugs/issues
- Created gameplan.md (stable) and progress_log.md (this file)
- Agreed on 5-phase plan with optional Phase 4 (PPO from scratch)
- Reviewed external milestone suggestions — adopted structure,
  adjusted CleanRL role and marked generalization as future phase
- Repo is on `v2-dev` branch, pushed to origin

### Session — April 15, 2026
- Developed and reviewed Phase 2 implementation plan (6 steps)
- External reviewer provided suggestions (n_envs freq adjustment, best-model proxy, etc.)
- Implemented Steps 1–5: MarioMetricsCallback, PPO training loop, VecMonitor,
  CheckpointCallback (500K), EvalCallback (1-env, 10K freq), evaluate.py with
  model loading + flag_capture_rate summary, --model CLI arg
- All steps verified with short CPU smoke tests
- Created Phase2_Step6_TODO.md for GPU training on personal PC

### Session — April 16, 2026
- Training started on 3080 Ti (personal PC)
- At 2M steps: mean_x_pos ~1294, max_x_pos 3158, flag% 3%, reward ~1923
- Progress tracking healthy at this point

### Session — April 17, 2026
- Training paused at ~3.5M (Ctrl+C overnight), added --resume CLI
- Resumed training from checkpoint, completed 3.5M → 5M (~4 hours)
- Total training: ~12 hours for 5M steps at ~185 step/s
- Added ProgressBarCallback fix for resumed training offset
- Added hard-stop callback to prevent overshoot past total_timesteps
- Improved video recording: 3x upscale (768×720) with live stats overlay
- Updated .gitignore to allow results/ to be pushed
- Pulled and analyzed all training results
- **Phase 2 NOT SOLVED**: peak flag_capture_rate 6%, entropy collapsed to 0
- Agent learns forward progress well (peak x_pos 2086) but too slow for flag
- Archived Phase 2, beginning Phase 3 planning
