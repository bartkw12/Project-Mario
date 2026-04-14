# Project Mario — Progress Log

> **Purpose**: Volatile, session-to-session tracking. Current state, blockers,
> decisions, experiment results, and next actions. Update every session.
> For stable project reference → see `gameplan.md`.

---

## Current State

**Phase**: 2 — Baseline PPO Training (1-1)
**Branch**: `v2-dev`
**Hardware**: CPU (dev machine) → 3080 Ti for training
**Last session**: April 14, 2026
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

## Phase 2 Checklist

- [ ] Implement PPO training loop in `src/train.py` (SB3, CnnPolicy)
- [ ] EvalCallback with best model saving to `results/models/`
- [ ] TensorBoard logging to `results/logs/`
- [ ] Custom callbacks for x_pos tracking, flag capture rate (`src/callbacks.py`)
- [ ] Baseline run: 5M timesteps, 8 envs, default config
- [ ] Monitor progress against benchmarks (see gameplan.md §4)
- [ ] Achieve ≥80% flag capture over 50 eval episodes (Phase 2 solved criterion)

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

---

## Blockers

*None currently.*

---

## Experiment Results

*No experiments run yet. Will be populated starting Phase 2.*

<!-- Template for experiment entries:
### Experiment: [name]
- **Date**: YYYY-MM-DD
- **Config change**: [what changed vs baseline]
- **Timesteps**: [how many]
- **Result**: mean_x_pos=?, flag_rate=?, ep_rew_mean=?
- **Verdict**: [keep / discard / investigate]
-->

---

## Next Actions

1. Plan Phase 2: PPO training implementation
2. Implement training loop in `src/train.py` using SB3 PPO + CnnPolicy
3. Build custom callbacks for Mario-specific metrics
4. Run baseline 5M step training on 3080 Ti
5. Reopen VS Code at `Project-Mario/` folder (fix workspace root)

---

## Session Notes

### Session — April 13, 2026
- Reviewed old v1 codebase, identified 8 bugs/issues
- Created gameplan.md (stable) and progress_log.md (this file)
- Agreed on 5-phase plan with optional Phase 4 (PPO from scratch)
- Reviewed external milestone suggestions — adopted structure,
  adjusted CleanRL role and marked generalization as future phase
- Repo is on `v2-dev` branch, pushed to origin
