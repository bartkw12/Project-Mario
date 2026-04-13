# Project Mario — Progress Log

> **Purpose**: Volatile, session-to-session tracking. Current state, blockers,
> decisions, experiment results, and next actions. Update every session.
> For stable project reference → see `gameplan.md`.

---

## Current State

**Phase**: 1 — Foundation & Environment Stack
**Branch**: `v2-dev`
**Hardware**: CPU (dev machine)
**Last session**: April 13, 2026

---

## Phase 1 Checklist

- [ ] Set up clean project structure (see gameplan.md §3)
- [ ] Create `pyproject.toml` with modern dependencies
- [ ] Build `src/envs/mario_env.py` — env factory, Gymnasium-compatible
- [ ] Implement wrapper stack: JoypadSpace → SkipFrame → Grayscale → Resize(84) → FrameStack(4)
- [ ] Shimmy / manual compat shim for gym-super-mario-bros → Gymnasium
- [ ] Deterministic seeding (env, torch, numpy)
- [ ] `configs/default.yaml` + `src/config.py` (dataclass loader)
- [ ] Sanity check: random policy 1000 steps, verify obs shape `(n_envs, 4, 84, 84)`
- [ ] Record random-agent video → `results/videos/`
- [ ] Move old v1 code to `legacy/`

### Known Compatibility Challenge
`gym-super-mario-bros` → `nes-py` → `gym==0.21`. Need shimmy or manual adapter
to bridge to Gymnasium + SB3 2.x. This is the hardest part of Phase 1.

**Shimmy 2.0 risk**: shimmy jumped from 1.3.0 → 2.0.0 (May 2024) → 2.0.1 (Apr 2026).
Major version bump — need to verify `[gym-v21]` extra still works. Fallback: shimmy 1.3.0
or a hand-rolled adapter.

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

1. Start Phase 1: restructure repo, install modern deps, build env factory
2. Test shimmy compatibility with gym-super-mario-bros on this machine
3. Get random agent running and recording video

---

## Session Notes

### Session — April 13, 2026
- Reviewed old v1 codebase, identified 8 bugs/issues
- Created gameplan.md (stable) and progress_log.md (this file)
- Agreed on 5-phase plan with optional Phase 4 (PPO from scratch)
- Reviewed external milestone suggestions — adopted structure,
  adjusted CleanRL role and marked generalization as future phase
- Repo is on `v2-dev` branch, pushed to origin
