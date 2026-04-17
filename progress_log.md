# Project Mario — Progress Log

> **Purpose**: Volatile, session-to-session tracking. Current state, blockers,
> decisions, experiment results, and next actions. Update every session.
> For stable project reference → see `gameplan.md`.

---

## Current State

**Phase**: 3 — Ablations & Iteration
**Branch**: `v2-dev`
**Hardware**: CPU (dev VM) + 3080 Ti (personal PC)
**Last session**: April 17, 2026
**Previous phases**: Phase 1, Phase 2 archived in `progress_log_archive/`

---

## Phase 2 Summary (for reference)

Baseline PPO (5M steps, default config) completed but **not solved**.
- Peak mean_x_pos: 2086, peak flag_capture_rate: 6%, final: 0%
- **Root causes**: entropy collapsed to zero (`ent_coef=0.01` too low),
  agent too slow (runs out of NES time, not deaths), policy degraded post-5M.
- Best model saved. Checkpoints at every 500K steps available for analysis.
- Full details: `progress_log_archive/progress_log_phase2.md`

---

## Phase 3 Checklist

**Goal**: Achieve ≥80% flag capture over 50 eval episodes through targeted ablations.

### Priority 1 — Fix entropy collapse
- [ ] Ablation A: `ent_coef=0.02` (2x baseline), 5M steps
- [ ] Ablation B: `ent_coef=0.03` (3x baseline), 5M steps
- [ ] Monitor entropy stays above ~0.3 throughout training

### Priority 2 — Incentivize speed
- [ ] Ablation C: `forward_scale=0.5` (5x baseline) with best ent_coef from above
- [ ] Ablation D: `forward_scale=1.0` (10x baseline)
- [ ] Track: do episodes finish via flag rather than time limit?

### Priority 3 — Action space reduction
- [ ] Ablation E: `RIGHT_ONLY` (5 actions) instead of `SIMPLE_MOVEMENT` (7 actions)
- [ ] Fewer actions = less entropy drift, faster learning

### Priority 4 — Extended training (if needed)
- [ ] Ablation F: 10M steps with best config from above
- [ ] Only run if 5M with tuned params shows progress but isn't solved

### Stretch
- [ ] Ablation G: Learning rate schedule (linear decay)
- [ ] Ablation H: Sticky death penalty (penalize more when dying in same spot)

**Solved = ≥80% flag capture over 50 eval episodes (same as Phase 2 criterion).**

---

## Phase 2 Lessons Learned (informing Phase 3)

1. `ent_coef=0.01` prevents entropy collapse at 1M but fails by 5M — need higher.
2. `forward_scale=0.1` teaches rightward movement but not urgency — agent dawdles.
3. 51% of eval episodes hit the NES time limit — speed is the primary failure mode.
4. The agent *can* reach the flag (max_x_pos=3161) — the policy knows the route,
   it just doesn't execute fast enough or consistently enough.
5. DummyVecEnv at 185 FPS is slow — 12 hours for 5M steps. Consider SubprocVecEnv
   benchmark before running multiple ablations.
6. Post-training policy degradation (5M+) was caused by entropy collapse making
   the policy brittle. Hard-stop callback now prevents this.
7. Eval with 5 episodes + deterministic=True produces identical runs when entropy
   is collapsed. Consider stochastic eval or more episodes for Phase 3 monitoring.

---

## V1 Bugs to Avoid (carried forward)

1. `CustomReward.last_time` never initialized → crash. **Don't port this wrapper.**
2. `VecNormalize(clip_reward=1.0)` squashes +100 flag bonus to ±1. **Skip VecNormalize.**
3. Only 2 envs → use 8.
4. 1M steps too few → start at 5M.
5. `ent_coef=0.005` + `clip=0.1` → premature convergence.
   **Updated**: `ent_coef=0.01` also insufficient at 5M steps. Phase 3 tests higher.
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
| 2026-04-14 | DummyVecEnv default, SubprocVecEnv opt-in | Safer debugging; both work on Windows     |
| 2026-04-15 | Callback freqs adjusted for n_envs        | SB3 callbacks fire per env.step()         |
| 2026-04-15 | Best model = mean reward (proxy)          | Flag capture rate judged via evaluate.py  |
| 2026-04-17 | --resume CLI for interrupted training     | SB3 PPO.load + reset_num_timesteps=False  |
| 2026-04-17 | Hard stop callback at total_timesteps     | SB3 overshoots due to rollout completion  |
| 2026-04-17 | Phase 2 not solved → Phase 3 ablations    | Entropy collapse + slow agent identified  |

---

## Blockers

*None currently.*

---

## Experiment Results

### Experiment: Phase 2 Baseline (reference)
- **Date**: 2026-04-16 to 2026-04-17
- **Config**: default.yaml (ent_coef=0.01, forward_scale=0.1, clip=0.2, 8 envs, SIMPLE_MOVEMENT)
- **Timesteps**: 5M (3.5M + 1.5M resumed), ~12 hours total
- **Result**: mean_x_pos=1824 (final), peak=2086. flag_rate=0% (final), peak=6%. Peak eval reward=2134.4. Entropy collapsed to 0.
- **Verdict**: Not solved. Baseline reference for Phase 3 ablations.

<!-- Template for Phase 3 experiments:
### Experiment: Ablation [X] — [name]
- **Date**: YYYY-MM-DD
- **Config change**: [what changed vs baseline]
- **Timesteps**: [how many]
- **Result**: mean_x_pos=?, flag_rate=?, ep_rew_mean=?, entropy_final=?
- **Verdict**: [keep / discard / investigate]
-->

---

## Ablation Plan

One variable at a time. Each ablation compared against Phase 2 baseline.

| ID | Variable | Baseline | Test Value | Hypothesis |
|----|----------|----------|------------|------------|
| A  | ent_coef | 0.01     | 0.02       | Prevent entropy collapse, maintain exploration |
| B  | ent_coef | 0.01     | 0.03       | Stronger exploration pressure |
| C  | forward_scale | 0.1 | 0.5        | Incentivize faster rightward movement |
| D  | forward_scale | 0.1 | 1.0        | Even stronger speed incentive |
| E  | action_space | SIMPLE (7) | RIGHT_ONLY (5) | Fewer actions = easier to learn + less entropy drift |
| F  | timesteps | 5M      | 10M        | More training with healthy entropy |

**Run order**: A → B (pick best ent_coef) → C → D (pick best forward_scale) → E → F if needed.

---

## Next Actions

1. Plan Phase 3 implementation: config support for experiment overrides
2. Benchmark SubprocVecEnv vs DummyVecEnv before running multiple ablations
3. Run Ablation A (ent_coef=0.02) on 3080 Ti
4. Evaluate, compare against baseline, decide next ablation

---

## Session Notes

### Session — April 17, 2026
- Analyzed Phase 2 training results in detail (TensorBoard + evaluations.npz)
- Identified 3 root causes: entropy collapse, slow agent, post-5M degradation
- Archived Phase 2 progress log to `progress_log_archive/progress_log_phase2.md`
- Created Phase 3 progress log with ablation plan
- Priority: fix entropy first, then speed, then action space
