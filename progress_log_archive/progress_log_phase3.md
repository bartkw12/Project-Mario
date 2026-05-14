# Project Mario — Progress Log

> **Purpose**: Volatile, session-to-session tracking. Current state, blockers,
> decisions, experiment results, and next actions. Update every session.
> For stable project reference → see `gameplan.md`.

---

## Current State

**Phase**: 3 — COMPLETED (SOLVED)
**Branch**: `v2-dev`
**Hardware**: CPU (dev VM) + 3080 Ti (personal PC)
**Last session**: May 14, 2026
**Previous phases**: Phase 1, Phase 2 archived in `progress_log_archive/`
**Result**: **87.5% flag capture (175/200)** — target was ≥80%

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

### Completed
- [x] Ablations A–J: Entropy scheduling, forward_scale, reward shaping, target_kl calibration
- [x] Ablation K: D extended to 10M (failed — D-family caps at 18%)
- [x] Ablation L: Moderate reward shaping (flag_bonus=100, time_penalty=-0.05) — peaked 30%
- [x] Ablation M: Resume L 9.0M with LR halved (1.25e-4) — peaked 36%
- [x] Ablation N: Resume M 10.5M with LR halved again (6.25e-5) — peaked 64%
- [x] Ablation O: Resume N 13.5M with LR halved (3.125e-5) — peaked 71% (200-ep confirmed)
- [x] Ablation P: Resume O 14.5M with LR halved (1.5625e-5) — regressed to 68% (LR exhausted)
- [x] Ablation Q: Resume O with higher entropy floor — catastrophic collapse (entropy discontinuity)
- [x] Ablation R: Resume O with n_epochs=3 — 64% (200-ep), worse than O
- [x] Multi-seed Phase L: Seeds 1/2/3 — seed 1 peaked 54%, seed 2 42%, seed 3 dead (8%)
- [x] **Multi-seed Phase M: Seed 1 resumed from 8.0M with LR÷2 — 87.5% (175/200) SOLVED**

**Final model**: `results/multiseed_s1_M/models/checkpoints/ppo_mario_12000000_steps.zip`
**Result**: 87.5% stochastic flag capture (175/200 episodes) — **PROJECT SOLVED**

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
| 2026-04-14 | DummyVecEnv default, SubprocVecEnv opt-in | Safer debugging; both work on Windows     |
| 2026-04-17 | Phase 2 not solved → Phase 3 ablations    | Entropy collapse + slow agent identified  |
| 2026-04-19 | Entropy schedule (linear decay 0.05→floor)| Static ent_coef always collapses eventually |
| 2026-04-20 | forward_scale=0.3 as standard             | Sweet spot — 0.2 didn't help, 0.3 stable  |
| 2026-04-20 | EntropyCollapseDetector (diagnostic mode) | Detect entropy erosion early              |
| 2026-04-21 | target_kl=0.05 as standard                | Calibrated to env's natural KL (~0.031 median) |
| 2026-04-22 | Stochastic eval (50 eps) as ground truth  | Deterministic eval was giving identical trajectories |
| 2026-05-05 | Checkpoint sweep as standard workflow     | best_model often not the true best        |
| 2026-05-06 | Moderate reward shaping (flag=100, tp=-0.05)| D too gentle (18% cap), J too aggressive (fragile) |
| 2026-05-06 | Compounding LR halving at frontier        | Prevents collapse, enables continued climbing |

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

## Ablation Summary

Full details in `notebooks/ablation_journal/ablation_journal.md`.

| ID | Key Change | Best Flag% | Outcome |
|----|-----------|-----------|----------|
| A  | ent_coef=0.02 | 0% stoch | Marginal — entropy still oscillates |
| B  | ent_coef=0.03 | 22% (best_model) | Good peak but volatile |
| Ent Sched | 0.05→0.02 linear | 0% stoch | Schedule works but floor too low |
| C  | forward_scale=0.2 | 0% stoch | Worse than base schedule |
| D  | forward_scale=0.3 | 18% (5.0M) | First non-degrading run |
| F  | D extended to 10M | 8% | Failed — entropy schedule bug on resume |
| G  | flag_bonus=200, tp=-0.1 | 0% stoch | 36% training peak but collapsed |
| H  | target_kl=0.015 | 0% | Too restrictive — never learned |
| I  | target_kl=0.05 | 0% stoch | Learning + stable, still climbing |
| J  | I extended to 10M | 26% (8.0M) | Former champion, narrow window |
| K  | D + target_kl + 10M | 18% | D-family ceiling confirmed |
| L  | Moderate shaping, fresh 10M | 30% (9.0M) | Broke D ceiling |
| M  | Resume L 9.0M, LR÷2 | 36% (best_model) | LR halving works |
| N | Resume M 10.5M, LR÷2 | 64% (13.5M) | Strong — former champion |
| O | Resume N 13.5M, LR÷2 | 71% (14.5M) | 200-ep confirmed champion (seed 42) |
| P | Resume O 14.5M, LR÷2 | 68% | Regressed — LR halving exhausted |
| Q | Resume O, ent_coef_final=0.04 | 0% (collapsed) | Entropy discontinuity on resume |
| R | Resume O, n_epochs=3 | 64% (200-ep) | Wider window but lower ceiling |
| Multi-seed L | Seeds 1/2/3, fresh L recipe | 54%/42%/8% | Seed 1 exceptional |
| **Multi-seed M** | **Seed 1, resume 8.0M, LR÷2** | **87.5% (200-ep)** | **SOLVED** |

---

## Next Actions

**Phase 3 is complete. Project solved.**

Potential future work (not required):
- Record a showcase video of the winning model
- Test generalization to other levels (1-2, 1-3, etc.)
- Write up findings for documentation/blog post

---

## Session Notes

### Session — May 14, 2026 — PROJECT SOLVED
- Multi-seed Phase M (seed 1): **87.5% flag capture (175/200)**
- 200-episode evaluation confirms result well above 80% target
- Winning recipe: L config + seed 1 + one LR halving (2.5e-4 → 1.25e-4)
- Total training: 12M steps (~4 hours GPU across two phases)
- Phase 3 complete

### Session — May 10–13, 2026
- Ablation R: n_epochs=3 isolated test — 64% (200-ep), worse than O
- Multi-seed Phase L: seed 1 = 54%, seed 2 = 42%, seed 3 = 8%
- Seed 1 selected for continuation through compounding pipeline

### Session — May 8–9, 2026
- Ablation O: 71% (200-ep re-eval, 142/200) — new champion
- Ablation P: LR halving to 1.5625e-5 regressed to 68% — exhausted
- Ablation Q: Entropy floor change caused catastrophic collapse (repeat of K lesson)
- Failure analysis: x≈2470 bottleneck accounts for 57% of O's failures

### Session — May 7, 2026
- Ablation N: 64% flag rate (32/50) — new champion
- Compounding LR halving strategy: L(30%) → M(36%) → N(64%)
- Created Ablation O config (LR=3.125e-5, resume from N 13.5M)
- 16 points away from solving

### Session — May 5–6, 2026
- Checkpoint sweep infrastructure built (scripts/checkpoint_sweep.py)
- Found J 8.0M = 26% as champion via sweep (hidden behind worse best_model)
- Ablation K failed (D caps at 18%)
- Ablation L: moderate reward shaping → 30%
- Ablation M: resume L + half LR → 36%

### Session — April 17–22, 2026
- Ablations A–J completed (see ablation journal for full details)
- Key discoveries: entropy schedule, target_kl=0.05, stochastic eval bug fix
- Stochastic re-evaluation revealed ranking inversion (training peaks ≠ robustness)
