# Phase 3 — Ablation & Hyperparameter Tuning Journal

> **Project**: Super Mario Bros 1-1 PPO Agent
> **Phase**: 3 — Ablations & Iteration
> **Goal**: ≥80% flag capture over 50 stochastic eval episodes
> **Hardware**: RTX 3080 Ti (training) / 5900X CPU (dev)
> **Started**: April 18, 2026

---

## Phase 2 Baseline (reference point)

Before Phase 3 began, the Phase 2 baseline established the starting point.

| Setting | Value |
|---|---|
| ent_coef | 0.01 (static) |
| forward_scale | 0.1 |
| death_penalty | -15.0 |
| flag_bonus | 50.0 |
| time_penalty | N/A (not implemented) |
| num_envs | 8 (DummyVecEnv) |
| n_steps | 512 |
| batch_size | 256 |
| eval_freq | 10,000 |
| n_eval_episodes | 5 |
| total_timesteps | 5,000,000 |

| Metric | Value |
|---|---|
| Eval x_pos | 1,824 |
| Peak mean_x_pos | 2,086 |
| Peak max_x_pos | 3,161 |
| Peak flag_rate | 6% |
| Final flag_rate | 0% |
| Final entropy | 0.00 (collapsed) |
| FPS | ~185 |
| Training time | ~12 hours |

**Root causes identified**:
1. Entropy collapsed to zero — policy became deterministic and brittle
2. Agent too slow — 51% of eval episodes hit the NES time limit
3. Post-5M policy degradation from entropy collapse making policy brittle

**Key insight**: The agent *can* reach the flag (max_x_pos=3161). The policy knows the route but doesn't execute fast enough or consistently enough.

---

## Infrastructure Changes (before ablations)

Before running ablations, several infrastructure improvements were made to support faster iteration:

| Change | Before | After | Rationale |
|---|---|---|---|
| `--name` CLI flag | N/A | Routes outputs to `results/<name>/` | Isolate ablation results from baseline |
| `--subproc` CLI flag | DummyVecEnv only | SubprocVecEnv opt-in | Parallelize CPU env stepping |
| `cfg.env.movement` wiring | Hardcoded SIMPLE_MOVEMENT | Config-driven (SIMPLE/RIGHT_ONLY) | Enable action space ablations |
| num_envs | 8 | 16 | Better CPU utilization on 5900X (24 threads) |
| n_steps | 512 | 1,024 | Larger rollout buffer, fewer collection/update transitions |
| batch_size | 256 | 512 | Fewer gradient updates per rollout (buffer is now 16K transitions) |
| eval_freq | 10,000 | 100,000 | Reduced eval overhead (~30-60 min saved per run) |
| n_eval_episodes | 5 | 10 | Better signal than 5, less overhead than 20 |

**Combined impact**: Training time dropped from ~12 hours to ~3-5 hours per 5M steps.

---

## Ablation A — Static `ent_coef=0.02`

**Date**: April 18, 2026
**Config**: `configs/experiments/ablation_a.yaml`
**Hypothesis**: Doubling ent_coef from 0.01 to 0.02 will prevent entropy collapse and maintain exploration through 5M steps.

| Changed | Baseline | Ablation A |
|---|---|---|
| ent_coef | 0.01 | **0.02** |

### Results

| Metric | Baseline | Ablation A |
|---|---|---|
| Eval x_pos | 1,824 | **2,354** (+29%) |
| Peak flag_rate | 6% | 6% |
| Final flag_rate | 0% | 1% |
| Final entropy | 0.00 | **-0.23** (better) |
| FPS | ~185 | **~365** (SubprocVecEnv) |

### Entropy Behavior
Entropy did **not** collapse permanently but **oscillated wildly**:
- Collapsed briefly to ~0 at 3.0M steps
- Recovered to ~0.3–0.99 from 3.1–3.6M
- Collapsed again near 0 at 4.4–4.5M
- Recovered to ~0.2–0.5 by end

### Verdict: **Marginal improvement** — entropy still unstable.

The eval result (x_pos=2354, identical across all 20+ episodes) confirmed deterministic eval with collapsed entropy gives identical runs. The policy improved but still can't sustain exploration.

### Lessons
- `ent_coef=0.02` prevents *permanent* collapse but not oscillation
- Higher ent_coef alone insufficient — the clipped PPO objective overpowers static entropy

---

## Ablation B — Static `ent_coef=0.03`

**Date**: April 19, 2026
**Config**: `configs/experiments/ablation_b.yaml`
**Hypothesis**: Stronger exploration pressure (3x baseline) will stabilize entropy.

| Changed | Baseline | Ablation B |
|---|---|---|
| ent_coef | 0.01 | **0.03** |

### Results

| Metric | Baseline | Ablation A | Ablation B |
|---|---|---|---|
| Eval x_pos | 1,824 | 2,354 | ~low |
| Peak flag_rate | 6% | 6% | **28%** |
| Final flag_rate | 0% | 1% | 0% |
| Final entropy | 0.00 | -0.23 | -0.27 |
| FPS | ~185 | ~365 | **~716** |

### Key Finding
**28% peak flag rate** — the policy briefly learned to finish the level consistently around 2.5–3M steps. But entropy followed the same oscillation pattern as A: healthy → collapse → recover → collapse. By 5M, mean_x_pos crashed to 450 (barely moves) and final flag_rate was 0%.

### Verdict: **Best peak so far, but same instability.**

### Lessons
- Higher static ent_coef delays collapse and enables higher peak performance
- But static ent_coef cannot prevent eventual oscillation — confirmed the reviewer's warning
- **Decision**: Both A and B collapse late → proceed to entropy schedule contingency

---

## Ablation Ent Schedule — Linear decay `ent_coef: 0.05 → 0.02`

**Date**: April 19, 2026
**Config**: `configs/experiments/ablation_ent_schedule.yaml`
**Hypothesis**: A linear entropy schedule (high early → lower late) gives strong exploration when the policy is learning, with a floor to prevent collapse.

| Changed | Ablation B | Ent Schedule |
|---|---|---|
| ent_coef | 0.03 (static) | **0.05 → 0.02 (linear)** |

**Implementation**: SB3 doesn't support callable `ent_coef`, so an `EntropyScheduleCallback` was created that mutates `model.ent_coef` directly at each step, linearly interpolating from start to end.

### Results

| Metric | Ablation B | Ent Schedule |
|---|---|---|
| Peak flag_rate | 28% | **39%** |
| Final flag_rate | 0% | 0% |
| Final entropy | -0.27 | -0.21 |
| Sustained 20%+ flag_rate | No (brief peak) | **Yes (~3.3M–4M steps)** |

### Entropy Behavior
- **Stable from 0–3.5M steps** — no collapses. The schedule kept entropy high (~1.0) during the critical learning phase
- Degraded from ~0.7 at 4M to ~0.1 at 4.9M — the 0.02 floor wasn't high enough
- But no wild oscillations like A/B — much smoother trajectory

### Verdict: **Best entropy approach so far. Proved that scheduling works.**

### Lessons
- Entropy schedule eliminated the early oscillation/collapse that plagued static ent_coef
- The 0.02 floor is too low — needs to be 0.03+ (every run that reached 0.02 eventually collapsed)
- Best model was saved around 3.3–3.5M steps, well before training ended
- **Decision**: Entropy is manageable now → move to speed ablations

---

## Ablation C — `forward_scale=0.2` + Entropy Schedule

**Date**: April 20, 2026
**Config**: `configs/experiments/ablation_c.yaml`
**Hypothesis**: Slightly more forward reward (2x baseline) incentivizes faster play without drowning the death penalty.

| Changed | Ent Schedule | Ablation C |
|---|---|---|
| forward_scale | 0.1 | **0.2** |
| ent_coef schedule | 0.05 → 0.02 | 0.05 → 0.02 |

### Results

| Metric | Ent Schedule | Ablation C |
|---|---|---|
| Eval x_pos | 2,472 | 1,942 |
| Eval steps | 248 (death) | 254 (death) |
| Peak flag_rate | 39% | 27% |
| Final flag_rate | 0% | 0% |
| Final entropy | -0.21 | **-0.43** (healthier) |

### Verdict: **Worse than base ent schedule.** Degraded late (peaked at 4.3M then crashed, same pattern as before).

### Lessons
- `forward_scale=0.2` didn't meaningfully improve speed
- The eval dying at step 254 means the agent reaches ~x=1942 then dies — not a timeout issue
- **Decision**: Try the next step up (0.3)

---

## Ablation D — `forward_scale=0.3` + Entropy Schedule

**Date**: April 20, 2026
**Config**: `configs/experiments/ablation_d.yaml`
**Hypothesis**: Stronger speed incentive (3x baseline) pushes agent past timeout failures.

| Changed | Ent Schedule | Ablation D |
|---|---|---|
| forward_scale | 0.1 | **0.3** |
| ent_coef schedule | 0.05 → 0.02 | 0.05 → 0.02 |

### Results

| Metric | Ent Schedule | Ablation C | Ablation D |
|---|---|---|---|
| Eval x_pos | 2,472 | 1,942 | 2,130 |
| Eval steps | 248 | 254 | **2,005 (timeout)** |
| Final mean_x_pos | 1,373 | 1,178 | **2,003** |
| Peak flag_rate | 39% | 27% | 22% |
| Final flag_rate | 0% | 0% | **22%** (still climbing) |
| Final entropy | -0.21 | -0.43 | **-0.61** (healthiest) |
| Policy degraded? | Yes | Yes | **No — still climbing** |

### Key Finding: Fundamentally Different Behavior
Ablation D was the **first run where the policy did NOT degrade**:
- mean_x_pos was **still climbing** at 5M: 1556 → 1837 → 2003
- flag_rate was **still climbing**: 4% → 11% → 22%
- Entropy held at -0.61 — healthiest final entropy of any run
- Eval hit timeout (2005 steps = NES timer expired) — agent is alive but slow

### Verdict: **Most promising run. Upward trend suggests more training could push to target.**

### Lessons
- `forward_scale=0.3` found the sweet spot — enough urgency without destabilizing
- The agent survives but needs more speed or more time
- The fact that the trend was still climbing strongly motivated extended training
- **Decision**: Skip RIGHT_ONLY ablation, go to 10M extended training

---

## Ablation F — 10M Steps (resumed from Ablation D)

**Date**: April 20, 2026
**Config**: `configs/experiments/ablation_f.yaml`
**Hypothesis**: Ablation D was still improving at 5M — 10M steps should push past 80% flag rate.

| Changed | Ablation D | Ablation F |
|---|---|---|
| total_timesteps | 5,000,000 | **10,000,000** |

**Method**: Resumed from Ablation D's final model at 5M steps using `--resume`.

### Results

| Metric | Ablation D (5M) | Ablation F (5M→10M) |
|---|---|---|
| Final mean_x_pos | 2,003 | **314** (catastrophic) |
| Peak flag_rate | 22% | 26% (brief) |
| Final flag_rate | 22% | **0%** |
| Final entropy | -0.61 | **0.00** (collapsed) |
| Final ep_len | ~274 | **30** (instant death) |

### What Went Wrong
**Entropy collapsed completely at ~8.3M steps.** The policy went from functional (2003 x_pos, 22% flags) to catastrophically dead (314 x_pos, 30 steps = instant death).

**Root cause — entropy schedule bug on resume**: The `EntropyScheduleCallback` used `model._total_timesteps` which SB3 inflates on resume (existing + requested steps). The schedule thought it was doing a full 0.05 → 0.02 sweep, but actually operated in a narrow 0.04 → 0.03 band — effectively static. And static entropy is what we proved doesn't work.

The ent_coef barely moved: 0.04 → 0.03 over 5M additional steps. This was insufficient, and entropy collapsed to 0 around 8.3M.

### Verdict: **Failed due to entropy schedule bug.** The bug has since been fixed.

### Lessons
- **Critical bug discovered**: `EntropyScheduleCallback` must use an explicit `total_timesteps` from config, not `model._total_timesteps`
- Bug fix implemented: callback now takes `total_timesteps` as constructor arg
- Policy that took hours to build can be destroyed in minutes by entropy collapse
- **Decision**: Need reward shaping overhaul (time penalty + larger flag bonus) alongside fixed entropy

---

## Ablation G — Reward Shaping Overhaul

**Date**: April 20, 2026  
**Config**: `configs/experiments/ablation_g.yaml`

**Hypothesis**: The agent needs stronger incentives to (a) finish fast and (b) actually care about the flag. Current reward structure gives ~700 reward from forward movement alone (at forward_scale=0.3) — the +50 flag bonus is only 7% of that. Adding a time penalty creates urgency.

| Changed | Ablation D (base) | Ablation G |
|---|---|---|
| flag_bonus | 50 | **200** |
| time_penalty | 0 (N/A) | **-0.1** (new feature) |
| ent_coef_final | 0.02 | **0.03** (higher floor) |
| forward_scale | 0.3 | 0.3 |
| ent_coef start | 0.05 | 0.05 |
| total_timesteps | 5,000,000 | 5,000,000 |

**What's new in the code**:
- `SimpleRewardShaping` now accepts `time_penalty` — a per-step negative reward that makes the agent prefer faster routes
- `EntropyScheduleCallback` fixed to work correctly on resume
- `ent_coef_final` raised to 0.03 (every run with 0.02 floor eventually collapsed)

### Expected Impact
- **time_penalty=-0.1**: Over 2005 steps (full timeout), this adds -200.5 total penalty. The agent must offset this by moving forward or finishing. Staying still = losing reward.
- **flag_bonus=200**: Now 22% of a full-level forward reward (~900 at forward_scale=0.3). Captures the flag should produce a noticeable value function gradient.
- **ent_coef floor 0.03**: Keeps exploration alive longer based on empirical evidence.

### Results

| Metric | Ablation D | Ablation G |
|---|---|---|
| Eval x_pos | 2,130 | 1,820 |
| Peak mean_x_pos | 2,003 | **2,179** |
| Final mean_x_pos | 2,003 | **331** (collapsed) |
| Peak flag_rate | 22% | **36%** |
| Final flag_rate | 22% | **0%** |
| Peak ep_rew_mean | — | **2,777** |
| Final entropy | -0.61 | **-0.001** (collapsed) |
| Final ep_len | ~274 | **27** (instant death) |
| ent_coef schedule | 0.05→0.02 | 0.05→0.03 (worked correctly) |
| FPS | ~700 | **~623** |

### Flag Rate Timeline
- **0–3.5M**: Negligible (0–3%), agent still learning the level
- **3.5M–4.0M**: Ramp-up phase, flag_rate climbing to 6–12%
- **4.0M–4.3M**: First sustained window ≥10% (peaked 24% at 4.26M)
- **4.3M–4.4M**: Brief dip to 0–4% (entropy wobble)
- **4.4M–4.9M**: **Golden window** — sustained 10–36% flag rate, peak **36% at 4.82M**
- **4.9M–5.0M**: Cliff collapse — entropy crashed from ~0.5 to -0.001, flag_rate dropped to 0%

### Entropy Behavior
- Schedule operated correctly: ent_coef 0.0499 → 0.0300 over 5M steps
- Entropy was healthy (~0.7–1.0) from 0–4M steps
- Began eroding at ~4.0M — first dip below 0.1 at 4.03M
- **Cliff-edge collapse at ~4.9M**: entropy went from ~0.5 to -0.001 in ~100K steps
- Unlike Ablation D's smooth final entropy (-0.61), G's entropy hit a catastrophic cliff

### Verdict: **Highest peak ever (36%), but entropy collapse destroyed the policy.**

Reward shaping clearly worked — the agent learned faster, reached higher flag rates, and the time penalty + flag bonus reshaped incentives correctly. But the 0.03 entropy floor was **still not enough** to prevent late-stage collapse. The collapse was more sudden and catastrophic than previous runs.

### Lessons
- Reward shaping is effective: peak flag_rate improved from 22% (D) to 36% (G)
- time_penalty creates urgency — agent learned to move faster in the golden window
- flag_bonus=200 gave the flag meaningful weight in the reward structure
- **ent_coef floor of 0.03 is still insufficient** — even with correct scheduling, entropy collapsed
- The collapse is more cliff-like than gradual — once entropy drops below some threshold, it's irrecoverable
- Ablation D's stability (no collapse) vs G's higher peak but collapse suggests the reward shaping amplifies policy gradient signals, which accelerates both learning AND entropy erosion
- **Key insight**: The problem isn't the floor value — it's that PPO's clipped objective eventually overpowers *any* static entropy floor in this environment

---

## Ablation H — `target_kl=0.015` (KL Divergence Cap)

**Date**: April 20, 2026
**Config**: `configs/experiments/ablation_h.yaml`

**Hypothesis**: Ablation G's cliff-edge entropy collapse at 4.9M was caused by a cascade of oversized policy updates — once one update pushes the policy too far toward determinism, the next rollout produces lower-entropy data, which yields even larger gradients, accelerating the collapse. SB3's `target_kl` parameter truncates the epoch loop within each PPO update when KL divergence exceeds `1.5 * target_kl`, breaking this positive feedback loop at the source. This is preventive (stops bad updates from happening) rather than reactive (detecting collapse after the fact).

| Changed | Ablation G | Ablation H |
|---|---|---|
| target_kl | None | **0.015** |

**What's new in the code**:
- `target_kl` added to `TrainingConfig` dataclass and wired to PPO constructor (+ resume path)
- `EntropyCollapseDetector` callback added — runs in diagnostic/logging-only mode (no stopping). Intercepts `logger.dump()` to capture `train/entropy_loss` and `train/approx_kl`, computes rolling entropy velocity, KL clip fraction, and logs all diagnostics under `collapse/` prefix in TensorBoard.

### Results

| Metric | Ablation G | Ablation H |
|---|---|---|
| Eval x_pos | 1,820 | **315** |
| Peak mean_x_pos | 2,179 | **806** |
| Final mean_x_pos | 331 | **621** |
| Peak flag_rate | 36% | **1%** (single episode at 540K) |
| Final flag_rate | 0% | **0%** |
| Peak ep_rew_mean | 2,777 | **943** |
| Final entropy | -0.001 (collapsed) | **-0.59** (healthy) |
| Entropy collapsed? | Yes (cliff at 4.9M) | **No** |
| Policy degraded? | Yes (catastrophic) | **No (but never learned)** |

### Eval Trajectory
The evaluations.npz data revealed **zero learning progression** across the entire 5M steps:
- Eval reward oscillated between ~316 and ~341 from start to finish (std_dev=0.0 within each eval — all 10 episodes identical)
- Occasional timeout episodes (ep_len=2005, reward=-603) scattered throughout
- The agent died in ~27–40 steps for the entire run — never progressed past the first few obstacles
- **The final model produced identical results to the 100K model**

### KL Divergence Analysis — The Smoking Gun

Comparing KL distributions between G (uncapped) and H (target_kl=0.015):

| Statistic | Ablation G (no cap) | Ablation H (target_kl=0.015) |
|---|---|---|
| KL min | 0.0000 | -0.0000 |
| KL 25th | 0.0233 | 0.0118 |
| KL median | **0.0306** | 0.0166 |
| KL 75th | 0.0428 | 0.0306 |
| KL 90th | 0.0688 | 0.0636 |
| KL max | 1.2603 | 0.3268 |
| KL mean | 0.0495 | 0.0314 |
| Would exceed 0.015 | **96%** | 58% |
| Would exceed 0.0225 (SB3 cutoff = 1.5×target) | — | **35%** |

**Ablation G's *median* KL was 0.031 — more than double the target.** This means a normal, productive PPO update in this environment naturally requires KL ~0.03. SB3 truncates epochs when `approx_kl > 1.5 * target_kl = 0.0225`, so **96% of G's updates would have been throttled** under this cap.

The `collapse/kl_clip_frac` metric confirmed this: **50–80% of recent updates in the rolling window exceeded `target_kl` throughout the entire run**. Many updates were cut to 1–2 epochs out of 4 (or even epoch 1 alone exceeded the limit), effectively halving or quartering the learning rate.

### Why The Agent Never Learned
`target_kl=0.015` is a textbook value for standard Atari/MuJoCo PPO. But this environment with aggressive reward shaping (forward_scale=0.3, time_penalty=-0.1, flag_bonus=200) produces inherently **higher-variance returns** that require larger policy updates to make progress. The KL cap prevented the cascade that destroys policies — but it also prevented the large updates needed for the policy to transition from random behavior to purposeful movement.

The entropy staying healthy (-0.59) confirms the mechanism works as intended: no large updates → no entropy erosion → no collapse. But also no large updates → no learning.

### Verdict: **Failed as a training run. Excellent as a diagnostic.**

Ablation H proved three important things:
1. **`target_kl` does prevent entropy collapse** — the mechanism is sound
2. **0.015 is far too restrictive for this environment** — it throttled ~96% of productive updates
3. **The natural KL baseline for this environment + reward shaping is ~0.03** — any `target_kl` must be set above this to permit learning

### Lessons
- `target_kl=0.015` is standard for clean Atari but wrong for shaped-reward Mario — environment-specific calibration is essential
- **Always check the natural KL distribution before setting target_kl** — compare with an uncapped run first (Ablation G gave us this data retroactively)
- The `EntropyCollapseDetector` and `collapse/kl_clip_frac` metric proved their value immediately — they diagnosed the throttling in the logs
- Entropy was the healthiest of any run (-0.59) because the agent was effectively prevented from learning at all — healthy entropy without learning is not a success
- **Next step**: `target_kl=0.05` would let ~75% of normal updates through (above G's median of 0.031) while still catching the catastrophic cascades (G's max KL was 1.26, cliff collapses would produce spikes >0.10)

---

## Ablation I — `target_kl=0.05` (Calibrated KL Cap)

**Date**: April 21, 2026
**Config**: `configs/experiments/ablation_i.yaml`

**Hypothesis**: Ablation H proved `target_kl` prevents entropy collapse but 0.015 was far too restrictive (throttled 96% of productive updates). The environment's natural KL median is 0.031 (from Ablation G). Setting `target_kl=0.05` sits above G's 75th percentile (0.043), letting ~75% of normal updates through. SB3's cutoff at `1.5 × 0.05 = 0.075` passes ~90% of natural updates while still catching catastrophic cascades (KL >0.10).

| Changed | Ablation H | Ablation I |
|---|---|---|
| target_kl | 0.015 | **0.05** |

### Results

| Metric | Ablation G (no cap) | Ablation H (0.015) | Ablation I (0.05) |
|---|---|---|---|
| Peak mean_x_pos | 2,179 | 806 | **1,519** |
| Final mean_x_pos | 331 | 621 | **1,488** |
| Peak flag_rate | 36% | 1% | **3%** |
| Final flag_rate | 0% | 0% | **2%** |
| Peak ep_rew_mean | 2,777 | 943 | **1,870** |
| Final entropy | -0.001 (collapsed) | -0.59 (healthy) | **-0.57** (healthy) |
| Entropy collapsed? | Yes (cliff at 4.9M) | No | **No** |
| Policy degraded? | Catastrophic | No (never learned) | **No (still climbing)** |
| KL exceeding SB3 cutoff | N/A | 35% | **2.0%** |
| FPS | ~623 | — | **~912** |

### KL Divergence — Calibration Confirmed

| Statistic | Ablation G (no cap) | Ablation H (0.015) | Ablation I (0.05) |
|---|---|---|---|
| KL median | 0.031 | 0.017 | **0.026** |
| KL 75th | 0.043 | 0.031 | **0.036** |
| KL 90th | 0.069 | 0.064 | **0.046** |
| KL max | 1.260 | 0.327 | **0.127** |
| Exceeding SB3 cutoff | N/A | 35% | **2.0%** |

`target_kl=0.05` is correctly calibrated: only 2% of updates hit the SB3 cutoff (6/305), compared to 35% at 0.015. The policy makes full-sized productive updates without throttling. The max KL (0.127) is well below G's catastrophic cascade values (max 1.260), confirming target_kl would catch any runaway updates.

### mean_x_pos Acceleration — The Key Signal

| Phase | Start x_pos | End x_pos | Delta |
|---|---|---|---|
| 0–1M | 687 | 611 | -76 (exploring) |
| 1–2M | 611 | 770 | +159 |
| 2–3M | 763 | 934 | +171 |
| 3–4M | 913 | 1,045 | +132 |
| 4–5M | 1,058 | 1,519 | **+461** (accelerating) |

The learning curve was **accelerating** at the end of training. The 4–5M phase produced 3x more progress than any previous phase. This strongly motivated extending to 10M.

### Entropy Behavior
- Healthy throughout: median ranged from -0.577 to -1.136 per phase
- Final entropy: -0.571 — comparable to H (-0.59) but **with actual learning**
- No collapses, no oscillations, no cliff edges
- The entropy schedule (0.05 → 0.03) combined with target_kl=0.05 produced the first run that was both **learning AND stable** simultaneously

### Verdict: **First run combining learning with stability. Still climbing at 5M.**

Ablation I didn't reach G's peak performance (3% vs 36% flag_rate) because it trained more conservatively. But unlike G, the policy was **still improving** at 5M with **no signs of degradation**. This is the same pattern as Ablation D (still climbing at 5M) but now with the target_kl safety net for extended training.

### Lessons
- `target_kl=0.05` is correctly calibrated — permits normal learning (98% of updates unthrottled) while guarding against cascades
- **Learning + stability simultaneously achieved** — prior runs had one or the other, never both
- The 4–5M acceleration (+461 mean_x_pos) suggests the policy is in a rapid improvement phase — extending training should yield significant gains
- Slower learning than G (3% vs 36% peak flag_rate at 5M) is expected — target_kl limits maximum update size, trading peak speed for stability
- **Decision**: Extend to 10M to capitalize on the accelerating learning curve

---

## Ablation J — 10M Steps (resumed from Ablation I)

**Date**: April 22, 2026
**Config**: `configs/experiments/ablation_j.yaml`

**Hypothesis**: Ablation I was accelerating at 5M (mean_x_pos +461 in 4–5M phase, flag_rate climbing to 3%). This is the same extension strategy as Ablation F (which extended D), but with three protections F didn't have: (1) target_kl=0.05 prevents cascade collapse, (2) entropy schedule bug fixed, (3) EntropyCollapseDetector provides diagnostics.

| Changed | Ablation I | Ablation J |
|---|---|---|
| total_timesteps | 5,000,000 | **10,000,000** |

**Method**: Resumed from Ablation I's final model at 5M steps using `--resume`.

### Results

| Metric | Ablation I (5M) | Ablation J (5M→10M) |
|---|---|---|
| Peak mean_x_pos | 1,519 | **2,480** |
| Final mean_x_pos | 1,488 | **1,032** (degraded) |
| Peak flag_rate | 3% | **42%** (highest ever) |
| Final flag_rate | 2% | **1%** |
| Peak ep_rew_mean | 1,870 | **3,167** (highest ever) |
| Final entropy | -0.571 | **-1.006** (eroded) |
| Entropy collapsed? | No | **No** (gradual erosion, not cliff) |
| Policy degraded? | No | **Yes** (gradual, not catastrophic) |
| KL exceeding SB3 cutoff | 2.0% | **2.3%** |
| FPS | ~912 | **~746** |

### Flag Rate Timeline
- **5–6M**: Warming up, 2% avg, 7% peak — resuming I's trajectory
- **6–7M**: Steady improvement, 4% avg, 11% peak
- **7–8M**: Breakout phase, 9% avg, **26%** peak — agent learning to finish consistently
- **8–9M**: **Golden window** — 14% avg, **42% peak** at 8.9M ← highest flag_rate of any run
- **9–10M**: Degradation — 4% avg, entropy eroding, policy losing reliability

### mean_x_pos Timeline
- **5–6M**: avg=1488, peak=1734 — steady from I
- **6–7M**: avg=1650, peak=1916 — climbing
- **7–8M**: avg=1855, peak=2326 — strong progress
- **8–9M**: avg=1855, peak=**2463** — peak performance, but *average* stopped climbing
- **9–10M**: avg=1140, peak=2480 — average crashed, but peak still high (inconsistent policy)

### Entropy Behavior — Gradual Erosion (Not Cliff Collapse)
- **5–6M**: median=-0.578, healthy — carrying forward from I
- **6–7M**: median=-0.390, rising — actually more exploratory (ent_coef schedule hitting floor)
- **7–8M**: median=-0.359, stable — productive exploration
- **8–9M**: median=-0.296, eroding — entropy drifting toward 0
- **9–10M**: median=-0.508, decaying — accelerating erosion, final entropy **-1.006**

**Critical difference from G**: Ablation G's entropy went from 0.5 to -0.001 in ~100K steps (cliff collapse). J's entropy eroded from -0.578 to -1.006 over 5M steps — a **gradual decline, not a catastrophic cliff**. `target_kl=0.05` prevented the cascade mechanism that destroyed G, but it could not prevent the slow, steady drift toward determinism.

### KL Divergence — target_kl Working as Intended

| Statistic | Ablation I (5M) | Ablation J (5M→10M) |
|---|---|---|
| KL median | 0.026 | **0.033** (slightly higher — more confident policy) |
| KL 75th | 0.036 | **0.042** |
| KL max | 0.127 | **0.100** |
| Exceeding SB3 cutoff | 2.0% | **2.3%** |

KL distribution barely changed between I and J — target_kl kept updates in bounds. Only 2.3% of updates hit the SB3 cutoff at 0.075. The max KL (0.100) was well below G's catastrophic 1.260. **target_kl is doing its job** — the degradation is not from oversized updates but from cumulative small shifts toward determinism.

### Stochastic Evaluation (Post-Training)

During analysis, a **critical bug was discovered and fixed** in `src/evaluate.py`: the evaluation function was hardcoded with `deterministic=True`, which made all multi-episode evaluations produce **identical trajectories**. Every prior eval "flag_rate" from evaluate.py was binary (same episode repeated N times). A `--stochastic` CLI flag was added to fix this.

Results with stochastic evaluation (50 episodes):

| Model | Flag Rate | mean_x_pos | max_x_pos | mean_reward |
|---|---|---|---|---|
| best_model (saved by eval callback) | **12% (6/50)** | 1,704 | 3,161 | 2,119 |
| 7.5M checkpoint | 0% (0/50) | — | — | — |

### Video Recording (10 stochastic episodes)

The best_model was recorded for 10 stochastic episodes — **1 out of 10 captured the flag** (episode 10). The agent navigates the full level, reaching x=3161, demonstrating it knows the complete route. Failures appear to be consistency and timing issues at specific obstacles, not capability limits.

### Verdict: **Highest peak ever (42% training, 12% stochastic eval). target_kl prevents cliff collapse but not gradual erosion.**

Ablation J is the most successful run by both peak training metrics (42% flag_rate, 3167 reward) and verified stochastic evaluation (12% over 50 episodes). The policy learned to capture the flag at a meaningful rate and the video confirms it knows the full route. However, the policy degraded after ~8.5M steps due to gradual entropy erosion — a different and strictly better failure mode than the catastrophic cliffs of G/F.

### Lessons
- `target_kl=0.05` successfully prevents cliff collapse in extended training — G died in 100K steps, J eroded over 5M steps
- **42% peak flag_rate from training metrics** — highest of any ablation, confirming the full stack (reward shaping + entropy schedule + target_kl) works
- **Gradual erosion is a strictly better failure mode** — the best_model checkpoint captures peak performance, and the degradation is slow enough that training could be stopped earlier
- The ent_coef floor (0.03) is insufficient for 10M+ training — entropy eroded below the floor in the 8–10M range
- **Stochastic eval is essential** — deterministic eval was giving meaningless identical trajectories. The true flag rate (12%) gives an honest picture of policy consistency
- **Video recording reveals qualitative insights** — the agent knows the route but fails at specific obstacles, suggesting the policy needs more consistent execution, not more exploration
- **Decision**: Need to either (a) raise entropy floor for long runs, (b) use EntropyCollapseDetector in stop mode to freeze at peak, or (c) try flat ent_coef + target_kl for very long training

---

## Critical Bug Fix: `evaluate.py` Deterministic Hardcode

**Date**: April 22, 2026

During Ablation J analysis, a critical bug was discovered in `src/evaluate.py`: the `evaluate()` function was hardcoded with `deterministic=True` in the `model.predict()` call. This meant:

- **All multi-episode evaluations produced identical trajectories** — the same actions in the same order every time
- Every prior evaluate.py "flag_rate" was binary: if the single deterministic trajectory reached the flag, 100%; if not, 0%
- Training-time eval (SB3's EvalCallback with `n_eval_episodes=10`) was also affected — all 10 episodes per checkpoint were identical

**Fix**: Added a `--stochastic` CLI flag to `src/evaluate.py` and `src/config.py`. When set, `model.predict()` uses `deterministic=False`, producing varied trajectories that reveal the policy's true consistency.

**Impact on prior results**: All eval "flag_rate" numbers in this journal from evaluations.npz and evaluate.py runs are unreliable as measures of consistency. Training metrics (`mario/flag_capture_rate` from the rolling episode buffer across 16 envs) remain valid because they use stochastic rollouts.

**Files changed**: `src/evaluate.py` (added `stochastic` parameter, `--stochastic` CLI flag), `src/config.py` (added `stochastic` field to EvalConfig).

---

## Comparison Summary

| Run | ent_coef | fwd_scale | flag_bonus | time_pen | target_kl | Peak Flag% | Final Flag% | Entropy Stable? | Degraded? |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | 0.01 static | 0.1 | 50 | — | — | 6% | 0% | No (collapsed) | Yes |
| **Ablation A** | 0.02 static | 0.1 | 50 | — | — | 6% | 1% | No (oscillated) | Yes |
| **Ablation B** | 0.03 static | 0.1 | 50 | — | — | **28%** | 0% | No (oscillated) | Yes |
| **Ent Schedule** | 0.05→0.02 | 0.1 | 50 | — | — | **39%** | 0% | Mostly (late decay) | Yes |
| **Ablation C** | 0.05→0.02 | 0.2 | 50 | — | — | 27% | 0% | Better | Yes |
| **Ablation D** | 0.05→0.02 | **0.3** | 50 | — | — | 22% | **22%** | **Yes** | **No** |
| **Ablation F** | 0.04→0.03* | 0.3 | 50 | — | — | 26% | 0% | No (bug) | Catastrophic |
| **Ablation G** | 0.05→0.03 | 0.3 | **200** | **-0.1** | — | **36%** | 0% | No (cliff at 4.9M) | Yes (catastrophic) |
| **Ablation H** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.015** | 1% | 0% | **Yes** | No (never learned) |
| **Ablation I** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.05** | 3% | **2%** | **Yes** | **No** (still climbing) |
| **Ablation J** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.05** | **42%** | 1% | Mostly (gradual erosion) | Yes (gradual) |

\* Ablation F's schedule was effectively static due to the resume bug.
\*\* Ablation J stochastic eval: **12% flag rate (6/50)** on best_model. 1/10 video episodes captured the flag.

---

## Key Learnings

### On Entropy
1. **Static ent_coef always collapses eventually** — PPO's clipped objective overpowers it in long runs
2. **Entropy schedule is essential** — linear decay from 0.05 → floor works, but even 0.03 floor isn't enough
3. **Entropy oscillation predicts policy degradation** — when entropy starts bouncing near 0, the policy will crash within ~500K steps
4. **Best model checkpoints are critical** — the best model is often saved 1–2M steps before entropy collapse
5. **Entropy collapse is cliff-like, not gradual** — Ablation G went from 0.5 to -0.001 in ~100K steps. Once it starts, it's irrecoverable.
6. **No static floor has prevented collapse** — 0.01, 0.02, 0.03 all fail. The problem requires a fundamentally different approach (flat entropy, early stopping, or adaptive control).

### On Speed & Reward Shaping
7. **forward_scale=0.3 is the sweet spot** — 0.2 didn't help, 0.3 produced the only non-degrading run
8. **Higher forward_scale doesn't guarantee faster play** — it just rewards movement. Without time penalty, the agent has no cost for dawdling.
9. **Flag bonus at +50 is too small** — relative to accumulated forward reward (~700), the flag is only 7% of total reward. Agent has weak incentive to actually finish.
10. **Reward shaping works but amplifies entropy erosion** — Ablation G's stronger rewards produced faster learning (36% peak vs 22%) but also faster entropy collapse. Stronger gradients = faster convergence = faster determinism.

### On KL Divergence & target_kl
11. **target_kl prevents entropy collapse** — Ablation H had the healthiest entropy of any run (-0.59) because oversized updates were blocked
12. **target_kl must be calibrated to the environment's natural KL** — 0.015 is textbook for clean Atari but throttled 96% of productive updates in shaped-reward Mario (natural median KL ≈ 0.031)
13. **Always measure natural KL from an uncapped run before setting target_kl** — Ablation G's data retroactively provided this baseline
14. **Healthy entropy without learning is not success** — preventing collapse is necessary but not sufficient; the agent must still be able to make meaningful policy updates

### On Training Infrastructure
15. **SubprocVecEnv is essential** — 2x FPS improvement on 5900X (185 → 365+)
16. **Reducing eval overhead matters** — eval_freq 10K → 100K + fewer episodes saved 30–60 min per run
17. **Resume works but has pitfalls** — SB3 inflates `_total_timesteps` on resume, which broke the entropy schedule. Always use explicit config values, not model internals.
18. **EntropyCollapseDetector proved its value** — `collapse/kl_clip_frac` immediately diagnosed the throttling problem in Ablation H

### On Methodology
19. **One variable at a time is correct but slow** — when possible, combine proven improvements
20. **Upward trend at end of training > high peak that degrades** — Ablation D's 22% final flag_rate was more valuable than Ent Schedule's 39% peak that crashed to 0%
21. **The agent can reach the flag** — every run hit max_x_pos=3161. The problem is consistency and speed, not capability.
22. **A "failed" run can be the most informative** — Ablation H produced zero learning but gave us the exact KL baseline needed to calibrate target_kl correctly

### On target_kl Calibration (Ablations I & J)
23. **target_kl=0.05 is the correct calibration for this environment** — Only 2% of updates hit SB3 cutoff (vs 35% at 0.015), letting normal learning proceed while catching cascades
24. **target_kl prevents cliff collapse, not gradual erosion** — J's entropy eroded from -0.578 to -1.006 over 5M steps, but never hit the catastrophic cliff that destroyed G in 100K steps. These are two distinct failure modes.
25. **Gradual entropy erosion is a strictly better failure mode** — The best_model checkpoint captures peak performance, and degradation is slow enough to detect and stop. Cliff collapse destroys the policy before any checkpoint can save it.
26. **The ent_coef floor (0.03) is insufficient for 10M+ training** — Entropy eroded below the schedule floor in J's 8–10M range. Longer runs need a higher floor (0.04+) or flat ent_coef.
27. **Learning + stability can coexist** — Ablation I was the first run simultaneously learning AND stable. Prior runs had one or the other. The combination of entropy schedule + target_kl is the key.

### On Evaluation
28. **Deterministic eval masks true performance** — A critical bug in evaluate.py hardcoded `deterministic=True`, making all multi-episode evals produce identical trajectories. All prior eval "flag_rate" numbers from evaluate.py/evaluations.npz were binary (same trajectory repeated).
29. **Stochastic eval reveals the real policy** — With stochastic eval, J's best_model achieves 12% flag rate (6/50). The agent CAN reach the flag but is inconsistent. This is the only honest performance number.
30. **Video recording is invaluable for diagnosis** — Watching 10 stochastic episodes (1 flag capture) reveals failure modes that metrics can't: where the agent hesitates, which obstacles it fails at, timing issues.
31. **Training metrics remain valid** — `mario/flag_capture_rate` from the rolling episode buffer uses stochastic rollouts across 16 envs, so training-time flag_rate numbers in this journal are trustworthy. Only evaluate.py and evaluations.npz were affected by the bug.

---

## What's Next

Ablation J achieved **42% peak flag_rate** (training) and **12% stochastic eval** (6/50 best_model) — the highest verified result so far. The agent captures the flag in 1/10 video recordings. `target_kl=0.05` prevented the catastrophic cliff collapse that destroyed Ablation G, but gradual entropy erosion still degraded the policy over 10M steps.

**Gap to target**: 12% → 80% stochastic eval (68 percentage points remaining).

**Key observations from video + eval**:
- The agent knows the complete route (max_x_pos=3161 in every run)
- It captures the flag occasionally (12% of the time) — this is a consistency problem, not a capability problem
- Failures appear to be timing/execution issues at specific obstacles, not directional confusion

**Options for next ablation**:

1. **Higher entropy floor** — Raise ent_coef_final from 0.03 to 0.04 or 0.05 to combat gradual erosion in extended training. The 0.03 floor erodes over 10M steps.
2. **EntropyCollapseDetector in stop=True mode** — Freeze training when entropy erosion is detected, preserving peak policy. J's best_model was saved well before training ended.
3. **RIGHT_ONLY action space** — Reduce from 7 to 5 actions, lower entropy ceiling but potentially faster convergence and less erosion.
4. **Re-evaluate earlier ablation best models stochastically** — G, D, Ent Schedule best_models may have higher true flag rates than originally reported (all prior evals were deterministic/identical).
5. **Flat ent_coef + target_kl for very long runs** — Skip the schedule entirely, use static ent_coef=0.03 or 0.04 + target_kl=0.05 for 20M+ steps.

**Solved = ≥80% flag capture over 50 stochastic eval episodes.**

Progress tiers:
- <20%: Not meaningfully improved ← **Ablation J best_model is here (12%)**
- 20–50%: Significant progress, keep iterating
- 50–80%: Strong result, one more knob turn likely solves
- ≥80%: **Solved**
