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

## Stochastic Re-evaluation of All Best Models

**Date**: April 22, 2026

With the deterministic eval bug fixed, every ablation's `best_model` checkpoint was re-evaluated with 50 stochastic episodes. This is the single most important analysis in this journal — it retroactively changes the model rankings and invalidates the assumption that training peak flag_rate predicts deployment performance.

### Raw Results

| Run | Stoch Flag% | mean_x_pos | max_x_pos | mean_reward | mean_steps |
|---|---|---|---|---|---|
| **Ablation D** | **16% (8/50)** | 1,911 | 3,161 | 2,383 | 281 |
| **Ablation B** | **14% (7/50)** | 2,100 | 3,161 | 2,219 | 219 |
| **Ablation J** | **12% (6/50)** | 1,704 | 3,161 | 2,119 | — |
| **Ablation F** | **8% (4/50)** | 1,849 | 3,161 | 2,306 | 236 |
| Baseline | 0% (0/50) | 2,524 | 2,850 | 2,554 | 837 |
| Ablation A | 0% (0/50) | 1,671 | 2,679 | 1,717 | 324 |
| Ent Schedule | 0% (0/50) | 1,608 | 3,038 | 1,660 | 260 |
| Ablation C | 0% (0/50) | 1,435 | 2,009 | 1,626 | 199 |
| Ablation G | 0% (0/50) | 1,353 | 2,026 | 1,651 | 171 |
| Ablation I | 0% (0/50) | 1,193 | 2,022 | 1,444 | 169 |

### The Ranking Inversion — Training Peaks vs Stochastic Reality

| Run | Training Peak Flag% | Stochastic Eval Flag% | Rank Change |
|---|---|---|---|
| Ablation G | **36%** (2nd highest) | **0%** | Was top-tier → bottom |
| Ent Schedule | **39%** (highest) | **0%** | Was #1 → bottom |
| Ablation D | 22% (4th) | **16%** (best) | Was middle → **#1** |
| Ablation B | 28% (3rd) | **14%** (2nd) | Was middle → **#2** |
| Ablation J | **42%** (highest ever) | **12%** (3rd) | Was #1 → 3rd |

The correlation between training peak flag_rate and stochastic eval is **inverted** for the top models. The runs with the highest training peaks (G: 36%, Ent Schedule: 39%) produced best_model checkpoints that score 0% stochastically. The run with moderate training metrics but stable entropy (D: 22%) produced the most robust checkpoint.

### Why Deterministic Eval Overstated Performance

SB3's `EvalCallback` selects the best model based on deterministic evaluation reward. With `deterministic=True`, `model.predict()` always selects the argmax action — producing a single fixed trajectory. This means:

1. **The best_model is the checkpoint where the single best deterministic path scored highest** — not where the policy was most consistent
2. **A narrow, overfit policy can score perfectly on one trajectory** while failing on all others. G's best_model deterministic path reached x=1,820 but stochastically maxed at 2,026 — the policy was highly peaked around one trajectory
3. **The "golden window" checkpoints were saved during entropy erosion** — when the policy was becoming deterministic. This made the deterministic eval look great (the policy was converging to one good path) while the stochastic robustness was collapsing

### Why D and B Won

**Ablation D** (16%, 8/50):
- Entropy was healthy at save time (-0.61) — the healthiest final entropy of any 5M run at that point
- The policy **never degraded** — it was still climbing at 5M, so the best_model was saved at a point of genuine learning, not peak-before-collapse
- No reward shaping amplification — `flag_bonus=50`, no `time_penalty`. The gentler reward signal produced slower but more robust learning
- 8/50 episodes captured the flag, mean_x_pos=1,911 — the policy is competitive across diverse trajectories

**Ablation B** (14%, 7/50):
- Simple static `ent_coef=0.03` — no schedule, no target_kl
- Despite entropy oscillation during training, the best_model was saved during a "good oscillation" window where the policy was both capable and exploratory
- Highest mean_x_pos of any model (2,100) and fastest (mean_steps=219) — this policy plays aggressively and efficiently
- 7/50 episodes captured the flag — nearly as consistent as D

### Why G and Ent Schedule Failed Stochastically

**Ablation G** (0%, 0/50):
- max_x_pos=2,026 — the agent never even reached the flag area (3,161). Under stochastic sampling, it consistently dies around x=1,100–2,000
- The best_model was saved during the "golden window" (4.4–4.9M) when entropy was already eroding. The deterministic path was good, but the policy had narrowed to the point where any deviation from the optimal trajectory results in death
- mean_steps=171 — the agent dies quickly, suggesting it learned fast aggressive play but with no fallback strategies

**Ent Schedule** (0%, 0/50):
- max_x_pos=3,038 — one episode got close but still didn't finish. The policy has some range but not enough to consistently reach the end
- Pattern of x_pos clustering around ~690, ~1,675, ~2,470 suggests the policy has "walls" — specific obstacles it can pass deterministically but fails at stochastically
- The high training peak (39%) was during a sustained window (3.3–4M), but the best_model checkpoint was apparently saved during the late-stage entropy decay, not at peak robustness

### The Baseline Anomaly

The Phase 2 Baseline scored 0% but had the highest mean_x_pos (2,524) and uniquely reached x=2,850 in 14/50 episodes — farther than any other model's stochastic max except those that captured the flag. It never captured the flag because mean_steps=837 — it plays so slowly that it consistently hits the NES timer. This model *knows the level* better than most but moves too slowly, confirming the original Phase 2 diagnosis.

### Impact on the Ablation Narrative

This re-evaluation fundamentally changes the story:

1. **"G had the best reward shaping"** → G's reward shaping produced fast training metrics but a brittle checkpoint. D's gentler approach produced a more robust policy.

2. **"Ent Schedule had the best entropy strategy"** → Ent Schedule had good training-time entropy behavior, but its best_model was saved when entropy was decaying. The best stochastic model (D) had a simpler setup.

3. **"J is the culmination of all improvements"** → J ranks 3rd stochastically (12%) behind D (16%) and B (14%). The target_kl + reward shaping + entropy schedule stack produced the highest *training peaks* (42%) but not the most robust checkpoint.

4. **"Training peak flag_rate = progress toward 80% goal"** → Training peaks are unreliable predictors. Stochastic eval is the only honest measure. The project is closer to the goal than the training peaks suggested for D/B, and much farther than they suggested for G/Ent Schedule.

### Implications for Future Work

1. **Switch EvalCallback to stochastic evaluation** — The SB3 EvalCallback should use `deterministic=False` so the best_model checkpoint captures stochastic robustness, not single-trajectory performance. This is the highest-priority code change.

2. **D-family is the best foundation for extension** — D's configuration (forward_scale=0.3, ent_coef 0.05→0.02, no reward shaping overhaul) is the most promising base to extend with target_kl protection. The reward shaping from G/J may be counterproductive.

3. **Entropy health at checkpoint time is the key predictor** — Models with healthy entropy when saved (D: -0.61, B: oscillating but in good phase) are stochastically robust. Models with eroding entropy when saved (G: during golden window collapse, Ent Schedule: during late decay) are stochastically brittle.

4. **All future ablation results must include stochastic eval** — Training metrics and deterministic eval are insufficient. Every run should be evaluated with `--stochastic --episodes 50` on its best_model as the ground truth.

---

## Comparison Summary

| Run | ent_coef | fwd_scale | flag_bonus | time_pen | target_kl | Peak Flag% | Final Flag% | Stoch Eval | Entropy Stable? | Degraded? |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | 0.01 static | 0.1 | 50 | — | — | 6% | 0% | **0%** (0/50) | No (collapsed) | Yes |
| **Ablation A** | 0.02 static | 0.1 | 50 | — | — | 6% | 1% | **0%** (0/50) | No (oscillated) | Yes |
| **Ablation B** | 0.03 static | 0.1 | 50 | — | — | **28%** | 0% | **14%** (7/50) | No (oscillated) | Yes |
| **Ent Schedule** | 0.05→0.02 | 0.1 | 50 | — | — | **39%** | 0% | **0%** (0/50) | Mostly (late decay) | Yes |
| **Ablation C** | 0.05→0.02 | 0.2 | 50 | — | — | 27% | 0% | **0%** (0/50) | Better | Yes |
| **Ablation D** | 0.05→0.02 | **0.3** | 50 | — | — | 22% | **22%** | **16%** (8/50) ★ | **Yes** | **No** |
| **Ablation F** | 0.04→0.03* | 0.3 | 50 | — | — | 26% | 0% | **8%** (4/50) | No (bug) | Catastrophic |
| **Ablation G** | 0.05→0.03 | 0.3 | **200** | **-0.1** | — | **36%** | 0% | **0%** (0/50) | No (cliff at 4.9M) | Yes (catastrophic) |
| **Ablation H** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.015** | 1% | 0% | — | **Yes** | No (never learned) |
| **Ablation I** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.05** | 3% | **2%** | **0%** (0/50) | **Yes** | **No** (still climbing) |
| **Ablation J** | 0.05→0.03 | 0.3 | 200 | -0.1 | **0.05** | **42%** | 1% | **12%** (6/50) | Mostly (gradual erosion) | Yes (gradual) |

\* Ablation F's schedule was effectively static due to the resume bug.
★ Ablation D is the **current best verified model** at 16% stochastic flag capture (8/50).

**Key takeaway**: Training Peak Flag% and Stochastic Eval are poorly correlated. G (36% peak → 0% stochastic) and Ent Schedule (39% peak → 0% stochastic) demonstrate that high training peaks can produce brittle checkpoints. D (22% peak → 16% stochastic) demonstrates that stable training with healthy entropy produces robust checkpoints.

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

### On Stochastic Re-evaluation (The Ranking Inversion)
32. **Training peak flag_rate is a poor predictor of stochastic robustness** — G (36% peak → 0% stochastic) and Ent Schedule (39% peak → 0% stochastic) had the highest training peaks but produced checkpoints that score 0/50 stochastically. D (22% peak → 16% stochastic) and B (28% peak → 14% stochastic) had moderate peaks but the best stochastic scores. The correlation is inverted.
33. **Deterministic EvalCallback selects for narrow policies, not robust ones** — The best_model checkpoint captures the moment when a single deterministic trajectory scores highest. A policy that is converging toward one good trajectory (i.e., entropy is eroding) will produce a high deterministic score while its stochastic diversity collapses. This is exactly what happened to G and Ent Schedule.
34. **Entropy health at checkpoint save time is the key predictor of stochastic robustness** — D (entropy -0.61 at save) → 16% stochastic. G (entropy eroding during "golden window") → 0% stochastic. Healthy entropy means the policy has fallback strategies when stochastic noise deviates from the optimal path.
35. **Reward shaping amplified training metrics but degraded checkpoint quality** — D (no time_penalty, flag_bonus=50): 16% stochastic. G (time_penalty=-0.1, flag_bonus=200): 0% stochastic. The stronger reward signal pushed G to learn faster and reach higher training peaks, but the amplified gradients accelerated entropy erosion, producing a brittle policy at checkpoint time.
36. **Ablation D is the current best verified model** — 16% stochastic flag capture (8/50), mean_x_pos=1,911, max_x_pos=3,161. Configuration: forward_scale=0.3, ent_coef 0.05→0.02, no reward shaping overhaul, no target_kl. Its defining trait: the policy never degraded and entropy stayed healthy (-0.61).
37. **Future checkpoint selection must use stochastic evaluation** — The EvalCallback should be switched to `deterministic=False` so it selects best_model based on stochastic robustness. All future ablation results must include `--stochastic --episodes 50` as ground truth.
38. **The Baseline is "far-reaching but slow"** — mean_x_pos=2,524 (highest of any model), reached x=2,850 in 14/50 episodes, but 0% flag captures and mean_steps=837. It knows the level better than most models but times out consistently, confirming the original Phase 2 timeout diagnosis.

---

## Checkpoint Sweep — Finding the True Best Models

**Date**: May 5, 2026

The stochastic re-evaluation of `best_model` checkpoints (April 22) revealed the ranking inversion, but only tested one checkpoint per ablation. A full sweep of all periodic checkpoints was run to find the true best model across the entire training history.

**Tool**: `scripts/checkpoint_sweep.py` — 50 stochastic episodes per checkpoint.

### Ablation D — Full Sweep (12 models)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **5.0M** | **18% (9/50)** | 1,983 | 3,161 | 2,481 | 262 |
| 2 | final_model | 18% (9/50) | 1,883 | 3,161 | 2,354 | 253 |
| 3 | best_model | 14% (7/50) | 1,893 | 3,161 | 2,352 | 310 |
| 4 | 4.5M | 14% (7/50) | 1,856 | 3,161 | 2,316 | 247 |
| 5 | 4.0M | 2% (1/50) | 1,439 | 3,161 | 1,776 | 201 |
| 6–12 | ≤3.5M | 0% | ≤1,252 | — | — | — |

**Key finding**: D's 5.0M checkpoint (last periodic save) beats `best_model`. Confirms D was still improving at training end. Monotonic progression — no collapses, no oscillation.

### Ablation J — Golden Window Sweep (5 models)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **8.0M** | **26% (13/50)** | 2,275 | 3,161 | 2,881 | 253 |
| 2 | 8.5M | 6% (3/50) | 1,759 | 3,161 | 2,183 | 204 |
| 3 | best_model | 2% (1/50) | 1,619 | 3,161 | 2,002 | 169 |
| 4 | 9.5M | 0% (0/50) | 1,715 | 1,953 | 2,116 | 190 |
| 5 | 9.0M | 0% (0/50) | 721 | 898 | 723 | 525 |

**Key finding**: J's 8.0M is the **new project champion** at 26%. The degradation is steeper than the journal suggested: 26% → 6% → 0% in just 1M steps. The `target_kl` prevented instant death (like G) but erosion still destroys the policy in ~1M steps once it begins.

### Ablation B — Full Sweep (11 models)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **best_model** | **22% (11/50)** | 2,096 | 3,161 | 2,219 | 227 |
| 2 | 3.0M | 14% (7/50) | 1,964 | 3,161 | 2,038 | 381 |
| 3 | 2.5M | 4% (2/50) | 1,754 | 3,161 | 1,832 | 216 |
| 4 | 2.0M | 4% (2/50) | 1,362 | 3,161 | 1,407 | 184 |
| 5–11 | others | 0% | — | — | — | — |

**Key finding**: B's `best_model` (saved at ~2.5–3M) is genuinely its peak — but entropy collapse at 3.5M (mean_x=303, instant death) shows how volatile static ent_coef is. B's success is luck-of-timing, not stability.

### Updated Global Rankings

| Rank | Model | Flag% | Family | Stable? |
|---|---|---|---|---|
| 1 | **J 8.0M** | **26% (13/50)** | J (reward shaping + target_kl) | Narrow window |
| 2 | **B best_model** | **22% (11/50)** | B (static ent_coef=0.03) | No — volatile |
| 3 | **D 5.0M** | **18% (9/50)** | D (forward_scale=0.3 + ent schedule) | **Yes** |
| 4 | D final_model | 18% (9/50) | D | Yes |
| 5 | B 3.0M | 14% (7/50) | B | No |
| 6 | D best_model / D 4.5M | 14% (7/50) | D | Yes |

### Insights

39. **J 8.0M is the project's best model** — 26% stochastic flag capture, hidden behind a `best_model` that scored only 2%. The deterministic EvalCallback saved the wrong checkpoint entirely.
40. **J's "gradual erosion" is actually a ~500K–1M step cliff in stochastic terms** — 26% at 8.0M → 6% at 8.5M → 0% at 9.0M. `target_kl` slows the cliff but doesn't prevent eventual death.
41. **D is the only family with monotonic improvement and zero collapses** — every checkpoint is better than the last. This makes D the safest base for extension.
42. **B's success is timing-dependent luck** — best_model was saved during a brief good window. The 3.5M checkpoint (mean_x=303) shows catastrophic collapse within 500K steps of peak. Static ent_coef is fundamentally unreliable.
43. **Checkpoint sweeps are essential infrastructure** — The "true best" model was never the one labeled `best_model` for D or J. Periodic checkpoints with post-hoc stochastic evaluation is the correct workflow.

---

## Ablation K — D Extended to 10M with target_kl + Higher Entropy Floor

**Date**: May 5, 2026
**Config**: `configs/experiments/ablation_k.yaml`

**Hypothesis**: D was still climbing at 5.0M (18%). Extending to 10M with `ent_coef_final=0.03` (higher floor) + `target_kl=0.05` (cascade protection) should sustain learning and push past D's ceiling.

| Changed | Ablation D | Ablation K |
|---|---|---|
| total_timesteps | 5,000,000 | **10,000,000** |
| ent_coef_final | 0.02 | **0.03** |
| target_kl | None | **0.05** |

**Method**: Resumed from D's 5.0M checkpoint. Entropy schedule jumped ent_coef from D's floor (0.02) to K's schedule value (~0.04) on resume.

### Results — Checkpoint Sweep (50 stochastic episodes each)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **9.0M** | **18% (9/50)** | 1,742 | 3,161 | 2,179 | 211 |
| 2 | 5.5M | 16% (8/50) | 1,954 | 3,161 | 2,433 | 312 |
| 3 | 6.0M / 7.5M | 8% (4/50) | — | 3,161 | — | — |
| 4 | 10.0M / final / 8.5M | 6% (3/50) | — | 3,161 | — | — |
| — | 7.0M / 8.0M / 9.5M | 0% (0/50) | — | — | — | — |

### Training Observations

- **CollapseDetector fired twice** (~5.9M and ~7.1M): 3–4 warnings each, self-resolved. `target_kl` prevented cascades.
- **Peak training flag_rate**: 26% at 8.9M (matches J's training peak)
- **Final training flag_rate**: 17–18% (policy alive at 10M, not collapsed)
- **Severe oscillation**: Unlike D's monotonic climb, K oscillates wildly (0–26% across checkpoints)
- **FPS**: ~1.2–1.4K steps/s (2 hours total for 5M new steps)

### Verdict: **Failed to beat D. D-family has a ceiling at ~18%.**

K's best checkpoint (9.0M at 18%) merely ties D 5.0M. Five million additional steps with stability protections produced zero improvement in stochastic robustness.

### Why It Failed

1. **ent_coef discontinuity on resume**: Jumping from 0.02→0.04 degraded the working policy immediately. K's 5.5M (16%) is worse than D's 5.0M (18%) it resumed from. The agent spent ~3.5M steps recovering to where D already was.

2. **D's reward structure has an inherent ceiling**: `flag_bonus=50` is only ~7% of accumulated forward reward. The agent has insufficient incentive to consistently push through hard late-level obstacles. More training time and better protections cannot overcome a ceiling imposed by the reward structure itself.

3. **Higher entropy floor trades stability for consistency**: The 0.03 floor keeps the policy alive (no collapse at 10M — first D-family extension to survive) but prevents consolidation of gains, creating the oscillation pattern.

### Lessons

44. **D-family caps at ~18% stochastic flag rate** — neither more training (K: 10M) nor stability protections (`target_kl`, higher entropy floor) can push past it. The ceiling is in the reward structure, not the training stability.
45. **Resuming with a different entropy schedule is disruptive** — the ent_coef jump cost ~3.5M steps of recovery. Fresh runs or matching the resume state are preferable.
46. **`target_kl=0.05` + `ent_coef_final=0.03` successfully prevents collapse through 10M steps** — the protections work as intended. The policy survived where F died. But survival ≠ improvement.
47. **The project needs stronger reward signal to break past 18-26%** — both families (D gentle, J aggressive) have limitations. A middle-ground reward approach is the logical next experiment.

---

## Ablation L — Moderate Reward Shaping (Middle Ground)

**Date**: May 5–6, 2026
**Config**: `configs/experiments/ablation_l.yaml`

**Hypothesis**: D-family caps at 18% (reward too gentle). J-family reaches 26% in a narrow window (reward too aggressive, accelerates entropy erosion). A middle-ground reward (`flag_bonus=100`, `time_penalty=-0.05`) with stability protections (`target_kl=0.05`, `ent_coef 0.05→0.03`) should break D's ceiling while maintaining broader stability than J.

| Changed | Ablation D | Ablation J | Ablation L |
|---|---|---|---|
| flag_bonus | 50 | 200 | **100** |
| time_penalty | 0 | -0.1 | **-0.05** |
| ent_coef_final | 0.02 | 0.03 | **0.03** |
| target_kl | None | 0.05 | **0.05** |
| total_timesteps | 5M | 10M (resumed) | **10M (fresh)** |

**Method**: Fresh 10M run (no resume).

### Results — Checkpoint Sweep (50 stochastic episodes each)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **9.0M** | **30% (15/50)** | 2,302 | 3,161 | 2,905 | 263 |
| 2 | 8.5M | 24% (12/50) | 1,917 | 3,161 | 2,407 | 227 |
| 3 | best_model | 20% (10/50) | 2,124 | 3,161 | 2,670 | 234 |
| 4 | 8.0M | 12% (6/50) | 1,555 | 3,161 | 1,935 | 175 |
| 5 | 5.5M | 10% (5/50) | 1,957 | 3,161 | 2,433 | 267 |
| 6 | final_model | 8% (4/50) | 1,739 | 3,161 | 2,161 | 206 |
| — | 9.5M | 0% (0/50) | 571 | 594 | 227 | 1,847 |
| — | 6.0M | 0% (0/50) | 387 | 1,130 | 340 | 426 |

### Training Profile

- **0–5M (slow cook)**: 0–3% training flag_rate, learning forward movement
- **5–5.5M (breakthrough)**: First sustained flag captures, 10% stochastic
- **5.5–9.0M (climbing)**: Steady improvement to 30% peak, training flag% hits 28% at 9.4M
- **9.5M (catastrophe)**: Policy freezes — mean_x=571, steps=1,847
- **10.0M (partial recovery)**: 4% flag rate
- **Training time**: ~4h 15min

### Stability

CollapseDetector fired in multiple bursts (~720K, 1.8M, 2.5M, 3.0–3.4M, 3.8–3.9M, 6.0–6.2M). The 9.5M catastrophe is the worst single-checkpoint collapse in the project — policy completely degenerates then partially recovers. The useful window is 8.0–9.0M (3 checkpoints above 12%).

### Verdict: **New project champion at 30%. Moderate reward shaping hypothesis validated.**

L 9.0M beats J 8.0M (26%) by 4 percentage points. The middle-ground reward (flag_bonus=100, time_penalty=-0.05) broke D's 18% ceiling while slightly widening J's narrow performance window (3 checkpoints ≥12% vs J's 1).

### Updated Global Rankings

| Rank | Model | Flag% | Family |
|---|---|---|---|
| 1 | **L 9.0M** | **30% (15/50)** | L (moderate shaping + target_kl) |
| 2 | J 8.0M | 26% (13/50) | J (aggressive shaping + target_kl) |
| 3 | L 8.5M | 24% (12/50) | L |
| 4 | B best_model | 22% (11/50) | B (static ent_coef) |
| 5 | L best_model | 20% (10/50) | L |
| 6 | D 5.0M / K 9.0M | 18% (9/50) | D / K |

### Lessons

48. **Moderate reward shaping outperforms both extremes** — flag_bonus=100 + time_penalty=-0.05 is the new best configuration. D (50/0) was too gentle, J (200/-0.1) too aggressive.
49. **L still suffers from narrow-peak fragility** — the useful window (8.0–9.0M) is only ~1M steps wide, similar to J. The 9.5M catastrophe shows the policy can still fully collapse despite `target_kl` protection.
50. **30% is a new ceiling to break** — 50 percentage points remain to the 80% target. The climbing phase (5.5–9M) shows the policy learns progressively, suggesting either longer stable training or better reward structure could push higher.
51. **Fresh runs avoid the resume ent_coef discontinuity** — Unlike K (resumed with mismatched schedule), L's fresh start allowed clean learning from scratch.

---

## Ablation M — Resume L 9.0M with Halved LR

**Date**: May 6, 2026
**Config**: `configs/experiments/ablation_m.yaml`

**Hypothesis**: L peaked at 30% (9.0M) then catastrophically collapsed at 9.5M. Halving LR to 1.25e-4 reduces update magnitude, allowing the policy to consolidate gains past the collapse point.

| Changed | Ablation L | Ablation M |
|---|---|---|
| lr | 0.00025 | **0.000125** |
| total_timesteps | 10M | **12M** |

**Method**: Resumed from L 9.0M checkpoint. 3M additional steps (~1h 15min).

### Results — Checkpoint Sweep (50 stochastic episodes each)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **best_model** | **36% (18/50)** | 2,470 | 3,161 | 3,130 | 264 |
| 2 | 10.5M | 36% (18/50) | 2,436 | 3,161 | 3,080 | 290 |
| 3 | 9.5M | 28% (14/50) | 2,176 | 3,161 | 2,742 | 248 |
| 4 | 11.5M | 16% (8/50) | 1,782 | 3,161 | 2,229 | 195 |
| 5 | 12.0M | 14% (7/50) | 2,026 | 3,161 | 2,518 | 302 |
| 6 | 10.0M | 12% (6/50) | 1,597 | 3,161 | 1,981 | 213 |
| — | 11.0M | 0% (0/50) | 949 | 2,026 | 1,113 | 260 |

### Key Comparison — L vs M at the Same Checkpoints

| Checkpoint | L (LR=2.5e-4) | M (LR=1.25e-4) |
|---|---|---|
| 9.5M | 0% (catastrophic) | **28%** (stable) |
| 10.0M | 4% (partial recovery) | 12% |
| 10.5M | — | **36%** (new peak) |

Halved LR directly prevented L's 9.5M catastrophe and enabled continued climbing to 36%.

### Stability

- 6 checkpoints ≥10% (vs L's 3) — broader useful window
- 11.0M = 0% collapse (mean_x=949), but recovers at 11.5M — less severe than L's 9.5M death
- CollapseDetector fired at ~10.2M and ~10.9M, self-resolved
- Peak training flag_rate: 38% (at 9.5M and 10.7M)
- Final training flag_rate: 29% at 12M — policy alive at end

### Verdict: **New project champion at 36%. Halved LR strategy validated.**

M best_model beats L 9.0M (30%) by 6 percentage points. The same resume-with-lower-LR pattern that failed for K (due to ent_coef discontinuity) succeeded here because the entropy schedule was near-continuous (0.035 at resume point).

### Updated Global Rankings

| Rank | Model | Flag% | Family |
|---|---|---|---|
| 1 | **M best_model / M 10.5M** | **36% (18/50)** | M (L + half LR) |
| 2 | L 9.0M | 30% (15/50) | L |
| 3 | M 9.5M | 28% (14/50) | M |
| 4 | J 8.0M | 26% (13/50) | J |
| 5 | L 8.5M | 24% (12/50) | L |
| 6 | B best_model | 22% (11/50) | B |

### Lessons

52. **Halving LR at the performance frontier prevents collapse and enables continued improvement** — L's catastrophic 9.5M → M's stable 28% at same checkpoint. The policy was dying from oversized updates, not fundamental instability.
53. **"Resume from peak with lower LR" is a repeatable strategy** — 1 hour of cheap training yielded +6% over L's entire 10M run. This pattern should be iterated.
54. **Entropy erosion still eventually kills** — M's 11.0M dip (0%) shows the policy remains vulnerable, just on a longer timescale. The oscillation pattern (36% → 0% → 16% → 14%) suggests further LR reduction or additional protections are needed.

---

## Ablation N — Resume M 10.5M with LR Halved Again

**Date**: May 7, 2026
**Config**: `configs/experiments/ablation_n.yaml`

**Hypothesis**: M peaked at 36% (10.5M) then dipped at 11.0M. Halving LR again to 6.25e-5 should stabilize further and allow continued climbing.

| Changed | Ablation M | Ablation N |
|---|---|---|
| lr | 0.000125 | **0.0000625** |
| total_timesteps | 12M | **13.5M** |

**Method**: Resumed from M 10.5M checkpoint. 3M additional steps (~1h 20min).

### Results — Checkpoint Sweep (50 stochastic episodes each)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **13.5M** | **64% (32/50)** | 2,858 | 3,161 | 3,645 | 355 |
| 2 | best_model | 60% (30/50) | 2,820 | 3,161 | 3,590 | 355 |
| 3 | final_model | 58% (29/50) | 2,747 | 3,161 | 3,496 | 345 |
| 4 | 12.5M | 36% (18/50) | 2,511 | 3,161 | 3,179 | 282 |
| 5 | 13.0M | 26% (13/50) | 2,318 | 3,161 | 2,914 | 293 |
| 6 | 11.5M | 18% (9/50) | 2,227 | 3,161 | 2,750 | 442 |
| 7 | 12.0M | 16% (8/50) | 1,950 | 3,161 | 2,444 | 210 |
| — | 11.0M | 0% (0/50) | 1,135 | 2,225 | 1,359 | 243 |

### The Compounding LR Strategy

| Stage | LR | Peak Flag% | Additional Steps |
|---|---|---|---|
| L (fresh) | 2.5e-4 | 30% | 10M |
| M (resume L 9.0M) | 1.25e-4 | 36% | +3M |
| **N (resume M 10.5M)** | **6.25e-5** | **64%** | **+3M** |

Each halving produces a larger gain: +6% → **+28%**.

### Training Profile

- **10.5–11.0M**: Severe collapse inherited from M's pattern — CollapseDetector fired 10 consecutive warnings, 0% at checkpoint
- **11.0–12.5M**: Steady recovery (0% → 18% → 36%)
- **12.5–13.5M**: Explosive growth — training flag_rate 38% → 62% at finish
- **Still climbing at 13.5M**: No sign of convergence or plateau

### Verdict: **Breakthrough result at 64%. Policy within striking distance of 80% target.**

The compounding LR halving strategy produced near-doubling of performance (36% → 64%) in just 3M steps. The policy survived a severe early collapse and kept climbing — lower LR makes collapses recoverable rather than terminal. The final training flag_rate (62%) closely matches the stochastic sweep (64%), indicating the policy is genuinely robust at this level.

### Lessons

55. **LR halving compounds — each iteration produces larger gains** — L→M was +6%, M→N was +28%. Lower LR allows stable exploitation of increasingly refined policies.
56. **Lower LR makes collapses recoverable** — N survived 10 consecutive collapse warnings at 10.7–11.0M and fully recovered. At higher LR this would have been terminal.
57. **The policy was still climbing at 13.5M with training flag_rate 62%** — suggesting further training at this or lower LR could push past 64%.

---

## Ablation O — Resume N 13.5M with LR Halved Again

**Date**: May 7, 2026
**Config**: `configs/experiments/ablation_o.yaml`

**Hypothesis**: N peaked at 64% (13.5M) and was still climbing (training flag_rate 62%). The compounding LR strategy has produced +6%, +28% gains on successive halvings. One more halving should push past 64% toward the 80% target.

| Changed | Ablation N | Ablation O |
|---|---|---|
| lr | 0.0000625 | **0.00003125** |
| total_timesteps | 13.5M | **16.5M** |

**Method**: Resumed from N 13.5M checkpoint. 3M additional steps (~1h 17min).

### Results — Checkpoint Sweep (50 stochastic episodes each)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **14.5M** | **74% (37/50)** | 2,923 | 3,161 | 3,746 | 330 |
| 2 | 15.0M | 66% (33/50) | 2,827 | 3,161 | 3,617 | 314 |
| 3 | 14.0M | 60% (30/50) | 2,723 | 3,161 | 3,451 | 407 |
| 4 | best_model | 54% (27/50) | 2,494 | 3,161 | 3,163 | 341 |
| 5 | 16.5M | 26% (13/50) | 2,117 | 3,161 | 2,655 | 282 |
| 6 | final_model | 14% (7/50) | 1,963 | 3,161 | 2,449 | 252 |
| 7 | 16.0M | 6% (3/50) | 1,596 | 3,161 | 1,977 | 190 |
| 8 | 15.5M | 0% (0/50) | 1,182 | 2,018 | 1,447 | 137 |

### The Compounding LR Strategy (updated)

| Stage | LR | Peak Flag% | Gain | Additional Steps |
|---|---|---|---|---|
| L (fresh) | 2.5e-4 | 30% | — | 10M |
| M (resume L 9.0M) | 1.25e-4 | 36% | +6% | +3M |
| N (resume M 10.5M) | 6.25e-5 | 64% | +28% | +3M |
| **O (resume N 13.5M)** | **3.125e-5** | **74%** | **+10%** | **+3M** |

### Training Profile

- **13.5–14.5M (golden window)**: Training flag% ramped 36→76%, peak at 14.5M
- **14.5–15.2M (plateau/decline)**: 64–71% training flag%, still strong
- **15.3–15.5M (collapse)**: Cliff-edge — training flag% → 0%, CollapseDetector warning at 15.45M (entropy_vel=-0.056, entropy=-0.507)
- **15.5–16.5M (partial recovery)**: Slow climb back to 21% by end — unique among collapses

### Key Observations

- **Golden window is ~1.5M steps wide** (14.0–15.0M): 3 checkpoints above 60%
- **Entropy collapse persists even at LR=3.125e-5** — the mechanism is not purely about update size but cumulative drift
- **Partial recovery is new behavior**: Previous collapses were permanent (G, F) or very slow (J). O's policy partially recovered from 0% to 26% in ~1M steps, suggesting the very low LR allows self-correction
- **Training metrics correlate well with stochastic eval at high quality**: peak training 76% ↔ stochastic 74% at same checkpoint

### Updated Global Rankings

| Rank | Model | Flag% | Family |
|---|---|---|---|
| 1 | **O 14.5M** | **74% (37/50)** | O (L + 3× LR halving) |
| 2 | O 15.0M | 66% (33/50) | O |
| 3 | N 13.5M | 64% (32/50) | N |
| 4 | O 14.0M / N best_model | 60% (30/50) | O / N |
| 5 | N final_model | 58% (29/50) | N |
| 6 | O best_model | 54% (27/50) | O |
| 7 | M best_model / M 10.5M | 36% (18/50) | M |

### Verdict: **New project champion at 74%. 6 percentage points from solving.**

### Extended Re-evaluation (200 episodes)

O 14.5M was re-evaluated with 200 stochastic episodes to get a tighter confidence interval:

| Metric | 50-episode estimate | 200-episode estimate |
|---|---|---|
| Flag rate | 74% (37/50) | **71.0% (142/200)** |
| mean_x_pos | 2,923 | 2,894 |
| mean_reward | 3,746 | 3,707 |
| mean_steps | 330 | 324 |

**True rate: ~71%** (95% CI: ~65–77%). The 50-episode sample was slightly optimistic.

### Failure Mode Analysis (200 episodes)

The 58 failures cluster at specific obstacles:

| Failure Zone | x_pos range | Count | Share |
|---|---|---|---|
| Early death | ~685–704 | 3 | 5% |
| Mid-level | ~1783–1960 | ~15 | 26% |
| **Late-level wall** | **~2469–2477** | **~33** | **57%** |
| Near-flag | ~2758–2761 | 3 | 5% |

**x≈2470 is the dominant bottleneck** — over half of all failures die at the same obstacle. Solving this single choke point would push from 71% to ~87%.

### Lessons

58. **Compounding LR strategy continues to work but with diminishing returns** — gains per halving: +6%, +28%, +10%. The +10% from O is smaller than N's +28%, indicating the strategy may be approaching its limit.
59. **Entropy collapse is inevitable regardless of LR** — even at 1/8th original LR, the policy still hits a collapse cliff. The fundamental cause is cumulative entropy erosion, not individual update magnitude.
60. **Very low LR enables partial collapse recovery** — a new phenomenon. The policy recovered from 0% to 26% post-collapse, suggesting the gradient signal is small enough that the policy can self-correct. This may be exploitable with longer training.
61. **200-episode eval reveals the true rate is ~71%, not 74%** — 50-episode estimates have ±12% CI at this quality level. Use larger samples for models near the target.
62. **Failures concentrate at x≈2470** — 57% of all failures die at the same late-level obstacle. The remaining gap is a single-obstacle consistency problem, not a general capability problem.

---

## Ablation P — Resume O 14.5M with LR Halved, Dense Checkpoints

**Date**: May 8, 2026
**Config**: `configs/experiments/ablation_p.yaml`

**Hypothesis**: One more LR halving with dense checkpoints (100K intervals) should capture any improvement within the golden window.

| Changed | Ablation O | Ablation P |
|---|---|---|
| lr | 0.00003125 | **0.000015625** |
| total_timesteps | 16.5M | **16.5M** (2M additional from 14.5M) |
| checkpoint_freq | 500K | **100K** (dense — new config field) |

**Method**: Resumed from O 14.5M checkpoint. 2M additional steps (~1h 5min). New `checkpoint_freq` config field added to `TrainingConfig` to support dense checkpointing.

### Results — Checkpoint Sweep (50 stochastic episodes each, 22 models)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | **14.8M** | **68% (34/50)** | 2,917 | 3,161 | 3,729 | 341 |
| 2 | 14.7M | 68% (34/50) | 2,786 | 3,161 | 3,564 | 318 |
| 3 | 14.9M | 66% (33/50) | 2,836 | 3,161 | 3,616 | 363 |
| 4 | 14.6M | 64% (32/50) | 2,856 | 3,161 | 3,646 | 336 |
| 5 | 16.4M | 50% (25/50) | 2,577 | 3,161 | 3,246 | 422 |
| 6 | final_model | 48% (24/50) | 2,507 | 3,161 | 3,190 | 272 |
| 7 | 16.0M | 44% (22/50) | 2,443 | 3,161 | 3,102 | 272 |
| — | 15.0M | 0% (0/50) | 448 | 702 | 46 | 1,928 |
| — | 15.1M | 0% (0/50) | 592 | 722 | 223 | 1,967 |

### The Compounding LR Strategy (final)

| Stage | LR | Peak Flag% | Gain | Additional Steps |
|---|---|---|---|---|
| L (fresh) | 2.5e-4 | 30% | — | 10M |
| M (resume L 9.0M) | 1.25e-4 | 36% | +6% | +3M |
| N (resume M 10.5M) | 6.25e-5 | 64% | +28% | +3M |
| O (resume N 13.5M) | 3.125e-5 | 71%* | +7% | +3M |
| **P (resume O 14.5M)** | **1.5625e-5** | **68%** | **-3%** | **+2M** |

\* O's true rate revised to 71% via 200-episode re-evaluation.

### Key Findings

- **LR halving has hit its limit** — P's best (68%) is *below* O's true rate (71%). Updates are now too small to improve the policy.
- **Collapse hit earlier** (15.0M vs O's 15.3M) and was more severe (0%, x_pos=448 — near-instant death)
- **Golden window was only ~300K steps** (14.6–14.9M) before the cliff — the narrowest of any ablation
- **Dense checkpoints confirmed**: no hidden peak between 500K markers. The true best was at 14.8M, consistent with coarser sampling
- **Post-collapse partial recovery**: policy climbed back to 50% at 16.4M, confirming low-LR self-correction

### Verdict: **Compounding LR strategy exhausted. P regresses from O.**

### Lessons

63. **LR=1.5625e-5 is below the useful threshold** — at this LR, updates are too small to overcome the x≈2470 obstacle. The policy can't learn new behaviors, only slowly drift.
64. **Dense checkpointing (100K) confirmed no hidden peak** — the 500K grid was not missing a materially better checkpoint. The true optimum aligns with the coarser sweep.
65. **The compounding LR strategy has a natural endpoint** — 5 iterations (L→M→N→O→P) identified the useful LR range as ~6.25e-5 to ~3.125e-5 for this policy. Below that, diminishing returns become negative returns.

---

## What's Next

**O 14.5M remains the project champion** at 71% stochastic flag capture (142/200).

**Gap to target**: 71% → 80% (9 percentage points remaining).

**What P proved**: The LR halving ladder is exhausted. Further reduction hurts rather than helps. The remaining gap is not a training stability or hyperparameter problem — it is a **single-obstacle consistency problem** at x≈2470, which accounts for 57% of all failures.

---

## Ablation Q — Resume O 14.5M with Higher Entropy Floor + Fewer Epochs

**Date**: May 8–9, 2026
**Config**: `configs/experiments/ablation_q.yaml`

**Hypothesis**: Collapses happen because entropy erodes below a critical threshold with `ent_coef_final=0.03`. Raising the floor to 0.04 + reducing `n_epochs` from 4 to 3 should extend the golden window.

| Changed | Ablation O | Ablation Q |
|---|---|---|
| ent_coef_final | 0.03 | **0.04** |
| n_epochs | 4 | **3** |
| total_timesteps | 16.5M | **17.5M** |
| checkpoint_freq | 500K | **100K** |

**Method**: Resumed from O 14.5M checkpoint. 3M additional steps (~55 min).

### Results — Checkpoint Sweep (50 stochastic episodes each, 32 models)

| Rank | Checkpoint | Flag% | mean_x | max_x | reward | steps |
|---|---|---|---|---|---|---|
| 1 | best_model | 72% (36/50) | 2,968 | 3,161 | 3,799 | 347 |
| 2 | 14.6M | 72% (36/50) | 2,955 | 3,161 | 3,781 | 351 |
| 3 | 16.6M | 28% (14/50) | 2,214 | 3,161 | 2,768 | 348 |
| 4 | 17.3M | 18% (9/50) | 2,015 | 3,161 | 2,521 | 255 |
| 5–32 | all others | 0–14% | — | — | — | — |

### What Went Wrong — The Entropy Discontinuity (Repeat of Ablation K)

**Q repeated the exact failure mode already documented in Ablation K (lesson #45).**

- O's ent_coef at 14.5M: **~0.032** (0.05→0.03 schedule at 88% progress)
- Q's ent_coef at 14.5M: **~0.042** (0.05→0.04 schedule at 83% of 17.5M)

The ent_coef **jumped 30% on resume** (0.032→0.042), immediately destabilizing the policy. The agent went from 72% to 0% (x_pos=297, instant death) within 200K steps. The policy was frozen at x=315 for over 1M steps (14.8–15.7M) before slowly recovering.

The 72% best_model is just O's checkpoint re-saved before Q's changes took effect — Q contributed nothing new.

### Verdict: **Failed. Entropy discontinuity on resume destroyed the policy (same as K).**

### Lessons

66. **Changing `ent_coef_final` on resume is destructive — now confirmed twice** (K: 0.02→0.04 destroyed D; Q: 0.03→0.04 destroyed O). The entropy schedule produces a discontinuity when the floor changes because the schedule interpolation recalculates the current ent_coef. This is a hard rule: **never change `ent_coef` or `ent_coef_final` when resuming.**
67. **The `n_epochs` change could not be evaluated** — the entropy discontinuity dominated. Whether n_epochs=3 helps or hurts remains unknown. Testing it requires an isolated run with matching entropy parameters.
68. **Recovery from collapse is possible but insufficient** — Q partially recovered to 28% (16.6M) and 18% (17.3M), but never approached O's 71%. Even with very low LR + higher entropy floor, a destroyed policy cannot fully recover to its pre-collapse quality.

---

## Ablation R — Resume O 14.5M with n_epochs=3 (Isolated Test)

**Date**: May 9–10, 2026
**Config**: `configs/experiments/ablation_r.yaml`

**Hypothesis**: Q's entropy discontinuity prevented testing n_epochs=3. This isolates it by keeping `ent_coef_final=0.03` (matching O exactly). Fewer gradient passes should extend the golden window.

| Changed | Ablation O | Ablation R |
|---|---|---|
| n_epochs | 4 | **3** |
| total_timesteps | 16.5M | **17.5M** |
| checkpoint_freq | 500K | **100K** |

**Method**: Resumed from O 14.5M. 3M additional steps.

### Results — Sweep (50 episodes)

| Metric | Best |
|---|---|
| Peak | 74% (final_model, 17.5M) |
| 2nd | 70% (15.7M, 14.7M) |
| Checkpoints ≥60% | 5 |

### 200-Episode Re-Eval of R final_model: **64% (128/200)**

Significantly worse than O's 71% (142/200). The 50-ep estimate (74%) was 10 points optimistic.

**Failure zones** (72 failures): early ~670–712 (5.6%), mid-early ~1433–1515 (9.7%), mid ~1787–2028 (31.9%), late wall ~2469–2476 (38.9%), near-flag ~2747–2850 (13.9%).

### Verdict: **n_epochs=3 extends training life (3M wide window vs 1M) but lowers the ceiling.** Policy oscillates rather than locking in. O 14.5M remains champion.

### Lessons

69. **n_epochs=3 prevents permanent collapse but degrades precision.** Fewer gradient passes mean the policy never consolidates obstacle passes as firmly — it spreads failures across all zones instead of concentrating at x≈2470.
70. **50-episode sweeps are dangerously noisy.** R: 74%→64% (10-point drop on re-eval). O: 74%→71% (3-point drop). Treat 50-ep results as rough screening only.
71. **"Resume from O and tweak hyperparameters" is exhausted.** LR halving (P), entropy floor (Q), and n_epochs (R) all failed to beat O's 71%.

---

## Multi-Seed Phase L — Fresh L Recipe with Seeds 1, 2, 3

**Date**: May 10–13, 2026
**Config**: `configs/experiments/ablation_l.yaml` with `--seed 1/2/3`

**Rationale**: The "resume from O" approach is exhausted. The compounding LR pipeline (L→M→N→O) is proven to reach 71% with seed 42. Running multiple seeds from scratch exploits seed variance — a luckier seed may exceed 80% through the same pipeline.

### Results — Phase L Sweep (50 episodes each)

| Seed | Best Checkpoint | Peak Flag% | At Step |
|---|---|---|---|
| **Seed 1** | 8.0M | **54%** | 8.0M |
| Seed 2 | best_model | 42% | ~8.5M |
| Seed 3 | best_model | 8% | ~9.2M |
| Original (seed 42) | 9.0M | 30% | 9.0M |

### Key Observations

- **Seed 1 is exceptional**: 54% at 8.0M — 24 points above the original seed 42 at the same phase. This seed learned the level faster and more robustly.
- **Seed 2** is marginally above original but not dramatic.
- **Seed 3** is dead — extreme entropy collapse throughout, never cracked 10%.

### Verdict: **Seed 1 is the most promising candidate.** Continue through M→N→O pipeline. With a 54% starting point (vs original's 30%), there's a realistic chance the compounding LR halvings push past 80%.

### Lessons

72. **Seed variance is massive** — same config produces 8% to 54% across 3 seeds. Seed selection is a legitimate optimization axis.
73. **Seed 1 learned earlier and more stably** than seed 42 — already 54% at 8.0M vs seed 42's 30% at 9.0M. This gives the compounding pipeline a better foundation to build on.

---

## What's Next

**O 14.5M remains the project champion** at 71% stochastic flag capture (142/200).

**Active strategy**: Continue seed 1 through M→N→O compounding LR pipeline. Next: Phase M (resume seed 1's 8.0M with LR=1.25e-4).

**Gap to target**: 71% → 80% (9 percentage points remaining).

**Hard rules established:**
- Do NOT change `ent_coef` or `ent_coef_final` when resuming
- Do NOT halve LR below 3.125e-5 (P proved negative returns)

---

## Multi-Seed Phase M — SOLVED: 87.5% Flag Capture (175/200)

**Date**: May 13–14, 2026
**Config**: `configs/experiments/ablation_m.yaml` with `--seed 1`
**Model**: `results/multiseed_s1_M/models/checkpoints/ppo_mario_12000000_steps.zip`

### Setup

Resumed from seed 1's Phase L best checkpoint (8.0M, 54%) with LR halved to 1.25e-4. Trained 4M additional steps (8.0M → 12.0M). This is the first LR halving in the compounding pipeline for this seed.

| Parameter | Value |
|---|---|
| lr | 1.25e-4 (halved from L's 2.5e-4) |
| n_epochs | 4 |
| ent_coef | 0.05 → 0.03 (linear schedule) |
| target_kl | 0.05 |
| total_timesteps | 12M |
| seed | 1 |
| Resume from | multiseed_s1_L 8.0M |

### Training Trajectory

The training flag_rate climbed steadily: 37% → 45% → 56% → 72% → **84% → 92%** by 12.0M. A brief entropy collapse warning at ~10.1M (flag_rate dropped to 7%) was self-correcting — the policy recovered within 200K steps and continued climbing.

### 50-Episode Sweep Results

| Rank | Checkpoint | Flag% |
|---|---|---|
| 1 | 12.0M | **82%** (41/50) |
| 2 | best_model | **82%** (41/50) |
| 3 | final_model | **82%** (41/50) |
| 4 | 11.0M | 72% (36/50) |
| 5 | 11.5M | 70% (35/50) |
| 6 | 9.0M | 64% (32/50) |

Three checkpoints at ≥80% — far more robust than any prior result.

### 200-Episode Confirmation: **87.5% (175/200)**

| Metric | Value |
|---|---|
| **Flag capture rate** | **87.5% (175/200)** |
| mean_x_pos | 3011 |
| max_x_pos | 3161 |
| mean_reward | 3869.5 |
| mean_steps | 362 |

This exceeds the 80% target by 7.5 percentage points. The 50-episode estimate (82%) was actually **conservative** — the true rate is higher.

### Failure Analysis (25/200 episodes failed)

| Zone | x range | Count | % of failures |
|---|---|---|---|
| Early | ~829–898 | 4 | 16% |
| Mid-early | ~1433–1435 | 2 | 8% |
| Mid | ~1788–1964 | 9 | 36% |
| Late wall | ~2466–2472 | 7 | 28% |
| Near-flag | ~2763–2764 | 3 | 12% |

The x≈2470 bottleneck that dominated O's failures (57% of all failures) is now just 28%. This seed solved the wall obstacle much more consistently than seed 42 ever did.

### Why This Seed Won

1. **Better early learning**: Seed 1 reached 54% after Phase L alone (vs 30% for seed 42). It entered Phase M with a stronger, more consolidated policy.
2. **Only needed one LR halving**: Seed 42 needed four halvings (L→M→N→O) and still peaked at 71%. Seed 1 solved the level in just two phases (L→M).
3. **Total training**: 12M steps total (~4 hours GPU). Seed 42 used 16.5M steps and never crossed 80%.

### The Winning Recipe (Complete Specification)

```yaml
env:
  game: SuperMarioBros-1-1-v0
  movement: SIMPLE_MOVEMENT
  frame_skip: 4
  frame_stack: 4
  obs_size: 84
  num_envs: 16

training:
  total_timesteps: 12000000
  lr: 0.000125
  n_steps: 1024
  batch_size: 512
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.05
  ent_coef_final: 0.03
  target_kl: 0.05

reward:
  forward_scale: 0.3
  death_penalty: -15.0
  flag_bonus: 100.0
  time_penalty: -0.05

seed: 1
```

**Method**: Train Phase L (LR=2.5e-4) for 10M steps → sweep → pick best checkpoint (8.0M) → resume Phase M (LR=1.25e-4) to 12M → evaluate.

### Verdict: **PROJECT SOLVED.**

### Lessons

74. **Seed selection + compounding LR is the winning strategy.** The recipe (L config + LR halvings) was already proven — it just needed a better seed.
75. **A stronger Phase L foundation dramatically reduces the work needed.** Seed 1 at 54% needed only one halving to reach 87.5%. Seed 42 at 30% needed four halvings and still capped at 71%.
76. **The 50-episode screening methodology works.** Screen with 50 episodes, confirm with 200. In this case the 50-ep estimate (82%) was actually conservative (true rate: 87.5%).
77. **Multi-seed is not expensive.** Three Phase L runs cost ~4.5 hours. The winning seed then needed only one more 2-hour Phase M run. Total additional cost to solve: ~6.5 hours of GPU compared to dozens of hours of failed ablations on seed 42.

---

## Project Summary

**Target**: ≥80% stochastic flag capture over 50 episodes on Super Mario Bros Level 1-1.

**Achieved**: **87.5% (175/200)** — confirmed with full 200-episode evaluation.

**Final model**: `results/multiseed_s1_M/models/checkpoints/ppo_mario_12000000_steps.zip`

**Total ablations**: A through R + multi-seed = 21 experiment configurations across Phase 3.

**Key breakthroughs**:
- Phase 1–2: Established SB3 PPO + CnnPolicy pipeline, reward shaping, entropy scheduling
- Ablation L: Found the core recipe (flag_bonus=100, time_penalty=-0.05, ent 0.05→0.03, target_kl=0.05)
- Ablations M–O (seed 42): Proved compounding LR strategy works (30% → 71%)
- Ablations P–R: Proved the "tweak and resume" approach is exhausted for seed 42
- Multi-seed Phase L: Exploited seed variance to find a better starting trajectory
- Multi-seed Phase M: **Solved** in one halving (54% → 87.5%)

**Progress tiers (final)**:
- <20%: Early progress ← D-family ceiling (18%)
- 20–50%: Significant progress ← L (30%), seed 1 L (54%)
- 50–80%: Strong result ← O seed 42 (71%)
- ≥80%: **SOLVED** ← **Seed 1 Phase M: 87.5% (175/200)** ✓
