# Project Mario — Master Gameplan

> **Purpose**: This file is the single source of truth for the project's goals, architecture,
> phases, and decisions. Reference it at the start of every session to restore context.
> Last updated: April 13, 2026.

---

## 1. Project Vision

**Primary Goal**: Train a PPO agent that can reliably complete Super Mario Bros Level 1-1
(World 1-1) from start to finish (flag capture).

**Future Goals** (in order of ambition):
1. Generalized agent that can play multiple levels (1-1 through 1-4, then beyond)
2. Beat the entire game (Worlds 1–8)
3. Rival human speedrun times on a single level (~340-350 second in-game timer remaining
   for world-class 1-1 runs; the fastest TAS completes 1-1 in roughly the equivalent of
   the 370+ timer range due to frame-perfect movement)

---

## 2. Recommended Stack

| Component            | Library / Tool                          | Version Target     | Why                                                                 |
|----------------------|-----------------------------------------|--------------------|---------------------------------------------------------------------|
| **Environment**      | `gym-super-mario-bros`                  | 7.4.0              | Only maintained Mario NES env. Needs Gymnasium compatibility shim.  |
| **Gym API**          | `gymnasium`                             | >= 0.29            | Modern standard. Replaces `gym`. 5-tuple step returns.              |
| **Compat Shim**      | `shimmy[gym-v21]`                       | >= 1.0             | Bridges old `gym 0.21` envs to Gymnasium API automatically.        |
| **NES Emulator**     | `nes-py`                                | 8.2.1              | Required by gym-super-mario-bros. C extension, platform-sensitive.  |
| **RL Framework**     | `stable-baselines3`                     | >= 2.3             | Industry standard for applied RL. Gymnasium-native from v2.0+.     |
| **Deep Learning**    | `torch`                                 | >= 2.1             | PyTorch 2.x with `torch.compile()`. CUDA 12.x support.             |
| **Config**           | `dataclasses` + YAML (or `hydra-core`)  | —                  | Clean experiment configs. Hydra is optional but impressive.         |
| **Experiment Track** | `tensorboard` (bundled with SB3)        | —                  | Training curves. Add `wandb` later for polish.                      |
| **Video Recording**  | `gymnasium.wrappers.RecordVideo`        | —                  | Capture agent gameplay for README/demos.                            |
| **Python**           | `>= 3.10`                               | 3.10–3.12          | Match PyTorch and SB3 compatibility.                                |

### Why NOT these alternatives:
- **`gym` (OpenAI)**: Unmaintained since 2022. Using it signals outdated skills.
- **CleanRL as primary**: Great for learning PPO internals. Not for polished training runs.
  We use it as a reference in Phase 4 (PPO from scratch), not as the main framework.
- **`torch 1.x`**: Missing `torch.compile()`, slower, old CUDA. No reason to stay.

---

## 3. Target Project Structure

```
project-mario/
├── README.md                      # Results, GIFs, how to reproduce
├── gameplan.md                    # THIS FILE — project plan and context
├── pyproject.toml                 # Modern Python packaging (or requirements.txt)
├── configs/
│   ├── default.yaml               # Base hyperparameters
│   └── experiments/               # Ablation/experiment variations
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── wrappers.py            # SkipFrame, reward shaping wrappers
│   │   └── mario_env.py           # Environment factory: create_mario_env()
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Load model → render / record video
│   ├── callbacks.py               # Custom SB3 callbacks (optional)
│   └── config.py                  # Dataclass-based config loading from YAML
├── results/
│   ├── models/                    # Saved checkpoints (.zip)
│   ├── logs/                      # TensorBoard logs
│   └── videos/                    # Recorded gameplay
├── notebooks/
│   └── analysis.ipynb             # Training curves, reward plots, comparisons
└── legacy/                        # Old v1 code (keep for reference, don't run)
    ├── SMv1_config.py
    ├── SMv1_model.py
    ├── SMv1_run.py
    └── SMv1_run_legacy.py
```

---

## 4. Known Issues from V1 (What to Fix / Avoid)

These are bugs and design problems found in the original codebase:

1. **`CustomReward` wrapper has an uninitialized `self.last_time`** — crashes on first
   pipe-stall check. This wrapper was disabled in favor of `SimpleShape`, but keep it
   out of v2 entirely.
2. **`VecNormalize(clip_reward=1.0)`** squashes all reward signals to ±1.0, destroying
   the contrast between the +100 flag bonus and small movement rewards. Either don't
   normalize rewards, or use a larger clip range.
3. **Only 2 parallel environments** — massively underutilizes GPU. A 3080 Ti can handle
   8–16 envs easily for this observation size (84×84 grayscale × 4 frames).
4. **1M timesteps is far too few** — Mario 1-1 typically needs 5–10M+ steps to solve.
5. **Entropy coefficient (0.005) too low + clip range (0.1) too tight** — the agent
   converges on a local policy (run right, never jump over obstacles) and stops exploring.
6. **No seeding** — runs are not reproducible.
7. **`SMv1_run_legacy.py` uses `channels_order='last'`** while training uses `'first'` —
   observation mismatch means legacy runner would produce garbage behavior even with a
   valid model.
8. **Model version confusion** — filenames like `v2`, `v3`, `v4` scattered in code with
   no tracking of what changed between versions.

---

## 5. Phases

### Phase 1 — Foundation & Environment Stack
**Hardware**: CPU only (local dev machine is fine)
**Branch**: `v2-dev`

#### Objectives
- [ ] Set up clean project structure (see Section 3)
- [ ] Create `pyproject.toml` or `requirements.txt` with modern dependencies
- [ ] Build `src/envs/mario_env.py`: environment factory function that returns a properly
      wrapped, Gymnasium-compatible Mario env
- [ ] Implement wrapper stack:
  - `JoypadSpace` (SIMPLE_MOVEMENT — 7 actions)
  - `SkipFrame` (repeat action for N frames, default 4)
  - Grayscale → Resize (84×84) → frame stack (4 frames)
  - NO `VecNormalize` on rewards initially (add later as ablation)
- [ ] Implement `shimmy` or manual compatibility shim for `gym-super-mario-bros` → Gymnasium
- [ ] Add deterministic seeding (env seed, torch seed, numpy seed)
- [ ] Create `configs/default.yaml` with baseline hyperparameters
- [ ] Build `src/config.py` to load YAML into a dataclass
- [ ] Sanity check: run random policy for 1000 steps, print rewards, verify obs shape
- [ ] Record sanity-check video of random agent (proves env pipeline works end-to-end)
- [ ] Move old v1 code to `legacy/` folder

#### Compatibility Notes
`gym-super-mario-bros` depends on `nes-py` which depends on `gym==0.21`. The cleanest
approach is to use `shimmy[gym-v21]` to wrap the old gym env into a Gymnasium env.
Alternatively, write a thin adapter class. This is the trickiest part of Phase 1 —
document whatever solution works in the README.

#### Definition of Done
- `python src/train.py --dry-run` creates the env, runs 100 steps, prints obs shape
  `(n_envs, 4, 84, 84)`, and exits cleanly.
- A 10-second video of random agent gameplay is saved to `results/videos/`.

---

### Phase 2 — Baseline PPO Training (Single Stage: 1-1)
**Hardware**: **GPU required — 3080 Ti** (clone repo to personal PC)
**Branch**: `v2-dev` → merge to `main` when baseline is validated

#### Objectives
- [ ] Build `src/train.py` with full SB3 PPO training loop
- [ ] Parallelize with `SubprocVecEnv`, **8 environments**
- [ ] Set up TensorBoard logging
- [ ] Implement `EvalCallback` with checkpoint saving (eval every 10,000 steps)
- [ ] Implement simple reward shaping wrapper:
  - +1.0 per unit of forward progress (x_pos delta)
  - -15 on death
  - +50 on flag capture
  - (No jump bonuses, no manual action rewards — let PPO learn movement)
- [ ] Train for **5M timesteps** as initial run
- [ ] Build `src/evaluate.py`: load best checkpoint, run 20 episodes, report stats
- [ ] Record best agent gameplay video
- [ ] If 5M is insufficient, extend to 10M

#### Baseline Hyperparameters (Starting Point)

```yaml
# configs/default.yaml
algorithm: PPO
policy: CnnPolicy

# Environment
env_id: SuperMarioBros-1-1-v0
action_space: SIMPLE_MOVEMENT
num_envs: 8
frame_skip: 4
frame_stack: 4
obs_size: 84

# PPO
learning_rate: 2.5e-4
n_steps: 512            # Per env → 4096 total per rollout with 8 envs
batch_size: 256          # Larger batch for stability with more envs
n_epochs: 4              # Fewer epochs per update to avoid overfitting rollout
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2          # SB3 default — don't over-tighten
ent_coef: 0.01           # More exploration than v1's 0.005
vf_coef: 0.5
max_grad_norm: 0.5
normalize_advantage: true

# Training
total_timesteps: 5_000_000
device: cuda
seed: 42

# Evaluation
eval_freq: 10_000
eval_episodes: 5
```

#### Key Changes from V1
| Parameter      | V1 Value | V2 Value | Rationale                                    |
|----------------|----------|----------|----------------------------------------------|
| num_envs       | 2        | 8        | 4× more experience per second                |
| n_steps        | 512      | 512      | Same per-env, but 4096 total (was 1024)      |
| batch_size     | 64       | 256      | Scale with total rollout size                 |
| n_epochs       | 8        | 4        | Less overfitting per rollout batch            |
| clip_range     | 0.1      | 0.2      | More room for policy updates                  |
| ent_coef       | 0.005    | 0.01     | Double exploration pressure                   |
| total_timesteps| 1M       | 5M       | 5× longer — 1M was way too short             |
| VecNormalize   | Yes      | No       | Remove reward squashing for now               |

#### Target Metrics & Milestones

**Mario 1-1 Layout Reference:**
- Level start: x_pos ≈ 40
- First pipe: x_pos ≈ 368
- Second pipe: x_pos ≈ 618
- Third pipe (tall): x_pos ≈ 734
- First gap (pit): x_pos ≈ 1110
- Staircase: x_pos ≈ 2840
- Flagpole: x_pos ≈ 3160
- **Total traversal: ~3120 pixels from start to flag**

**Progress Benchmarks (by timesteps):**

| Timesteps   | Expected x_pos (avg) | What This Means                              |
|-------------|----------------------|----------------------------------------------|
| 500K        | 300–500              | Learned to run right, may clear first pipe   |
| 1M          | 500–800              | Clearing pipes consistently                  |
| 2M          | 800–1200             | Approaching or clearing the first gap         |
| 3M          | 1200–2000            | Past the gap, navigating mid-level enemies    |
| 5M          | 2000–3160            | Reaching late-level or completing the stage   |
| 10M         | 3160 (consistent)    | Reliable flag capture most episodes           |

**These are rough targets. Actual progress depends on reward shaping and hyperparameters.**

**Success Criteria for "Solved 1-1":**
- Agent reaches flagpole (x_pos ≥ 3160) in **≥ 80% of evaluation episodes** over 50+
  eval runs
- Mean episode reward is positive and stable (not oscillating)
- Mean episode length is under 1500 steps (not timing out)

**Metrics to Log (TensorBoard):**
- `rollout/ep_rew_mean` — mean episode reward (primary signal)
- `rollout/ep_len_mean` — mean episode length
- `train/entropy_loss` — if this drops to near-zero, exploration has collapsed
- `train/policy_gradient_loss` — should be small and stable
- `train/value_loss` — should decrease over time
- `train/clip_fraction` — fraction of updates clipped; if >0.3, clip_range may be too tight
- `train/approx_kl` — KL divergence between old and new policy; if spiking, LR too high
- `eval/mean_reward` — from EvalCallback
- **Custom**: `eval/mean_x_pos` — average maximum x position reached (most intuitive metric)
- **Custom**: `eval/flag_capture_rate` — % of eval episodes where flag_get=True
- **Custom**: `train/fps` — frames per second (sanity: should be 800+ with 8 envs on 3080 Ti)

#### Definition of Done
- Agent captures the flag on 1-1 in ≥80% of 50 evaluation episodes.
- TensorBoard logs show clear upward training curve for reward and x_pos.
- Best model checkpoint saved. Gameplay video recorded.

---

### Phase 3 — Iteration & Reward Shaping Ablations
**Hardware**: 3080 Ti (same as Phase 2)
**Branch**: experiment branches off `main`

#### Objectives
- [ ] Run ablation experiments (one variable at a time, compare to baseline):
  - **Reward shaping**: no shaping vs. simple vs. aggressive (movement bonuses)
  - **Action space**: `SIMPLE_MOVEMENT` vs `RIGHT_ONLY` vs `COMPLEX_MOVEMENT`
  - **Entropy coefficient**: 0.005 vs 0.01 vs 0.02
  - **Number of environments**: 4 vs 8 vs 16
  - **n_steps**: 256 vs 512 vs 1024
  - **With vs without VecNormalize** (and different clip_reward values)
- [ ] Track all experiments in TensorBoard (separate run names)
- [ ] Identify best configuration
- [ ] Push timesteps to 10M+ if needed
- [ ] Optional: try learning rate scheduling (linear decay)

#### Key Principle
**Change one thing at a time.** Every experiment should differ from baseline by exactly
one hyperparameter or design choice. This is what distinguishes serious RL work from
random tinkering.

#### Definition of Done
- At least 4 ablation experiments completed and logged.
- Clear winner configuration identified with comparison plots.
- Best agent reliably solves 1-1 (≥90% success rate).

---

### Phase 4 — Implement PPO from Scratch (Learning Exercise)
**Hardware**: CPU for development/debugging, GPU for final training comparison
**Branch**: `ppo-scratch`

> **This phase is optional / time-permitting.** It's about deep understanding, not replacing
> SB3. The SB3 baseline from Phase 2/3 remains the primary trained agent.

#### Objectives
- [ ] Implement a standalone PPO training loop in PyTorch (single file or small module)
- [ ] Use CleanRL's PPO implementation as a reference (not copy — type it out, understand
      every line)
- [ ] Core components to implement:
  - **Rollout buffer**: collect (obs, action, reward, done, log_prob, value) tuples
  - **GAE (Generalized Advantage Estimation)**: compute advantages with λ-returns
  - **Clipped surrogate objective**: L^CLIP(θ) loss function
  - **Value function loss**: MSE between predicted and actual returns
  - **Entropy bonus**: encourage exploration via entropy of action distribution
  - **Minibatch updates**: shuffle rollout, split into minibatches, update K epochs
  - **Vectorized environment stepping**: sync multiple envs
- [ ] Key concepts to understand and document:
  - Why clipping? (prevent destructive policy updates)
  - Why GAE? (bias-variance tradeoff in advantage estimation)
  - What does entropy coefficient control? (exploration vs exploitation)
  - On-policy vs off-policy: why PPO can't reuse old data
  - The importance of advantage normalization
- [ ] Benchmark: train your PPO on Mario 1-1, compare learning curves to SB3 PPO
  - Expectation: similar trends, likely slower/worse absolute performance (SB3 is
    heavily optimized). That's fine — the goal is understanding, not beating SB3.

#### Reference Material
- CleanRL PPO: https://docs.cleanrl.dev/rl-algorithms/ppo/
- Spinning Up PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Original PPO paper: Schulman et al. 2017

#### Definition of Done
- Your PPO implementation trains on Mario 1-1 and shows a clear upward learning curve.
- You can explain every component of the algorithm from memory.
- Side-by-side plot: your PPO vs SB3 PPO learning curves.

---

### Phase 5 — Polish & Resume Packaging
**Hardware**: CPU for writing/plotting, GPU only if re-training
**Branch**: `main`

#### Objectives
- [ ] Write comprehensive README.md:
  - Problem statement and motivation
  - Environment details (Mario NES, observation space, action space)
  - Algorithm overview (PPO with diagram/explanation)
  - Training setup (hardware, hyperparameters, reward shaping)
  - Results (training curves, success rates, x_pos progression)
  - Failure analysis (what the agent still struggles with)
  - Lessons learned
  - How to reproduce (install, train, evaluate)
- [ ] Create highlight video/GIF of best agent completing 1-1
- [ ] Clean up code: remove dead code, add type hints to public APIs, ensure scripts
      have clear `--help` via argparse
- [ ] Create `notebooks/analysis.ipynb` with:
  - Training curve plots (reward, x_pos, success rate vs timesteps)
  - Ablation comparison charts
  - Before/after video frames
- [ ] Ensure full reproducibility: clone → install → train → evaluate should work

#### Definition of Done
- A stranger can clone the repo, follow README instructions, and reproduce training.
- README includes at least one GIF of the agent playing.
- Training curves are visible in README or linked notebook.

---

## 6. Future Phases (Post-Completion)

These phases extend the project beyond the primary goal. Tackle only after Phase 5.

### Future Phase A — Multi-Level Generalization
**Hardware**: Strong GPU (3080 Ti minimum, consider cloud GPU rental: A100/H100)

- Train on multiple stages: curriculum (1-1 → 1-2 → 1-3 → 1-4) or multi-stage sampling
- Evaluate on held-out stages the agent has never trained on
- Analyze where policies break: pipes, gaps, enemies, timing-sensitive jumps
- Compare stage-specific memorization vs. transferable behavior
- This is where the project becomes research-worthy

### Future Phase B — Speedrunning
**Hardware**: Cloud GPU rental strongly recommended (A100/H100 for long training runs)

- Optimize for completion speed, not just completion
- Reward shaping: heavy bonus for remaining time, penalty for slow movement
- May need `COMPLEX_MOVEMENT` action space for advanced techniques
- Compare to human speedrun times (best human 1-1 times are ~4.57 seconds real-time)
- This is the "wow factor" for a resume — "my RL agent rivals human speedrunners"
- Consider frame-perfect analysis: does the agent discover known speedrun strategies?

### Future Phase C — Full Game
**Hardware**: Serious compute (cloud GPU, likely multi-day training)

- Extend curriculum to all 32 stages (Worlds 1–8, 4 stages each)
- May need architectural changes (memory/RNN for navigating maze castles in World 4+)
- Long-horizon credit assignment becomes much harder
- This would be a genuinely impressive achievement in the RL community

---

## 7. Hardware Requirements Summary

| Phase                  | Hardware       | Where to Run           | Estimated Time        |
|------------------------|----------------|------------------------|-----------------------|
| Phase 1 (Foundation)   | CPU only       | Any dev machine        | Days of dev work      |
| Phase 2 (Baseline)     | 3080 Ti        | Personal PC            | 2–6 hours per 5M run  |
| Phase 3 (Ablations)    | 3080 Ti        | Personal PC            | Multiple runs, ~1 day |
| Phase 4 (PPO Scratch)  | CPU → 3080 Ti  | Dev machine, then PC   | Variable              |
| Phase 5 (Polish)       | CPU only       | Any dev machine        | Writing/plotting time |
| Future A (Multi-level) | 3080 Ti+       | Personal PC or cloud   | 10M+ steps, hours     |
| Future B (Speedrun)    | A100/H100      | Cloud GPU rental       | Long training runs    |
| Future C (Full game)   | A100/H100      | Cloud GPU rental       | Multi-day training    |

**Notes on 3080 Ti for Phase 2:**
- Expect ~800–1500 FPS (frames/steps per second) with 8 envs, 84×84 grayscale obs
- 5M steps ≈ 1–2 hours wall time at 1000 FPS
- 10M steps ≈ 2–4 hours
- GPU memory usage will be modest (~2–3 GB) — CnnPolicy is small

**When to consider cloud GPU:**
- When you need 50M+ timestep runs for generalization experiments
- When you want to run many experiments in parallel
- Lambda Labs, Vast.ai, and RunPod offer A100s at $1–2/hr

---

## 8. Key Decisions Log

Track important decisions here as the project evolves.

| Date       | Decision                                        | Rationale                                    |
|------------|-------------------------------------------------|----------------------------------------------|
| 2026-04-13 | Use SB3 as primary framework, not CleanRL       | Polish and reliability for resume showcase    |
| 2026-04-13 | Start with SIMPLE_MOVEMENT (7 actions)          | Standard for Mario RL. Expand later if needed |
| 2026-04-13 | No VecNormalize on rewards initially             | V1 reward clipping caused signal collapse     |
| 2026-04-13 | 8 parallel envs as default                      | Balance between speed and stability on 3080Ti |
| 2026-04-13 | Phase 4 (PPO scratch) is optional/time-based    | Learning exercise, not critical path          |
| 2026-04-13 | Move old code to legacy/, don't delete           | Reference for what was tried before           |

---

## 9. Quick Reference: Environment Details

**Super Mario Bros 1-1 Layout:**
- Start position: x ≈ 40
- First Goomba: x ≈ 200
- First pipe: x ≈ 368
- Second pipe: x ≈ 618
- Tall pipe: x ≈ 734
- First pit (gap): x ≈ 1110
- Bullet Bill area: x ≈ 1800
- Staircase: x ≈ 2840  
- Flagpole: x ≈ 3160

**Observation Space (after preprocessing):**
- Shape: `(4, 84, 84)` — 4 grayscale frames, 84×84 pixels
- Dtype: `uint8` (0–255) or `float32` (0.0–1.0 after normalization)

**Action Space (SIMPLE_MOVEMENT):**
- 0: NOOP
- 1: Right
- 2: Right + A (run right + jump)  
- 3: Right + B (run right + sprint)
- 4: Right + A + B (sprint + jump)
- 5: A (jump)
- 6: Left

**Info dict keys (from gym-super-mario-bros):**
- `x_pos`: Mario's horizontal pixel position
- `y_pos`: Mario's vertical pixel position
- `time`: In-game timer (counts down from 400)
- `life`: Lives remaining (starts at 2, index 0-based: 2 = 3 lives)
- `flag_get`: Boolean — True when Mario touches the flagpole
- `score`: In-game score
- `coins`: Coin count
- `stage`: Current stage number
- `world`: Current world number
- `status`: "small", "tall", or "fireball"

---

## 10. Session Checklist

At the start of each new session, do the following:
1. Re-read this file (`gameplan.md`) to restore project context
2. Check which phase we're currently in
3. Review the current branch (`git branch`)
4. Check for any work-in-progress from last session
5. Pick up where we left off

**Current Status**: Starting Phase 1 — Foundation & Environment Stack
