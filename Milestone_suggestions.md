**Milestone Breakdown**



I’d structure the project in 5 milestones. This gives you something that is realistic to finish, easy to talk about in interviews, and still ambitious.



**Milestone 1 — Modernize and Stabilize the Environment Stack**

**Goal**

Get a fully working, reproducible training/evaluation pipeline using a modern RL stack and a Mario environment that behaves correctly under a Gymnasium-style workflow. 

Gymnasium is the maintained successor to Gym, and SB3 2.8.0 is current and Gymnasium-native, while gym-super-mario-bros is still widely used but old enough 

that compatibility handling is often necessary.



**Deliverables**



* Clean repo structure (configs/, src/, results/, videos/, notebooks/)
* Seeded runs and deterministic-ish settings where possible
* Environment wrapper stack documented clearly
* Short sanity-check video showing random policy / scripted policy / evaluation loop works
* README section on environment compatibility and why Mario needs legacy glue despite using modern tooling. Gymnasium’s migration guide and Shimmy docs are especially relevant here.



**Resume signal**

This milestone tells people you can do ML systems integration, not just model training. That matters a lot in RL because environment correctness is often half the battle.



**Milestone 2 — Establish a Strong PPO Baseline on a Single Stage**

**Goal**

Train a PPO baseline that can reliably solve 1-1 (or another chosen stage) with proper logging, checkpoints, and repeated evaluation. PPO is a sensible baseline here because it is 

widely used, well-documented, and supported by both SB3 and CleanRL. CleanRL in particular presents PPO as one of the most popular DRL algorithms and provides transparent single-file 

reference implementations.



**Deliverables**



* Baseline PPO training script
* TensorBoard logging and/or W\&B tracking
* Saved checkpoints and best-model evaluation
* Evaluation script that runs multiple seeds and records videos
* Plots for:

  * episodic return
  * episode length
  * distance/progress
  * success rate over training
  * FPS / SPS (steps per second), which CleanRL also logs as a meaningful practical metric.



**Resume signal**

This shows you can create a reliable benchmark, not just run one lucky training job. RL Zoo’s design strongly reinforces that tuning, evaluation, and video recording are first-class parts of a serious workflow.



**Milestone 3 — Learn the Algorithm Deeply: Implement PPO from Scratch (or CleanRL-style)**

**Goal**

Implement your own compact PPO training loop in PyTorch after you have the SB3 baseline working. CleanRL explicitly argues that its single-file implementations are ideal for understanding 

every algorithm detail and for prototyping advanced features without fighting a large abstraction layer.



**Deliverables**



* A small standalone PPO implementation (or a heavily adapted CleanRL-inspired version)
* Clear explanation of:

  * clipped surrogate objective
  * GAE
  * entropy bonus
  * actor/critic loss
  * rollout collection
  * minibatch updates
  * vectorized environments. CleanRL’s PPO docs are a strong reference here.
* Benchmark comparison: your PPO vs SB3 PPO

  * not necessarily equal performance, but reasonable learning behavior and similar trends



**Resume signal**

This is what makes the project far more valuable than “I used PPO from a library.” It shows you understand both applied RL engineering and the algorithmic guts. That combination is very powerful for interviews.





**Milestone 4 — Go Beyond Memorization: Curriculum + Generalization**

**Goal**

Scale from one stage to multiple stages and show evidence of transfer or generalization. The Mario environment supports full-game and individual-stage variants, and OpenAI Retro specifically 

frames generalization across levels as a core RL research challenge.

Good ways to do this

* Curriculum learning: train on 1-1 → 1-2 → 1-3 → 1-4
* Multi-stage training: sample from a subset of stages
* Held-out evaluation: train on some stages, evaluate on unseen ones
* Random stage mode: if feasible, use random stages to test robustness. The environment docs mention random-stage support as part of the design.



**Deliverables**



* One curriculum experiment
* One generalization experiment
* Videos of success and failure modes
* Short analysis of where policies break:

  * pipes
  * gaps
  * enemies
  * timing-sensitive jumps
  * stage-specific memorization vs actual transferable behavior



**Resume signal**

This is the part that moves the project from game demo to real RL project. Generalization is a research-worthy topic; solving one fixed stage is much less impressive on its own.





**Milestone 5 — Turn It Into a Research-Style Case Study**

**Goal**

Package the project like a serious experiment:

* reproducible config files,
* ablation studies,
* training curves,
* videos,
* limitations,
* and lessons learned. RL Zoo and W\&B both emphasize experiment tracking and comparison as core practice.



**Deliverables**



* Strong README with:

  * problem statement
  * environment details
  * algorithm overview
  * experimental design
  * results
  * failure analysis
  * future work
* A short report or notebook with ablations:

  * reward shaping vs no shaping
  * curriculum vs no curriculum
  * action space choice (RIGHT\_ONLY / SIMPLE\_MOVEMENT / maybe COMPLEX\_MOVEMENT)
  * entropy coefficient or rollout length sensitivity
* Final highlight video montage



**Resume signal**

This is what makes recruiters or lab leads think: “this person knows how to run experiments and tell the story properly.”













































































































































































