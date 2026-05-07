Please remember these are just suggestions.



3rd party review of Project-Mario so far (post ablation O)



Blunt summary: Do not blow up the recipe now.

Your next gains are far more likely to come from better checkpoint exploitation and gentler fine-tuning than from another large conceptual change.



**What the latest results say**

1\) The winning family is now clear

The journal shows a very consistent storyline:



* Moderate reward shaping in L beat both the gentle D-family and the aggressive J-family, reaching 30% at 9.0M..md) 
* Lowering the learning rate in M improved that to 36%..md)
* Lowering it again in N produced a breakthrough to 64%..md) 
* Lowering it one more time in O pushed the project champion to 74% at 14.5M. 



That means the “compounding LR halving” strategy is not a fluke anymore — it is your most productive optimization axis so far. The journal explicitly records the gain progression as +6% → +28% → +10% across successive halvings.



**2) The remaining problem is no longer “learning”**

Your best O checkpoints already show:



* very high mean\_x\_pos (e.g. 2923 at 14.5M),
* consistent max\_x\_pos = 3161,
* strong mean\_reward,
* and a broad golden window from roughly 14.0M–15.0M, where multiple checkpoints are above 60%..md) 



So the bottleneck is not “the agent doesn’t know how to beat the level.”

The bottleneck is:



* capturing the very best checkpoint precisely, and
* preventing the post-peak erosion after \~15.3M..md) 



That is a much better problem to have.



**3) target\_kl=0.05 is probably good enough — don’t touch it**

The journal already concluded that:



* 0.015 throttled learning too hard in H,
* 0.05 was correctly calibrated in I/J,
* and the KL cap is still doing useful work in the later runs..md)



By the time you get to O, the dominant failure mode is no longer catastrophic overshoot, but cumulative entropy erosion and late-run decline. That means the KL cap is not your main problem anymore. I would leave target\_kl=0.05 alone. 



**What I think about Ablation O’s hyperparameters**

Your latest ablation\_o.yaml is actually quite good as a late-stage fine-tuning config:



* lr: 3.125e-5
* target\_kl: 0.05
* forward\_scale: 0.3
* flag\_bonus: 100
* time\_penalty: -0.05
* ent\_coef: 0.05 → 0.03
* num\_envs: 16
* n\_steps: 1024
* batch\_size: 512
* n\_epochs: 4 



I would not change the reward shaping right now. The journal already shows that:



* D’s reward structure was too gentle and hit a ceiling,
* J/G’s stronger shaping was too aggressive and brittle,
* L’s middle-ground shaping was the first real sweet spot,
* and O/N/M all improved on top of that same moderate-reward base. 



So: keep the L→M→N→O reward structure.



**The biggest strategic issue now**

Your checkpoint spacing is too coarse for how close you are

This is the single most important practical issue I see.

In O, your best checkpoint is 14.5M = 74%, but the surrounding checkpoints are:



* 14.0M = 60%
* 15.0M = 66%
* 15.5M = 0%.md) 



That means your true optimum may be sitting somewhere between 14.3M and 14.8M, and your current 500K checkpoint spacing is now too coarse for a run that is already close to the target.



**My conclusion**

If you launch another 3M-step run with the same coarse checkpoint intervals, you risk missing the actual winning checkpoint again.

This is the first thing I would fix.



**So where should you go next?**



**Option 1 (highest EV): checkpoint-densification run around the O golden window**

If your goal is to get to 80% as efficiently as possible, this is my top recommendation.

What to do conceptually

Take the current best family (O/N/L) and do a short continuation / micro-run around the winning region, with:



* no reward changes
* no target\_kl changes
* no environment/action changes
* same or slightly lower LR
* much denser checkpointing (think in terms of much smaller intervals than 500K)



Why

Because the journal shows the policy quality is changing materially within about 500K–1M steps near the end of training. In O, you went from 74% at 14.5M to 0% at 15.5M in about 1M steps. That means the “golden window” is now narrow enough that checkpoint granularity is a first-order concern..md)

My recommendation

This is the most likely path to 80% without a big architectural change.



**Option 2: one more LR halving, but do it surgically**

This is my second-choice recommendation.

The journal strongly supports that repeated LR halving is helping:



* L → M: 30% → 36%
* M → N: 36% → 64%
* N → O: 64% → 74% 



So yes — another halving is a reasonable idea.

But here is the important nuance:

I would not automatically do:



* another full +3M extension,
* with the same sparse checkpoint cadence,
* and just hope the next best checkpoint happens to land on a save boundary.



If you try another LR halve, I would do it:



* from the best-performing checkpoint family (O 14.5M),
* over a shorter extension window,
* with much denser checkpointing,
* because you only need +6 points, not another reinvention of the run.



**Option 3: stop-mode / active preservation**

This is my third choice.

The journal already shows that:



* the CollapseDetector fires meaningfully,
* later runs can partially recover,
* but the best checkpoint is still often earlier than the final model..md) 



That means an active “preserve peak” mode could help.

My take

This is reasonable, but I would still put it after checkpoint densification, because:



* dense checkpointing is simpler,
* more robust,
* and less sensitive to choosing the wrong stop trigger.



You are close enough that better checkpoint capture is probably more valuable than another layer of training control logic.



**What I would not do next**

1\) Don’t revisit D/J/G-family reward shaping

You’ve already learned this lesson:



* D-family is too gentle and seems capped,.md) 
* J/G-family is too aggressive and too brittle,.md) 
* L-family is the current sweet spot.



I would not spend the next run re-opening that question.



2\) Don’t touch target\_kl

It’s doing its job. The journal already shows that the current issue is gradual erosion despite bounded KL, not catastrophic overshoot. That means target\_kl=0.05 is not the first knob I’d move.



3\) Don’t change action space or env setup

You are too close to 80% for a sideways move like RIGHT\_ONLY or other structural changes. Those are earlier-phase explorations, not endgame optimizations.



**One important thing people forget at this stage**

**74% on 50 episodes is close, but still noisy**



That means O 14.5M is only 3 more successes out of 50 away from your target.

Why this matters

Before you assume you need a major new training innovation, I would make sure you understand the variance of your evaluation:



* rerun the same best checkpoint multiple times,
* or evaluate it over a larger total number of stochastic episodes,
* so you know whether 74% is a stable estimate or whether this checkpoint is already flirting with 80%.



This does not mean cherry-pick a lucky 50-episode batch.

It means: get a better estimate of the actual policy consistency of O 14.5M.

That is a very cheap and very smart next move.



**My concrete recommendation**

If I were you, I would do this next:

Step 1 — Re-evaluate O 14.5M more thoroughly

Before more training, run the current champion more extensively than one 50-episode batch, so you understand whether 74% is solid or whether you’re already basically at the boundary. Your journal already relies on stochastic 50-episode sweeps as the ground truth..md) 



Step 2 — Do a short, densely checkpointed continuation from O 14.5M

This is my highest-confidence recommendation.



* same reward shaping
* same target\_kl
* same rollout structure
* same or slightly lower LR
* dense checkpoints
* treat the goal as finding the best checkpoint, not “training to the end”



Step 3 — Only if that fails, consider one more LR-halving extension

Because the compounding LR pattern is still your strongest lever, but now the question is precision, not large-scale exploration.





