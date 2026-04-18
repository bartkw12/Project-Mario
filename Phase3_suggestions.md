Remember you get the final say on what is implemented and not, you have access to the code base and this reviewer does not. These are just suggested improvements!



REVIEW:



What the plan gets right (important)

1\. Correct ordering of interventions ✅

The sequence:



* Entropy
* Speed
* Action space
* Extended training



is exactly right given your Phase 2 diagnosis.

Your data shows:



* PPO can reach the flag
* PPO cannot sustain exploration
* PPO wastes time instead of dying



So attacking entropy first is non‑negotiable. Claude got this right.



**Where I’m critical (this matters)!**



**1. Entropy ablation is too shallow**

This is the biggest weakness of the plan.

What the plan assumes: “Increase ent\_coef from 0.01 → 0.02 → 0.03”



Why this may not be sufficient

Your Phase 2 collapse happened after \~5M steps, not early.

That strongly suggests:



* static entropy bonus is being overpowered by PPO’s clipped objective,
* not merely “too small”.



In long-horizon PPO, static entropy almost always collapses eventually unless:



* it is very large (hurts convergence), or
* it is scheduled or KL-controlled.



Risk

You may see:



* ablation A improves peak success (e.g. 10–15%)
* ablation B collapses later but still collapses
* you conclude “entropy didn’t fix it” when the real issue is entropy decay over time



Critical note

This plan does not include any entropy scheduling or KL‑based control, which are standard PPO stabilization techniques in long runs.



I am not saying “add them now” — but Phase 3 should acknowledge this risk explicitly.



What I would want added conceptually

At minimum, a note like:



“If static entropy fails to prevent late collapse, next step is entropy decay schedule or KL‑target control.”



Without that, the entropy section is a bit optimistic.



**2. Speed shaping jumps too aggressively**

You jump from:



* forward\_scale = 0.1 (Phase 2) → 0.5 → 1.0



That is a 5×–10× jump.

Why this is risky

Forward reward is applied every step. Increasing it too much can:



* drown out death penalty,
* encourage reckless movement,
* destabilize value estimates,
* bias policy toward “always move right” even when jumping is required.



Given that Mario already reaches mid‑level safely, you probably need:



* slightly more urgency, not brute force.



A more conservative progression (conceptually) would be:



* 0.2
* 0.3
* 0.5



Your plan skips the most likely “sweet spot”.

⚠️ Not wrong, but high variance.



**3. RIGHT\_ONLY action space is not a guaranteed win**

This is the most debatable design choice.

Why RIGHT\_ONLY can help



* Smaller action space
* Faster convergence
* Less dithering



Why RIGHT\_ONLY can backfire



* Some jumps require brief release of Right
* Mid-air control is subtly different
* Mario sometimes needs to slow down before jumps
* RIGHT\_ONLY removes NOOP and Left entirely



Given your data:



* Mario is too slow, not too indecisive
* He is not stuck oscillating



So the failure is not action indecision — it’s policy collapse + lack of urgency.

My take

RIGHT\_ONLY is reasonable as a late‑stage ablation, but the plan slightly oversells it as a convergence win.

It should be framed as:



“Potential simplification, not guaranteed improvement”



Right now the plan implies more confidence than warranted.





**4. Success criterion is rigid for Phase 3**



“Goal: ≥80% flag capture over 50 deterministic episodes.”



This is a Phase 4‑level bar.

For Phase 3 iteration, a more realistic framing would be:



* sustained non‑zero success (e.g. 20–40%)
* no entropy collapse over long horizons
* stability over time



Why this matters:



* PPO improvements often come in gradual plateaus
* jumping from 6% → 80% in one phase is ambitious



If Phase 3 reaches:



* 30–50% reliable success
* no late collapse
* faster traversal



That is already a huge win.

The plan risks labeling a strong improvement as “failure”.



**5. Extended training as a last resort is good — but expectations are off**

10M steps may help only if stability is fixed first.

If entropy still collapses:



* 10M will just collapse later and harder.



The plan implicitly assumes:



“If we just train longer with better hyperparams, it’ll solve.”



That is sometimes false in PPO.

I would want one explicit sentence:



“Extended training is only meaningful if entropy remains healthy beyond 5M.”





My honest recommendation

✅ Proceed with this plan with two mindset adjustments





1. Treat Phase 3 as stabilization, not “solve it or bust”
* Success = higher + sustained flag rate
* No late entropy collapse
* Faster traversal
* Not necessarily 80%







2\. Be ready to admit static entropy may not be enough

* If ablation A/B both collapse late:

  * that’s a result, not a failure	
  * it motivates Phase 3.5 (entropy schedule / KL target)











































