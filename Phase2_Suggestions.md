Remember these are just suggested Improvements and you get the final say in what goes and what stays for the project!



**1) The plan forgets the SB3 “frequency must account for n\_envs” rule**

This is the most important technical omission.

SB3 explicitly warns that callback frequencies like:



* save\_freq for CheckpointCallback, and
* eval\_freq for EvalCallback



are based on callback calls / env.step() calls, and with vectorized environments each call effectively corresponds to n\_envs timesteps. SB3 specifically recommends adjusting these frequencies using something like max(freq // n\_envs, 1) when you want comparable scheduling across different numbers of envs. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io]

Why this matters here

Your plan says:



* save every 500K steps, and
* evaluate every cfg.eval.eval\_freq.



But unless Claude explicitly adjusts those for n\_envs=8, your checkpoints/evals will happen 8× less often in terms of callback calls than intended. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io]

My verdict

This is not a small nitpick — it should be corrected in the implementation plan.



**2) Best-model selection is still misaligned with your real success criterion**

Your project’s success criterion in gameplan.md is ≥80% flag capture over 50 eval episodes. \[pypi.org]

But EvalCallback selects the “best model” using mean evaluation reward, not flag capture rate. SB3 documents EvalCallback as saving the best model according to mean reward on the evaluation environment. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io]

Why that matters

If your reward shaping is:



* forward progress,
* death penalty,
* flag bonus,



then mean reward is only a proxy for the thing you actually care about. A model that gets far but inconsistently finishes could potentially outrank a model that finishes more reliably depending on the reward scale. That is exactly why your custom Mario metrics (mean\_x\_pos, flag\_capture\_rate) are valuable. \[pypi.org]

My recommendation

I would not block the plan on this, but I would make Claude explicitly state:



“Best model saved by EvalCallback is best by mean reward, while project success is judged separately by flag capture rate and custom evaluation metrics.” \[stable-bas...thedocs.io], \[stable-bas...thedocs.io], \[pypi.org]



That single sentence would make the plan more intellectually honest.



**3) The plan still treats milestone benchmarks too confidently**

Step 6 says:



* monitor TensorBoard for 500K → x\_pos 300–500,
* 5M → 2000–3160+, etc.



Those numbers are inherited from your current gameplan.md, but you already know they are provisional heuristics, not validated expectations. Your own stable plan currently contains them as rough progress benchmarks, but they should not be treated like pass/fail acceptance thresholds yet. \[pypi.org]

My recommendation

Claude’s plan should explicitly reframe those as:



* heuristics
* or sanity-check targets
* not implementation truth.



This is a planning calibration issue, not a reason to reject the whole plan.\\



**4) The CUDA installation command is too risky / too specific**

Step 6 says:

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu124



I would not trust that line as written.

PyTorch 2.11 release notes explicitly say that:



* default PyPI wheels now use CUDA 13.0,
* and if you need an older CUDA line, the documented examples mention index URLs like cu126 or cu128.



I do not currently have evidence that cu124 is the right index for your target install path. So I would tell Claude:



“Do not hardcode a CUDA wheel URL in the Phase 2 plan; say to use the official PyTorch install selector / validated command for the target machine.”



This is a real issue because it could cause a bogus install failure on your GPU machine.



**5) DummyVecEnv as default is okay for first baseline stability, but the full 5M run should not assume it is the final backend**

You previously decided to prefer DummyVecEnv first on Windows for stability, and I think that was the right answer. SB3 explicitly warns that SubprocVecEnv on Windows requires the usual multiprocessing safeguards and can introduce extra complexity. \[stable-bas...thedocs.io]

However, Claude’s Step 2/Step 6 wording risks turning that temporary safety choice into a de facto long-run assumption:



* 8 envs,
* DummyVecEnv,
* full 5M baseline.



My issue

That is okay for the first short baseline, but before a multi-million-step run you should at least benchmark:



* DummyVecEnv
* vs. SubprocVecEnv (if stable)



because DummyVecEnv gives you vectorization but not multiprocessing, and long Mario runs can be environment-step bound. SB3’s vec-env docs explain the distinction clearly: DummyVecEnv is vectorization without multiprocessing; SubprocVecEnv adds multiprocessing. \[stable-bas...thedocs.io]

My recommendation

I would add:

“Use DummyVecEnv for the initial baseline implementation and short validation runs; benchmark against SubprocVecEnv before committing the full 5M run.” \[stable-bas...thedocs.io]



**Specific step-by-step comments**



**Step 1 — MarioMetricsCallback**

Good idea, but the verification is too weak.

Import testing only proves the file imports. It does not prove:



* \_on\_step() sees infos and dones,
* episode boundary logic works with VecEnv auto-reset semantics,
* or TensorBoard keys actually appear.



SB3’s callback docs emphasize that callbacks are called after each env.step() and that num\_timesteps grows by n\_envs in vectorized training. SB3’s vec-env docs also remind you that when a vec env returns done=True, the returned observation is already the next episode’s first observation, and the real final observation is exposed via terminal\_observation in infos. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io]

My take

The callback idea is fine, but verification should be:



* short training smoke test + check logger output not just import.



**Step 2 — training loop + VecMonitor**

Strong step overall.

I like:



* PPO("CnnPolicy", ...),
* hyperparams from config,
* TensorBoard,
* callback list.



The main thing missing is the frequency adjustment issue I already mentioned. Also, since your reward shaping still modifies rewards, remember that ep\_rew\_mean will reflect shaped reward, not “original game reward,” which is acceptable as long as you are explicit about it. SB3’s evaluation helper warns that wrappers affecting rewards will affect evaluation metrics too. \[stable-bas...thedocs.io], \[deepwiki.com]



**Step 3 — periodic checkpoints**

Good, but must account for n\_envs=8. SB3 explicitly warns about this for CheckpointCallback. \[stable-bas...thedocs.io]



**Step 4 — EvalCallback**

Good design, but same issue: eval\_freq must account for n\_envs. SB3 explicitly warns about this too. \[stable-bas...thedocs.io]

Also good that the plan uses a 1-env eval vec env with the same wrapper stack, which is exactly what SB3 recommends conceptually.



**Step 5 — update evaluate.py**

Yes, this should absolutely be part of Phase 2. Your own gameplan.md Phase 2 already expects trained-model evaluation and video capture. \[pypi.org]

The only thing I’d question is adding --model to config.py rather than keeping it as an evaluation-only CLI concern. That’s not wrong, but it can blur training CLI and evaluation CLI concerns if your shared config parser gets too broad.

My recommendation

This is minor:



* acceptable,
* but keep the CLI separation clean.



**Step 6 — full 5M run**

This is fine as a final execution step, but the plan should explicitly say:



* this comes after shorter validation runs,
* the throughput estimate is empirical,
* and the x-pos milestones are heuristic.



**What I would tell Claude to revise**

If you want to improve the plan before implementation, I’d tell Claude to make these edits:

**Must-fix**



* Adjust save\_freq and eval\_freq for n\_envs using SB3’s documented guidance. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io]
* Explicitly state that best-model saving is by mean reward, not by flag capture rate, and that final project success is judged separately. \[stable-bas...thedocs.io], \[stable-bas...thedocs.io], \[pypi.org]
* Replace the hardcoded CUDA cu124 install line with “use the official PyTorch install selector / validated GPU install command.”
* Soften the x-pos progression table into heuristic / sanity-check language. \[pypi.org]



**Strongly recommended**



* Clarify that DummyVecEnv is the initial default, not necessarily the final backend for a full long run. \[stable-bas...thedocs.io]
* Make Step 1 verification more meaningful than an import-only check.





Bottom line

Is this a good Phase 2 plan?

Yes — overall yes. It is structured, incremental, and much better than most plans I see. It respects your project scope and uses the SB3 callback system in the intended way.











































































































































































