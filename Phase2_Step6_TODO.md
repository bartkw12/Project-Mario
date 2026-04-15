# Phase 2 — Step 6: Full 5M Baseline Run (3080 Ti)

## Setup

- [ ] Pull latest from GitHub
  ```powershell
  git pull
  ```

- [ ] Create venv and install project dependencies
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -e .
  ```

- [ ] Check your NVIDIA driver is working
  ```powershell
  nvidia-smi
  ```
  Note the CUDA version in the top-right corner (this is the max your driver supports).

- [ ] Replace CPU torch with CUDA torch
  Go to https://pytorch.org/get-started/locally/ and select your OS + CUDA version to get the exact command. It will look like:
  ```powershell
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu<YOUR_VERSION>
  ```
  This overwrites the CPU torch installed by `pip install -e .`

- [ ] Verify GPU is available
  ```powershell
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  Expected: `True NVIDIA GeForce RTX 3080 Ti`

## Training

- [ ] Run the full 5M-step baseline
  ```powershell
  python -m src.train --config configs/default.yaml
  ```
  Expected runtime: ~1–4 hours at ~800–1500 FPS.

  This will:
  - Train PPO with CnnPolicy, 8 parallel envs
  - Log to TensorBoard at `results/logs/`
  - Save checkpoints every ~500K steps to `results/models/checkpoints/`
  - Save best model (by mean eval reward) to `results/models/best_model/best_model.zip`
  - Save final model to `results/models/final_model.zip`

## Monitor (optional, second terminal)

- [ ] Launch TensorBoard
  ```powershell
  .\.venv\Scripts\Activate.ps1
  tensorboard --logdir results/logs
  ```
  Open http://localhost:6006 in browser.

  **Sanity-check heuristics** (rough guides, not pass/fail):
  | Timesteps | mean_x_pos (approx) | What to expect |
  |-----------|--------------------|-|
  | 500K      | ~300–500           | Runs right, may clear first pipe |
  | 1M        | ~500–800           | Clearing pipes consistently |
  | 2M        | ~800–1200          | Approaching first gap |
  | 5M        | ~2000–3160+        | Late-level or flag |

## Evaluate

- [ ] Run 50-episode evaluation on best model
  ```powershell
  python -m src.evaluate --model results/models/best_model/best_model.zip --episodes 50
  ```

- [ ] Check the summary output for `flag_capture_rate`
  **Phase 2 success criterion: ≥80% flag capture rate over 50 episodes**

- [ ] (Optional) Record a video of the trained agent
  ```powershell
  python -m src.evaluate --model results/models/best_model/best_model.zip --episodes 5 --record
  ```
  Videos saved to `results/videos/`

## Notes

- **Do NOT install CUDA separately** — the CUDA runtime is bundled in the PyTorch wheel. You just need an up-to-date NVIDIA driver.
- `device: "auto"` in `configs/default.yaml` will automatically pick GPU if available.
- If FPS is below ~800, consider benchmarking SubprocVecEnv by temporarily editing `train.py` to pass `use_subproc=True` to `make_vec_env()`.
- Best model is saved by mean eval reward (proxy). The real success metric is flag capture rate from `evaluate.py`.
