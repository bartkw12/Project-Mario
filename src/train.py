"""Training entry point for Project Mario.

Usage:
    python -m src.train --dry-run              # Phase 1: verify env stack
    python -m src.train                        # Phase 2: run PPO training
    python -m src.train --resume path/to/model.zip  # resume from checkpoint
    python -m src.train --config path.yaml     # custom config
    python -m src.train --seed 123             # override seed
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.callbacks import EntropyScheduleCallback, MarioMetricsCallback, ProgressBarCallback
from src.config import Config, parse_args, set_global_seed
from src.envs import make_vec_env


def dry_run(cfg: Config) -> None:
    """Create vectorized env, run 100 steps, print obs shape, exit."""
    print(f"[dry-run] Creating {cfg.env.num_envs} envs...")
    vec_env = make_vec_env(cfg, use_subproc=False)

    obs = vec_env.reset()
    print(f"[dry-run] After reset: obs.shape = {obs.shape}")

    for i in range(100):
        actions = [vec_env.action_space.sample() for _ in range(cfg.env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)

    print(f"[dry-run] After 100 steps:")
    print(f"  obs.shape = {obs.shape}")
    print(f"  obs.dtype = {obs.dtype}")
    print(f"  rewards sample = {rewards[:3]}")

    vec_env.close()

    expected = (cfg.env.num_envs, cfg.env.frame_stack, cfg.env.obs_size, cfg.env.obs_size)
    assert obs.shape == expected, f"Shape mismatch: {obs.shape} != {expected}"
    print(f"\n[dry-run] PASSED — obs shape {obs.shape} matches expected {expected}")


def train(cfg: Config, resume_path: str | None = None, use_subproc: bool = False, name: str = "default") -> None:
    """Run PPO training with SB3."""
    base = Path("results") / name
    vec_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    print(f"[train] Experiment '{name}' — outputs → {base}/")
    print(f"[train] Creating {cfg.env.num_envs} training envs ({vec_type})...")
    env = make_vec_env(cfg, use_subproc=use_subproc)

    if resume_path:
        print(f"[train] Resuming from {resume_path}...")
        model = PPO.load(
            resume_path,
            env=env,
            tensorboard_log=str(base / "logs"),
            device=cfg.device,
        )
    else:
        print(f"[train] Initialising PPO (CnnPolicy, device={cfg.device})...")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=cfg.training.lr,
            n_steps=cfg.training.n_steps,
            batch_size=cfg.training.batch_size,
            n_epochs=cfg.training.n_epochs,
            gamma=cfg.training.gamma,
            gae_lambda=cfg.training.gae_lambda,
            clip_range=cfg.training.clip_range,
            ent_coef=cfg.training.ent_coef,
            tensorboard_log=str(base / "logs"),
            device=cfg.device,
            verbose=0,
        )

    # Checkpoint every 500K timesteps (adjusted for n_envs)
    checkpoint_freq = max(500_000 // cfg.env.num_envs, 1)
    checkpoint_dir = base / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_mario",
    )

    mario_cb = MarioMetricsCallback()
    progress_cb = ProgressBarCallback(cfg.training.total_timesteps, mario_cb=mario_cb)

    callbacks = [mario_cb, checkpoint_cb, progress_cb]

    # Optional entropy schedule
    if cfg.training.ent_coef_final is not None:
        ent_cb = EntropyScheduleCallback(cfg.training.ent_coef, cfg.training.ent_coef_final)
        callbacks.append(ent_cb)
        print(f"[train] Entropy schedule: {cfg.training.ent_coef} → {cfg.training.ent_coef_final} (linear)")
    else:
        print(f"[train] Entropy coefficient: {cfg.training.ent_coef} (static)")

    # Eval callback: 1-env eval with same wrapper stack, saves best model by mean reward.
    # Note: best model is selected by mean eval reward (proxy); project success
    # (>=80% flag capture) is judged separately via evaluate.py.
    eval_freq = max(cfg.eval.eval_freq // cfg.env.num_envs, 1)
    best_model_dir = base / "models" / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = base / "logs" / "eval"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_env = make_vec_env(cfg, use_subproc=False, num_envs=1)
    eval_cb = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=cfg.eval.deterministic,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        verbose=1,
    )
    callbacks.append(eval_cb)

    print(f"[train] Starting training for {cfg.training.total_timesteps:,} timesteps...")
    print(f"[train] Checkpoints every ~500K timesteps (save_freq={checkpoint_freq} calls, n_envs={cfg.env.num_envs})")
    print(f"[train] Eval every ~{cfg.eval.eval_freq:,} timesteps (eval_freq={eval_freq} calls, {cfg.eval.n_eval_episodes} episodes)")
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=resume_path is None,
    )

    save_path = base / "models" / "final_model"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[train] Final model saved to {save_path}.zip")

    eval_env.close()
    env.close()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.seed is not None:
        cfg.seed = args.seed

    set_global_seed(cfg.seed)

    if args.dry_run:
        dry_run(cfg)
    else:
        train(cfg, resume_path=args.resume, use_subproc=args.subproc, name=args.name)


if __name__ == "__main__":
    main()
