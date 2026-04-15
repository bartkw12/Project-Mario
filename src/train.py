"""Training entry point for Project Mario.

Usage:
    python -m src.train --dry-run              # Phase 1: verify env stack
    python -m src.train                        # Phase 2: run PPO training
    python -m src.train --config path.yaml     # custom config
    python -m src.train --seed 123             # override seed
"""

from pathlib import Path

from stable_baselines3 import PPO

from src.callbacks import MarioMetricsCallback
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


def train(cfg: Config) -> None:
    """Run PPO training with SB3."""
    print(f"[train] Creating {cfg.env.num_envs} training envs (DummyVecEnv)...")
    env = make_vec_env(cfg, use_subproc=False)

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
        tensorboard_log=str(Path("results/logs")),
        device=cfg.device,
        verbose=1,
    )

    callbacks = [MarioMetricsCallback()]

    print(f"[train] Starting training for {cfg.training.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
    )

    save_path = Path("results/models/final_model")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"[train] Final model saved to {save_path}.zip")

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
        train(cfg)


if __name__ == "__main__":
    main()
