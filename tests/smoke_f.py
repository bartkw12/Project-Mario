"""Smoke Test F: SubprocVecEnv on Windows.

Verifies that make_vec_env(use_subproc=True) works with nes-py's C extension
on Windows. Requires if __name__ == '__main__' guard for subprocess spawning.

Fallback: if this crashes, DummyVecEnv is acceptable for Phase 1.
"""

from src.config import Config
from src.envs import make_vec_env


def main():
    cfg = Config.from_yaml("configs/default.yaml")

    print(f"Creating SubprocVecEnv with {cfg.env.num_envs} envs...")
    vec_env = make_vec_env(cfg, use_subproc=True)

    print(f"Vec env type: {type(vec_env)}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Reset and check shape
    obs = vec_env.reset()
    print(f"\nAfter reset:")
    print(f"  obs.shape: {obs.shape}")
    print(f"  obs.dtype: {obs.dtype}")
    print(f"  obs min/max: {obs.min()}/{obs.max()}")

    # Run 10 steps
    for i in range(10):
        actions = [vec_env.action_space.sample() for _ in range(cfg.env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)

    print(f"\nAfter 10 steps:")
    print(f"  obs.shape: {obs.shape}")
    print(f"  obs.dtype: {obs.dtype}")
    print(f"  rewards.shape: {rewards.shape}")
    print(f"  dones.shape: {dones.shape}")

    vec_env.close()

    # Verify shape matches DummyVecEnv result
    expected = (cfg.env.num_envs, cfg.env.frame_stack, cfg.env.obs_size, cfg.env.obs_size)
    actual = obs.shape
    assert actual == expected, f"FAIL: expected {expected}, got {actual}"
    print(f"\nSmoke Test F: PASSED  (SubprocVecEnv obs shape: {actual})")


if __name__ == "__main__":
    main()
