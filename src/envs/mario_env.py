"""Environment factory functions for Super Mario Bros.

Two creation paths:
  make_single_env  — for debug, eval, and video recording (single Gymnasium env)
  make_vec_env     — for training (vectorized env with frame stacking)
"""

import gymnasium
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy import GymV21CompatibilityV0
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage

from src.config import Config
from src.envs.wrappers import SimpleRewardShaping, SkipFrame


def make_single_env(cfg: Config, seed: int | None = None, record_video: bool = False, video_dir: str | None = None, render_mode: str | None = None):
    """Create a single, fully-wrapped Gymnasium Mario env.

    Pipeline: mario → JoypadSpace → shimmy → SkipFrame → SimpleRewardShaping
              → ResizeObservation → GrayscaleObservation → (optional) RecordVideo

    Used for: debug, evaluation, video recording.
    NOT vectorized — returns a plain gymnasium.Env.
    """
    # Base env + discrete actions (old gym API)
    env = gym_super_mario_bros.make(cfg.env.game)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Bridge to Gymnasium 5-tuple API
    # render_mode="rgb_array" needed for video recording
    if render_mode is None and record_video:
        render_mode = "rgb_array"
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    # Custom wrappers
    env = SkipFrame(env, skip=cfg.env.frame_skip)
    env = SimpleRewardShaping(
        env,
        forward_scale=cfg.reward.forward_scale,
        death_penalty=cfg.reward.death_penalty,
        flag_bonus=cfg.reward.flag_bonus,
    )

    # Preprocessing — resize first (while still 3-channel) to avoid cv2
    # dropping the trailing channel dim, then grayscale to (H, W, 1)
    env = gymnasium.wrappers.ResizeObservation(env, shape=(cfg.env.obs_size, cfg.env.obs_size))
    env = gymnasium.wrappers.GrayscaleObservation(env, keep_dim=True)

    # Optional video recording (single-env path only)
    if record_video and video_dir:
        env = gymnasium.wrappers.RecordVideo(env, video_folder=video_dir)

    return env


def _make_env_thunk(cfg: Config, seed: int | None = None, render_mode: str | None = None):
    """Return a callable that creates a single env (for use with VecEnv)."""
    def _init():
        return make_single_env(cfg, seed=seed, render_mode=render_mode)
    return _init


def make_vec_env(cfg: Config, use_subproc: bool = False, num_envs: int | None = None, render_mode: str | None = None):
    """Create a vectorized env stack for training.

    Pipeline: num_envs × make_single_env → DummyVecEnv/SubprocVecEnv
              → VecMonitor → VecTransposeImage → VecFrameStack

    Args:
        num_envs: Override cfg.env.num_envs (e.g., 1 for eval).
        render_mode: Pass to underlying envs (e.g., "rgb_array" for video recording).
    """
    n = num_envs if num_envs is not None else cfg.env.num_envs
    env_fns = [
        _make_env_thunk(cfg, seed=cfg.seed + i if cfg.seed is not None else None, render_mode=render_mode)
        for i in range(n)
    ]

    if use_subproc:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Monitor for SB3 ep_rew_mean / ep_len_mean tracking
    vec_env = VecMonitor(vec_env)

    # Transpose (H, W, C) → (C, H, W) so frame stacking works on channel axis
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=cfg.env.frame_stack, channels_order="first")
    return vec_env
