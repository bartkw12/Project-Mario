"""Custom SB3 callbacks for tracking Mario-specific metrics during training.

MarioMetricsCallback — logs mean_x_pos, flag_capture_rate, and max_x_pos
to TensorBoard at episode boundaries.

ProgressBarCallback — tqdm progress bar with live Mario metrics and ETA.
"""

from collections import deque

import numpy as np
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class MarioMetricsCallback(BaseCallback):
    """Track Mario-specific metrics and log them to TensorBoard.

    At each step, checks which vectorized envs have finished an episode
    (dones[i] == True) and records x_pos and flag_get from the info dict.
    Maintains a rolling window of the last `window` episodes for averaging.

    Logged keys:
        mario/mean_x_pos        — rolling mean of final x_pos at episode end
        mario/max_x_pos         — best x_pos in the rolling window
        mario/flag_capture_rate — fraction of rolling window episodes with flag_get
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._window = window
        self._x_positions: deque[float] = deque(maxlen=window)
        self._flag_gets: deque[bool] = deque(maxlen=window)

    def _on_training_start(self) -> None:
        self._x_positions.clear()
        self._flag_gets.clear()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                self._x_positions.append(info.get("x_pos", 0))
                self._flag_gets.append(bool(info.get("flag_get", False)))

        if self._x_positions:
            self.logger.record("mario/mean_x_pos", np.mean(self._x_positions))
            self.logger.record("mario/max_x_pos", np.max(self._x_positions))
            self.logger.record(
                "mario/flag_capture_rate",
                np.mean(self._flag_gets),
            )

        return True


class ProgressBarCallback(BaseCallback):
    """tqdm progress bar showing timesteps, ETA, FPS, and Mario metrics.

    Reads episode data from a companion MarioMetricsCallback instance
    to display mean_x_pos and flag_capture_rate in the progress bar.
    """

    def __init__(self, total_timesteps: int, mario_cb: MarioMetricsCallback | None = None):
        super().__init__(verbose=0)
        self._total = total_timesteps
        self._mario_cb = mario_cb
        self._pbar: tqdm | None = None

    def _on_training_start(self) -> None:
        # Account for resumed training: start the bar at the current timestep
        initial = self.model.num_timesteps
        self._pbar = tqdm(
            total=self._total,
            initial=initial,
            unit="step",
            unit_scale=True,
            desc="Training",
            dynamic_ncols=True,
        )

    def _on_step(self) -> bool:
        if self._pbar is not None:
            # Update by n_envs steps each call
            self._pbar.update(self.training_env.num_envs)

            # Show live metrics from the companion callback
            postfix = {}
            if self._mario_cb and self._mario_cb._x_positions:
                postfix["x_pos"] = f"{np.mean(self._mario_cb._x_positions):.0f}"
                postfix["flag%"] = f"{np.mean(self._mario_cb._flag_gets):.1%}"
            self._pbar.set_postfix(postfix, refresh=False)

        # Hard stop when we've reached the target timesteps
        if self.num_timesteps >= self._total:
            return False

        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.close()


class EntropyScheduleCallback(BaseCallback):
    """Linearly decay model.ent_coef from start to end over training.

    SB3 does not support a callable for ent_coef (only lr and clip_range),
    so this callback mutates model.ent_coef directly at each step.

    Uses an explicit total_timesteps rather than model._total_timesteps,
    which SB3 inflates on resume (existing + requested steps). This ensures
    the schedule tracks the config's intended training length correctly.
    """

    def __init__(self, start: float, end: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self._start = start
        self._end = end
        self._total = total_timesteps

    def _on_step(self) -> bool:
        progress = min(self.model.num_timesteps / self._total, 1.0)
        self.model.ent_coef = self._start + (self._end - self._start) * progress
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True
