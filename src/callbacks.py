"""Custom SB3 callbacks for tracking Mario-specific metrics during training.

MarioMetricsCallback — logs mean_x_pos, flag_capture_rate, and max_x_pos
to TensorBoard at episode boundaries.
"""

from collections import deque

import numpy as np
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
