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


class EntropyCollapseDetector(BaseCallback):
    """Monitor entropy velocity and KL divergence for collapse detection.

    Intercepts SB3's logger.dump to capture ``train/entropy_loss`` and
    ``train/approx_kl`` after each PPO update.  Computes rolling diagnostics
    and logs them to TensorBoard under the ``collapse/`` prefix.

    In diagnostic mode (``stop=False``), only logs metrics and prints
    warnings.  In active mode (``stop=True``), halts training when sustained
    entropy collapse is detected.

    Logged keys:
        collapse/entropy             — entropy_loss from the PPO update
        collapse/entropy_velocity    — per-update change in entropy
        collapse/entropy_vel_rolling — rolling velocity over the full window
        collapse/approx_kl           — approx_kl from the PPO update
        collapse/kl_clip_frac        — fraction of recent updates exceeding target_kl
        collapse/flag_rate           — current flag capture rate from MarioMetricsCallback
    """

    def __init__(
        self,
        mario_cb: MarioMetricsCallback,
        window: int = 10,
        stop: bool = False,
        entropy_velocity_threshold: float = -0.05,
        patience: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._mario_cb = mario_cb
        self._window = window
        self._stop = stop
        self._ent_vel_threshold = entropy_velocity_threshold
        self._patience = patience

        self._entropy_history: deque[float] = deque(maxlen=window)
        self._kl_history: deque[float] = deque(maxlen=window)
        self._consecutive_warnings: int = 0
        self._original_dump = None
        self._should_stop = False

    def _on_training_start(self) -> None:
        self._entropy_history.clear()
        self._kl_history.clear()
        self._consecutive_warnings = 0
        self._should_stop = False

        # Intercept logger.dump to capture metrics before SB3 clears them.
        original_dump = self.logger.dump
        self._original_dump = original_dump
        detector = self  # avoid shadowing in closure

        def _capturing_dump(step=0):
            ent = detector.logger.name_to_value.get("train/entropy_loss")
            kl = detector.logger.name_to_value.get("train/approx_kl")
            if ent is not None:
                detector._entropy_history.append(ent)
            if kl is not None:
                detector._kl_history.append(kl)
            detector._log_diagnostics()
            return original_dump(step)

        self.logger.dump = _capturing_dump

    def _log_diagnostics(self) -> None:
        if not self._entropy_history:
            return

        entropy = self._entropy_history[-1]
        self.logger.record("collapse/entropy", entropy)

        if self._kl_history:
            self.logger.record("collapse/approx_kl", self._kl_history[-1])

        # Per-update velocity
        if len(self._entropy_history) >= 2:
            velocity = self._entropy_history[-1] - self._entropy_history[-2]
            self.logger.record("collapse/entropy_velocity", velocity)

        # Rolling velocity over full window
        rolling_vel = None
        if len(self._entropy_history) >= self._window:
            rolling_vel = (
                self._entropy_history[-1] - self._entropy_history[0]
            ) / (self._window - 1)
            self.logger.record("collapse/entropy_vel_rolling", rolling_vel)

        # KL clip fraction (only meaningful with target_kl)
        target_kl = getattr(self.model, "target_kl", None)
        if target_kl and self._kl_history:
            kl_clip_frac = sum(
                1 for kl in self._kl_history if kl > target_kl
            ) / len(self._kl_history)
            self.logger.record("collapse/kl_clip_frac", kl_clip_frac)

        # Flag rate from companion callback
        if self._mario_cb._flag_gets:
            self.logger.record(
                "collapse/flag_rate", np.mean(self._mario_cb._flag_gets)
            )

        # Collapse warning
        if rolling_vel is not None and rolling_vel < self._ent_vel_threshold:
            self._consecutive_warnings += 1
            parts = [
                f"entropy_vel={rolling_vel:.4f}/update",
                f"entropy={entropy:.4f}",
            ]
            if self._mario_cb._flag_gets:
                parts.append(
                    f"flag_rate={np.mean(self._mario_cb._flag_gets):.1%}"
                )
            print(
                f"[CollapseDetector] WARNING #{self._consecutive_warnings}: "
                + ", ".join(parts)
                + f" @ {self.num_timesteps:,} steps"
            )

            if self._stop and self._consecutive_warnings >= self._patience:
                print(
                    f"[CollapseDetector] STOPPING: "
                    f"{self._consecutive_warnings} consecutive warnings "
                    f"(patience={self._patience})"
                )
                self._should_stop = True
        elif rolling_vel is not None:
            self._consecutive_warnings = 0

    def _on_step(self) -> bool:
        return not self._should_stop

    def _on_training_end(self) -> None:
        if self._original_dump is not None:
            self.logger.dump = self._original_dump
