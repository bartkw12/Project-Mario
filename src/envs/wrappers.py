"""Custom Gymnasium-compatible wrappers for Super Mario Bros.

SkipFrame — Repeat each action for N frames, accumulating reward.
SimpleRewardShaping — Add shaped rewards based on x_pos progress, death, and flag capture.
"""

import gymnasium


class SkipFrame(gymnasium.Wrapper):
    """Repeat the same action for `skip` consecutive frames and sum the rewards.

    Reduces the effective decision frequency (e.g., skip=4 means the agent
    decides every 4th frame) while keeping the emulator running at full speed.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class SimpleRewardShaping(gymnasium.Wrapper):
    """Shape the reward signal based on forward progress, death, and flag capture.

    This is a plain gymnasium.Wrapper (NOT RewardWrapper) because the shaping
    logic needs access to info dict and internal state (_last_x_pos).

    Params:
        forward_scale: multiplier on x_pos delta per step (positive = reward rightward movement)
        death_penalty: reward added when Mario loses a life (should be negative)
        flag_bonus: reward added when Mario captures the flag (should be positive)
    """

    def __init__(self, env, forward_scale=0.1, death_penalty=-15.0, flag_bonus=50.0, time_penalty=0.0):
        super().__init__(env)
        self._forward_scale = forward_scale
        self._death_penalty = death_penalty
        self._flag_bonus = flag_bonus
        self._time_penalty = time_penalty
        self._last_x_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_x_pos = info.get("x_pos", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Forward progress
        x_pos = info.get("x_pos", self._last_x_pos)
        delta_x = x_pos - self._last_x_pos
        reward += self._forward_scale * delta_x

        # Time penalty (per-step cost incentivizes faster play)
        reward += self._time_penalty

        # Death penalty
        if info.get("life", 2) < 2:
            reward += self._death_penalty

        # Flag capture bonus
        if info.get("flag_get", False):
            reward += self._flag_bonus

        self._last_x_pos = x_pos
        return obs, reward, terminated, truncated, info
