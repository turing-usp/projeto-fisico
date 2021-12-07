import numpy as np
from car_env import RewardWrapper, ObservtionWrapper, Info
from car_env.wrapper import Wrappable


class CheckpointReward(RewardWrapper):
    def __init__(self, env: Wrappable, max_reward: float, min_velocity: float, max_velocity: float):
        super().__init__(env)
        self.max_reward = max_reward
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    def reward(self, agent_id: int, reward: float, info: Info) -> float:
        vel_mag = info["velocity"].dot(info["forward_vector"])
        if vel_mag > self.min_velocity:
            return reward + self.max_reward * min(1, vel_mag/self.max_velocity)
        else:
            return reward


class VelocityReward(RewardWrapper):
    def __init__(self, env: Wrappable,
                 coeff_per_second: float, warmup_time: float, min_velocity: float, max_velocity: float):
        super().__init__(env)
        self.coeff_per_second = coeff_per_second
        self.warmup_time = warmup_time
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        self.moving_weight = 0

    def reward(self, agent_id: int, reward: float, info: Info) -> float:
        vel_mag = info["velocity"].dot(info["forward_vector"])
        vel_mag = min(vel_mag, self.max_velocity)
        if vel_mag > self.min_velocity:
            step = info["time_passed"] / self.warmup_time
            weight_interval = (0, 1) if vel_mag > 0 else (-1, 0)
            self.moving_weight = np.clip(self.moving_weight, *weight_interval)
            self.moving_weight = np.clip(self.moving_weight + step, *weight_interval)
            return reward + self.moving_weight * self.coeff_per_second * vel_mag * info["time_passed"]
        else:
            return reward


class DeathPenalty(RewardWrapper):
    def __init__(self, env: Wrappable, penalty: float):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, agent_id: int, reward: float, info: Info) -> float:
        return reward-self.penalty*info["deaths"]


class BrakePenalty(RewardWrapper):
    def __init__(self, env: Wrappable, coeff_per_second: float):
        super().__init__(env)
        self.coeff_per_second = coeff_per_second

    def reward(self, agent_id: int, reward: float, info: Info) -> float:
        if info["action_brake"]:
            return reward-self.coeff_per_second * info["time_passed"] * abs(info["action_accelerator"])
        else:
            return reward


class HitIndicatorRemover(ObservtionWrapper):
    def observation(self, agent_id, obs):
        # Even indices indicate whether the ray hit something
        # Odd indices indicate the normalized distance for each ray (or the max if no object has hit)
        # Keep only the odd indices
        return obs[1::2]
