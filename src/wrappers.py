from typing import Callable, Dict, Tuple, TypeVar, Union
import numpy as np
from ray.rllib.utils.typing import AgentID
import gym
import gym.spaces

from car_env import actors
from car_env.core import CarEnv, Info, FloatArray
from car_env.wrapper import Wrapper, RewardWrapper, ObservationWrapper


T = TypeVar('T')


def lazy_value(init: Callable[[], T]) -> Callable[[], T]:
    def getter() -> T:
        nonlocal val
        if val is None:
            val = init()
        return val
    val = None
    return getter


class CheckpointReward(RewardWrapper):
    max_reward: float
    min_velocity: float
    max_velocity: float

    def __init__(self, env: Union[CarEnv, Wrapper], max_reward: float,
                 min_velocity: float, max_velocity: float) -> None:
        super().__init__(env)
        self.max_reward = max_reward
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
        vel_mag = info["velocity"].dot(info["forward_vector"])
        if vel_mag > self.min_velocity:
            return reward + info["new_checkpoints"] * self.max_reward * min(1, vel_mag/self.max_velocity)
        else:
            return reward


class VelocityReward(RewardWrapper):
    coeff_per_second: float
    warmup_time: float
    min_velocity: float
    max_velocity: float

    def __init__(self, env: Union[CarEnv, Wrapper], coeff_per_second: float, warmup_time: float,
                 min_velocity: float, max_velocity: float) -> None:
        super().__init__(env)
        self.coeff_per_second = coeff_per_second
        self.warmup_time = warmup_time
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        self.moving_weight = 0

    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
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
    penalty: float

    def __init__(self, env: Union[CarEnv, Wrapper], penalty: float) -> None:
        super().__init__(env)
        self.penalty = penalty

    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
        return reward-self.penalty*info["deaths"]


class BrakePenalty(RewardWrapper):
    coeff_per_second: float

    def __init__(self, env: Union[CarEnv, Wrapper], coeff_per_second: float) -> None:
        super().__init__(env)
        self.coeff_per_second = coeff_per_second

    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
        if info["action_brake"]:
            return reward-self.coeff_per_second * info["time_passed"] * abs(info["action_accelerator"])
        else:
            return reward


class HitIndicatorRemover(ObservtionWrapper):
    def observation(self, agent_id: AgentID, obs: FloatArray) -> FloatArray:
        # Even indices indicate whether the ray hit something
        # Odd indices indicate the normalized distance for each ray (or the max if no object has hit)
        # Keep only the odd indices
        return obs[1::2]

    @staticmethod
    def get_observation_space(_, source_space: gym.Space) -> gym.Space:
        assert isinstance(source_space, gym.spaces.Box)
        return gym.spaces.Box(low=source_space.low,  # type: ignore
                              high=source_space.high,  # type: ignore
                              shape=(source_space.shape[0]//2,))  # type: ignore


class RewardLogger(Wrapper):
    total_rewards: Dict[AgentID, float]

    def __init__(self, env: Union[CarEnv, Wrapper]) -> None:
        super().__init__(env)
        self.total_rewards = {}

    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        observations, rewards, dones, infos = super().step(action_dict)

        tracker = lazy_value(lambda: actors.agent_metric_tracker())
        for agent_id, r in rewards.items():
            self.total_rewards[agent_id] = self.total_rewards.get(agent_id, 0) + r
            deaths = infos[agent_id]["deaths"]
            if deaths != 0:
                for _ in range(deaths):
                    tracker().add_metric.remote('agent_reward', self.total_rewards[agent_id]/deaths)
                self.total_rewards[agent_id] = 0
        return observations, rewards, dones, infos
