from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import inspect
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID


if TYPE_CHECKING:
    from .core import CarEnv, FloatNDArray, Info, Curriculum


Wrappable = Union['CarEnv', 'Wrapper']


class Wrapper(MultiAgentEnv):
    env: Wrappable

    def __init__(self, env: Wrappable) -> None:
        self.env = env
        self.unwrapped.wrappers.append(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> CarEnv:
        return self.env.unwrapped

    def set_curriculum_phase(self, phase: int) -> None:
        self.env.set_curriculum_phase(phase)

    def step(self, action_dict: Dict[AgentID, FloatNDArray]) \
            -> Tuple[Dict[AgentID, FloatNDArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        return self.env.step(action_dict)

    def reset(self) -> Dict[AgentID, FloatNDArray]:
        return self.env.reset()

    def set_param(self, key: str, val: Any) -> bool:
        init_sig = inspect.signature(self.__class__.__init__)
        if key in init_sig.parameters:
            if not hasattr(self, key):
                raise RuntimeError(f"Object {self} has a parameter {key} but no attribute {key}")
            setattr(self, key, val)
            return True
        else:
            return False

    @staticmethod
    def get_observation_space(curriculum: Curriculum, source_space: gym.Space) -> gym.Space:
        return source_space

    @staticmethod
    def get_action_space(curriculum: Curriculum, source_space: gym.Space) -> gym.Space:
        return source_space

    def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space = None,
                          act_space: gym.Space = None) -> "MultiAgentEnv":
        return self.unwrapped.with_agent_groups(groups, obs_space, act_space)


class ObservtionWrapper(Wrapper):
    def reset(self):
        observations = super().reset()
        return self.__transform(observations)

    def step(self, action_dict: Dict[AgentID, FloatNDArray]) \
            -> Tuple[Dict[AgentID, FloatNDArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        observations, rewards, dones, infos = super().step(action_dict)
        return self.__transform(observations), rewards, dones, infos

    def __transform(self, observations: Dict[AgentID, FloatNDArray]) -> Dict[AgentID, FloatNDArray]:
        return {agent_id: self.observation(agent_id, obs)
                for agent_id, obs in observations.items()}

    @abstractmethod
    def observation(self, agent_id: AgentID, obs: FloatNDArray) -> FloatNDArray:
        raise NotImplementedError()


class RewardWrapper(Wrapper):
    def step(self, action_dict: Dict[AgentID, FloatNDArray]) \
            -> Tuple[Dict[AgentID, FloatNDArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        observations, rewards, dones, infos = self.env.step(action_dict)
        return observations, self.__transform(rewards, infos), dones, infos

    def __transform(self, rewards: Dict[AgentID, float], infos: Dict[AgentID, Info]) -> Dict[AgentID, float]:
        return {agent_id: self.reward(agent_id, r, infos[agent_id])
                for agent_id, r in rewards.items()}

    @abstractmethod
    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
        raise NotImplementedError()


class ActionWrapper(Wrapper):
    def step(self, action_dict: Dict[AgentID, FloatNDArray]) \
            -> Tuple[Dict[AgentID, FloatNDArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        return self.env.step(self.__transform(action_dict))

    def __transform(self, action_dict: Dict[AgentID, FloatNDArray]) -> Dict[AgentID, FloatNDArray]:
        return {agent_id: self.reward(agent_id, r)
                for agent_id, r in action_dict.items()}

    @abstractmethod
    def action(self, agent_id: AgentID, action: FloatNDArray) -> FloatNDArray:
        raise NotImplementedError()
