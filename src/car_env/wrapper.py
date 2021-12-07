from abc import abstractmethod
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .core import CarEnv


Wrappable = Union['CarEnv', 'Wrapper']


class Wrapper:
    def __init__(self, env: Wrappable):
        self.env = env

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> 'CarEnv':
        return self.env.unwrapped

    def set_curriculum_phase(self, phase):
        self.env.set_curriculum_phase(phase)

    def step(self, action_dict):
        return self.env.step(action_dict)

    def reset(self):
        return self.env.reset()

    @staticmethod
    def get_observation_space(curriculum, source_space):
        return source_space

    @staticmethod
    def get_action_space(curriculum, source_space):
        return source_space


class ObservtionWrapper(Wrapper):
    def reset(self):
        observations = super().reset()
        return self.__transform(observations)

    def step(self, action_dict):
        observations, rewards, dones, infos = super().step(action_dict)
        return self.__transform(observations), rewards, dones, infos

    def __transform(self, observations):
        return {agent_id: self.observation(agent_id, obs)
                for agent_id, obs in observations.items()}

    @abstractmethod
    def observation(self, agent_id, obs):
        raise NotImplementedError()


class RewardWrapper(Wrapper):
    def step(self, action_dict):
        observations, rewards, dones, infos = self.env.step(action_dict)
        return observations, self.__transform(rewards, infos), dones, infos

    def __transform(self, rewards, infos):
        return {agent_id: self.reward(agent_id, r, infos.get('agent_id'))
                for agent_id, r in rewards.items()}

    @abstractmethod
    def reward(self, agent_id, reward, info):
        raise NotImplementedError()


class ActionWrapper(Wrapper):
    def step(self, action_dict):
        return self.env.step(self.__transform(action_dict))

    def __transform(self, action_dict):
        return {agent_id: self.reward(agent_id, r)
                for agent_id, r in action_dict.items()}

    @abstractmethod
    def action(self, action):
        raise NotImplementedError()
