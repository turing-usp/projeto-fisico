"""Define as classes bases relacionadas aos wrappers do :class:`CarEnv`."""

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import inspect
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID


if TYPE_CHECKING:
    from .core import CarEnv, Info, CurriculumPhase, FloatArray


class Wrapper(MultiAgentEnv):
    """Um wrapper do :class:`CarEnv`.

    Args:
        env: o ambiente base desse wrapper.

    Attributes:
        env: o ambiente base desse wrapper.

    Note:
        Qualquer atributo do(s) ambiente(s) base pode ser acessado também pelo wrapper.
    """

    env: Union['CarEnv', 'Wrapper']

    def __init__(self, env: Union['CarEnv', 'Wrapper']) -> None:
        self.env = env
        self.unwrapped.wrappers.append(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> CarEnv:
        """Ver :func:`CarEnv.unwrapped`."""
        return self.env.unwrapped

    def set_curriculum_phase(self, phase: int) -> None:
        """Ver :func:`CarEnv.set_curriculum_phase`."""
        self.env.set_curriculum_phase(phase)

    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        """Ver :func:`CarEnv.step`."""
        return self.env.step(action_dict)

    def reset(self) -> Dict[AgentID, FloatArray]:
        """Ver :func:`CarEnv.reset`."""
        return self.env.reset()

    def set_param(self, key: str, val: Any) -> bool:
        """Atualiza um parâmetro desse wrapper.

        Args:
            key: o nome do parâmetro.
            val: o valor do parâmetro

        Raises:
            RuntimeError: em caso de erro de implementação do wrapper.

        Returns:
            ``True`` se o parâmetro existe, ``False`` caso contrário.
        """
        init_sig = inspect.signature(self.__class__.__init__)
        if key in init_sig.parameters:
            if not hasattr(self, key):
                raise RuntimeError(f"Object {self} has a parameter {key} but no attribute {key}")
            setattr(self, key, val)
            return True
        else:
            return False

    @staticmethod
    def get_observation_space(curriculum: List[CurriculumPhase], source_space: gym.Space) -> gym.Space:
        """Retorna o espaço de observações desse wrapper.

        Args:
            curriculum: conforme especificado em :class:`EnvConfig`.
            source_space: o espaço de observações do ambiente base.

        Returns:
            O espaço de observações do wrapper.
        """
        return source_space

    @staticmethod
    def get_action_space(curriculum: List[CurriculumPhase], source_space: gym.Space) -> gym.Space:
        """Retorna o espaço de ações desse wrapper.

        Args:
            curriculum: conforme especificado em :class:`EnvConfig`.
            source_space: o espaço de ações do ambiente base.

        Returns:
            O espaço de ações do wrapper.
        """
        return source_space

    def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space = None,
                          act_space: gym.Space = None) -> MultiAgentEnv:
        """Ver :func:`MultiAgentEnv.with_agent_groups`."""
        return self.unwrapped.with_agent_groups(groups, obs_space, act_space)


class ObservationWrapper(Wrapper):
    """Classe base para wrappers que modificam as observações geradas pelo ambiente.

    Para derivar essa classe, é necessário definir o método
    :func:`~ObservationWrapper.observation`. Caso o wrapper modifique
    o espaço de observações, é também necessário sobrescrever
    :func:`~Wrapper.get_observation_space`.
    """

    def reset(self):
        observations = super().reset()
        return self.__transform(observations)

    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        """Realiza um step no ambiente.

        Essa função modifica as observações utilizando :func:`ObservationWrapper.observation`.

        Para mais informações, ver :func:`CarEnv.step`.
        """
        observations, rewards, dones, infos = super().step(action_dict)
        return self.__transform(observations), rewards, dones, infos

    def __transform(self, observations: Dict[AgentID, FloatArray]) -> Dict[AgentID, FloatArray]:
        return {agent_id: self.observation(agent_id, obs)
                for agent_id, obs in observations.items()}

    @abstractmethod
    def observation(self, agent_id: AgentID, obs: FloatArray) -> FloatArray:
        """Transforma a observação de um único agente.

        Args:
            agent_id: O id do agente.
            obs: A observação.

        Returns:
            A observação transformada.
        """
        raise NotImplementedError()


class RewardWrapper(Wrapper):
    """Classe base para wrappers que modificam as recompensas geradas pelo ambiente.

    Para derivar essa classe, é necessário definir o método
    :func:`~RewardWrapper.reward`."""

    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        """Realiza um step no ambiente.

        Essa função modifica as recompensas utilizando :func:`RewardWrapper.reward`.

        Para mais informações, ver :func:`CarEnv.step`.
        """
        observations, rewards, dones, infos = self.env.step(action_dict)
        return observations, self.__transform(rewards, infos), dones, infos

    def __transform(self, rewards: Dict[AgentID, float], infos: Dict[AgentID, Info]) -> Dict[AgentID, float]:
        return {agent_id: self.reward(agent_id, r, infos[agent_id])
                for agent_id, r in rewards.items()}

    @abstractmethod
    def reward(self, agent_id: AgentID, reward: float, info: Info) -> float:
        """Transforma a recompesa de um único agente.

        Args:
            agent_id: O id do agente.
            reward: A recompensa.
            info: Um dicionário com informações adicionais relativas ao agente.

        Returns:
            A recompensa transformada.
        """
        raise NotImplementedError()


class ActionWrapper(Wrapper):
    """Classe base para wrappers que modificam as ações passadas ao ambiente.

    Para derivar essa classe, é necessário definir o método
    :func:`~ActionWrapper.action`. Caso o wrapper modifique
    o espaço de ações, é também necessário sobrescrever
    :func:`~Wrapper.get_action_space`.
    """

    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        """Realiza um step no ambiente.

        Essa função modifica as ações utilizando :func:`ActionWrapper.action`.

        Para mais informações, ver :func:`CarEnv.step`.
        """
        return self.env.step(self.__transform(action_dict))

    def __transform(self, action_dict: Dict[AgentID, FloatArray]) -> Dict[AgentID, FloatArray]:
        return {agent_id: self.reward(agent_id, r)
                for agent_id, r in action_dict.items()}

    @abstractmethod
    def action(self, agent_id: AgentID, action: FloatArray) -> FloatArray:
        """Transforma a ação de um único agente.

        Args:
            agent_id: O id do agente.
            action: A ação.

        Returns:
            A ação transformada.
        """
        raise NotImplementedError()
