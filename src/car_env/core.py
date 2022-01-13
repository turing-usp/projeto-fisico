"""Define o :class:`CarEnv` e outros componentes importantes."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override
from mlagents_envs.environment import UnityEnvironment
from ray.rllib.utils.typing import AgentID

from .config_side_channel import ConfigSideChannel
from .metrics_side_channel import MetricsSideChannel
from .schedulers import Scheduler
from .wrapper import Wrapper
from . import actors


def _flatten(obj: dict, d: Dict[str, Any] = None, prefix: str = '') -> Dict[str, Any]:
    if d is None:
        d = {}
    for key, val in obj.items():
        full_key = prefix + str(key)
        if isinstance(val, dict):
            _flatten(val, d=d, prefix=full_key + '/')
        else:
            d[full_key] = val
    return d


def _satisfies_constraint(flat_obj: Dict[str, Any], key: str, min_value: Union[float, int]) -> bool:
    if key not in flat_obj:
        print(f'Warning: missing constraint key "{key}" (in constraint {key} >= {min_value})')
        return False
    return flat_obj[key] >= min_value


def satisfies_constraints(obj: dict, constraints: Dict[str, Union[float, int]]) -> bool:
    """Verifica se um objeto satisfaz um conjunto de restrições.

    O dicionário de restrições segue o formato::

        constraints = {
            key1: val1,
            key2: val2,
            ...
        }

    e corresponde à expressão ``obj[key1] >= val1 and obj[key2] >= val2 and ...``.

    Args:
        obj: o objeto de interesse.
        constraints: o dicionário de restrições.

    Returns:
        Um bool indicando se ``obj`` satisfaz todas as ``constraints``.
    """
    flat_obj = _flatten(obj)
    return all(_satisfies_constraint(flat_obj, key, min_value)
               for key, min_value in constraints.items())


FloatArray = npt.NDArray[np.float64]


class Info(TypedDict):
    """Informações adicionais obtidas sobre cada gente após um step.

    Corresponde à variável ``info`` em
    ``obs, reward, done, info = env.step()`` (seguindo a convenção do :mod:`gym`).

    Attributes:
        time_passed: O tempo, em segundos, desde a última ação.
        new_checkpoints: O número de checkpoints pelo qual o agente passou
            desde a última ação (geralmente ``0`` ou ``1``).

        forward_vector: Um vetor unitário ``np.array([x, y, z])`` indicando a direção
            e sentido do carro, ou seja, um vetor unitário apontando da traseira para
            a frente do carro.
        velocity: A velocidade do carro (``np.array([vx, vy, vz])``),
            expressa em unidades por segundo.
        angular_velocity: A velocidade angular do carro (``np.array([wx, wy, wz])``),
            expressa em radianos por segundo.

        action_accelerator: O valor do acelerador na última ação tomada por esse agente
            (``action[0]``).
        action_steer: O valor do volante na última ação tomada por esse agente
            (``action[1]``).
        action_brake: O valor do freio na última ação tomada por esse agente
            (``action[2] <= 0``).

        deaths: O número de mortes desse agente desde a última ação
            (geralmente ``0`` ou ``1``).

    Note:
        O formato da ação passada ao agente (``action_{accelerator,steer,brake}``) é
        descrito em :func:`CarEnv.step`.
    """
    time_passed: float
    new_checkpoints: int

    forward_vector: FloatArray
    velocity: FloatArray
    angular_velocity: FloatArray

    action_accelerator: float
    action_steer: float
    action_brake: bool

    deaths: int


class CurriculumPhase(TypedDict, total=False):
    """Configurações para uma fase do currículo de treinamento.

    Attributes:
        when: Condições que devem ser satisfeitas para que essa fase do
            currículo seja iniciada. Para mais informações sobre esse
            atributo, ver a documentação da função :func:`satisfies_constraints`.
            Valor padrão: ``{}`` (nenhuma condição).
        for_iterations: O número de iterações de treinamento durante as quais
            as condições especificadas no atributo ``when`` devem ser satisfeitas.
            Valor padrão: ``1``.
        unity_config: As configurações do unity que devem ser modificadas nessa fase.
            Para mais informações sobre esse argumento, ver a documentação do
            atributo :attr:`CarEnv.unity_config`.
            Valor padrão: ``{}`` (nenhuma modificação).
        wrappers: As configurações dos wrappers que devem ser modificadas nessa fase.
            Para cada wrapper, o elemento ``wrappers[wrapper_name]`` é um dicionário
            com as configurações a serem modificadas.
            Valor padrão: ``{}`` (nenhuma modificação).
    """
    when: Dict[str, Union[float, int]]
    for_iterations: int
    unity_config: Dict[str, Union[ConfigSideChannel.Value, Scheduler]]
    wrappers: Dict[str, Dict[str, Any]]


class EnvConfig(TypedDict, total=False):
    """Configurações do :class:`CarEnv`.

    Attributes:
        file_name: Nome (ou caminho) do binário do Unity.
            Se ``None``, tenta se conectar a um editor do Unity.
            Valor padrão: ``None``.
        wrappers: Lista de wrappers que devem ser aplicados ao ambiente.
            Os wrappers serão aplicados na ordem em que aparecem na lista, a partir
            do índice zero.
            Valor padrão: ``[]``.
        curriculum: o currículo a ser utilizado durante o treinamento. Cada elemento
            dessa lista corresponde a uma fase do treinamento.
            Valor padrão:  ``[]``.
    """
    file_name: Optional[str]
    wrappers: List[Type[Wrapper]]
    curriculum: List[CurriculumPhase]


class CarEnv(Unity3DEnv):
    """Um wrapper do CarEnv em Python.

    O CarEnv é um ambiente de simulação de carro implementado com o Unity.
    A interface fornecida em python utiliza o rllib.

    Args:
        file_name: nome (ou caminho) do binário do Unity.
            Se ``None``, tenta se conectar a um editor do Unity.
            Valor padrão: ``None``.
        curriculum: o currículo a ser utilizado durante
            o treinamento.
            Valor padrão:
        *args: argumentos adicionais a serem passados para o :class:`Unity3DEnv`.
        **kwargs: argumentos adicionais a serem passados para o :class:`Unity3DEnv`.

    Attributes:
        curriculum: O currículo a ser utilizado durante o treinamento. Cada elemento
            dessa lista corresponde a uma fase do treinamento.
        unity_config: As configurações do unity na fase atual.
        last_actions: A última a ação tomada por cada agente.
        wrappers: Os wrappers que foram aplicados a esse ambiente.
    """

    _config_side_channel: ConfigSideChannel
    _metrics_side_channel: MetricsSideChannel

    curriculum: List[CurriculumPhase]

    unity_config: Dict[str, Union[ConfigSideChannel.Value, Scheduler]]
    last_actions: Dict[AgentID, FloatArray]
    _schedulers: List[Scheduler]
    _iters_satisfying_curriculum: int

    wrappers: List[Wrapper]

    def __init__(self,
                 *args,
                 file_name: str = None,
                 curriculum: List[CurriculumPhase] = [],
                 **kwargs) -> None:

        self._config_side_channel = ConfigSideChannel()
        self._metrics_side_channel = MetricsSideChannel()

        # Monkey patch to make rllib's Unity3DEnv instantiate the UnityEnvironment with a SideChannel
        original_init = UnityEnvironment.__init__
        try:
            def new_init(inner_self, *args, **kwargs):
                side_channels = kwargs.pop('side_channels', [])
                side_channels.extend(
                    [self._config_side_channel, self._metrics_side_channel])
                original_init(inner_self, *args, **kwargs,
                              side_channels=side_channels)
            UnityEnvironment.__init__ = new_init  # type: ignore
            super().__init__(*args,
                             episode_horizon=float('inf'),  # type: ignore
                             file_name=file_name,
                             no_graphics=file_name is None,
                             **kwargs)
        finally:
            UnityEnvironment.__init__ = original_init

        self.curriculum = curriculum

        self.unity_config = {}
        self.last_actions = {}
        self._schedulers = []
        self._iters_satisfying_curriculum = 0

        self.wrappers = []

    @property
    def unwrapped(self) -> CarEnv:
        """Retorna o ambiente original (sem wrappers)."""
        return self

    def set_curriculum_phase(self, phase: int) -> None:
        """Troca a fase do currículo de treinamento.

        Args:
            phase: a fase desejada, conforme especificado em :func:`CarEnv.curriculum`.

        Raises:
            RuntimeError: se algum dos parâmetros de wrappers especificados em
                ``self.curriculum[phase]`` não existir.
        """
        self.phase = phase
        logger = actors.param_logger()
        logger.update_param.remote('curriculum_phase', phase)

        counter = actors.agent_step_counter()
        counter.new_phase.remote()

        for k, v in self.curriculum[phase].get('unity_config', {}).items():
            old = self.unity_config.get(k)
            if isinstance(old, Scheduler):
                self._schedulers.remove(old)
            if isinstance(v, Scheduler):
                self._schedulers.append(v)
            self.unity_config[k] = v
            self._config_side_channel.set(k, v)

        for wrapper_name, wrapper_params in self.curriculum[phase].get('wrappers', {}).items():
            for param, val in wrapper_params.items():
                success = False
                for w in self.wrappers:
                    if w.set_param(param, val):
                        success = True
                        break
                if not success:
                    raise RuntimeError(f"Parameter {param} not found in any of the wrappers")
                logger.update_param.remote(f'{wrapper_name}/{param}', val)

    def set_curriculum_phase_from_rllib_result(self, result: dict) -> None:
        """Troca a fase do currículo com base nos resultados de um batch de treinamento.

        Geralmente chamado pela função :func:`car_env.DefaultCallbacks.on_train_result`

        Args:
            result: dicionário criado pelo rllib com os resultados de um batch
                de treinamento.

        Raises:
            RuntimeError: se algum dos parâmetros de wrappers especificados em
                ``self.curriculum[phase]`` não existir.
        """
        next_phase = self.phase+1
        if next_phase < len(self.curriculum):
            if satisfies_constraints(result, self.curriculum[next_phase].get('when', [])):
                self._iters_satisfying_curriculum += 1
            else:
                self._iters_satisfying_curriculum = 0

            min_iters = self.curriculum[next_phase].get('for_iterations', 1)
            if self._iters_satisfying_curriculum >= min_iters:
                self.set_curriculum_phase(next_phase)
                return

        for sch in self._schedulers:
            sch.step_to(result['agent_steps_this_phase'])

    @override(Unity3DEnv)
    def step(self, action_dict: Dict[AgentID, FloatArray]) \
            -> Tuple[Dict[AgentID, FloatArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:
        """Realiza um step no ambiente.

        Args:
            action_dict: O dicionário de ações para cada agente que solicitou
                uma ação no step anterior. Cada ação é um array
                ``np.array([accelerator, steer, brake])`` indicando:

                    :accelerator: a aceleração do carro
                        (um valor entre ``0`` e ``1``, inclusivo)
                    :steer: a posição do volante do carro
                        (um valor entre ``-1`` (esquerda) e ``1`` (direita), inclusivo)
                    :brake: o freio do carro
                        (o freio é ativado se ``brake <= 0``)

        Returns:
            Uma tupla de dicionários ``(obs, rewards, dones, infos)``.
            Os agentes presentes nesses dicionários são aqueles que precisam
            de ações no próximo step.
        """

        self.last_actions.update(action_dict)
        raw_obs, rewards, dones, _ = super().step(action_dict)

        counter = actors.agent_step_counter()
        counter.add_agent_steps.remote(len(action_dict))

        obs = self._get_obs(raw_obs)
        infos = self._get_infos(raw_obs, rewards)
        rewards = {agent_id: 0.0 for agent_id in rewards}

        return obs, rewards, dones, infos

    @override(Unity3DEnv)
    def reset(self) -> Dict[AgentID, FloatArray]:
        """Reseta todos o ambiente (incluindo todos os agentes).

        Returns:
            O estado de cada um dos agentes que precisa de uma ação.
        """
        if not self.last_actions:
            # first iteration => initialize the curriculum
            self.set_curriculum_phase(0)

        raw_obs = super().reset()
        return self._get_obs(raw_obs)

    def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space = None,
                          act_space: gym.Space = None) -> MultiAgentEnv:
        """Ver :func:`MultiAgentEnv.with_agent_groups`."""
        return self.unwrapped.with_agent_groups(groups, obs_space, act_space)

    @staticmethod
    def get_observation_space(curriculum: List[CurriculumPhase], wrappers: List[Wrapper] = []) -> gym.Space:
        """Retorna o espaço de observações para o ambiente especificado.

        Args:
            curriculum: conforme especificado em :class:`EnvConfig`.
            wrappers: conforme especificado em :class:`EnvConfig`.

        Returns:
            O espaço de observações.
        """

        config = curriculum[0].get('unity_config', {}) if len(curriculum) >= 1 else {}
        rays_per_direction = config.get('AgentRaysPerDirection', 3)
        assert isinstance(rays_per_direction, int)
        rays = 2*rays_per_direction + 1
        space = spaces.Box(0, 1, (2*rays,), dtype=np.float32)
        for w in wrappers:
            space = w.get_observation_space(curriculum, space)
        return space

    @staticmethod
    def get_action_space(curriculum: List[CurriculumPhase], wrappers: List[Wrapper] = []) -> gym.Space:
        """Retorna o espaço de ações para o ambiente especificado.

        Args:
            curriculum: conforme especificado em :class:`EnvConfig`.
            wrappers: conforme especificado em :class:`EnvConfig`.

        Returns:
            O espaço de ações.
        """

        space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        for w in wrappers:
            space = w.get_observation_space(curriculum, space)
        return space

    @staticmethod
    def get_policy(curriculum: List[CurriculumPhase], wrappers: List[Wrapper] = []) \
            -> Tuple[Optional[Type], gym.Space, gym.Space, Dict]:
        """Retorna a política do rllib que descreve o ambiente especificado.

        Args:
            curriculum: conforme especificado em :class:`EnvConfig`.
            wrappers: conforme especificado em :class:`EnvConfig`.

        Returns:
            A política do rllib que descreve o ambiente especificado.
        """

        obs_space = CarEnv.get_observation_space(curriculum, wrappers)
        action_space = CarEnv.get_action_space(curriculum, wrappers)
        return (None, obs_space, action_space, {})

    @staticmethod
    def _get_obs(raw_obs: Dict[AgentID, Tuple[FloatArray, FloatArray]]) -> Dict[AgentID, FloatArray]:
        """Cria o dicionário de observações a partir dos valores retornados pelo Unity.

        Args:
            raw_obs: as observações *cruas* retornadas pelo Unity.

        Returns:
            O dicionário de observações para cada agente especificado.
        """
        return {agent_id: s[0]
                for agent_id, s in raw_obs.items()}

    def _get_infos(self, raw_obs: Dict[AgentID, Tuple[FloatArray, FloatArray]], rewards: Dict[AgentID, float]) \
            -> Dict[AgentID, Info]:
        """Cria o dicionário de infos a partir dos valores retornados pelo Unity.

        Args:
            raw_obs: as observações *cruas* (i.e. sem passar pela função :func:`CarEnv._get_obs`)
                retornadas pelo Unity.
            rewards: as recompensas cruas retornadas pelo Unity.

        Returns:
            O dicionário de infos para cada agente especificado.
        """
        return {
            agent_id: Info(
                time_passed=s[1][0],
                new_checkpoints=s[1][1],
                forward_vector=s[1][2:5],
                velocity=s[1][5:8],
                angular_velocity=s[1][8:11],
                action_accelerator=self.last_actions[agent_id][0],
                action_steer=self.last_actions[agent_id][1],
                action_brake=self.last_actions[agent_id][2] <= 0,
                deaths=-int(rewards[agent_id]),
            )
            for agent_id, s in raw_obs.items()
        }

    @staticmethod
    def from_config(config: EnvConfig) -> Union[CarEnv, Wrapper]:
        """Cria um novo :class:`CarEnv` a partir das configurações dadas.

        Args:
            config: As configurações do :class:`CarEnv`.

        Returns:
            O ambiente, possivelmente com wrappers.
        """
        config = config.copy()
        wrappers = config.pop('wrappers', [])

        env = CarEnv(**config)
        wrapper_configs = config.get('curriculum', [{}])[0].get('wrappers', {})
        for wrapper_type in wrappers:
            env = wrapper_type(env, **wrapper_configs.get(wrapper_type.__name__, {}))
        return env


tune.register_env("car_env", CarEnv.from_config)
