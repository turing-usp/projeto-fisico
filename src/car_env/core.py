from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from mlagents_envs.environment import UnityEnvironment
from ray.rllib.utils.typing import AgentID

from .config_side_channel import ConfigSideChannel
from .metrics_side_channel import MetricsSideChannel
from .schedulers import Scheduler
from .wrapper import Wrapper


def flatten(obj: dict, d: Dict[str, Any] = None, prefix: str = '') -> Dict[str, Any]:
    if d is None:
        d = {}
    for key, val in obj.items():
        full_key = prefix + str(key)
        if isinstance(val, dict):
            flatten(val, d=d, prefix=full_key + '/')
        else:
            d[full_key] = val
    return d


def satisfies_constraint(flat_obj: Dict[str, Any], key: str, min_value: Union[float, int]):
    if key not in flat_obj:
        print(f'Warning: missing constraint key "{key}" (in constraint {key} >= {min_value})')
        return False
    return flat_obj[key] >= min_value


def satisfies_constraints(obj: dict, constraints: "Contraints"):
    flat_obj = flatten(obj)
    return all(satisfies_constraint(flat_obj, key, min_value)
               for key, min_value in constraints.items())


FloatNDArray = npt.NDArray[np.float64]


class Info(TypedDict):
    time_passed: float
    new_checkpoints: int

    forward_vector: FloatNDArray
    velocity: FloatNDArray
    angular_velocity: FloatNDArray

    action_accelerator: float
    action_steer: float
    action_brake: bool

    deaths: int


Contraints = Dict[str, Union[float, int]]
UnityConfig = Dict[str, Union[ConfigSideChannel.Value, Scheduler]]
WrapperConfig = Dict[str, Dict[str, Any]]


class CurriculumPhase(TypedDict, total=False):
    when: Contraints
    for_iterations: int
    unity_config: UnityConfig
    wrappers: WrapperConfig


Curriculum = List[CurriculumPhase]


class EnvConfig(TypedDict, total=False):
    file_name: Optional[str]
    wrapper_types: Dict[str, Type[Wrapper]]
    curriculum: Curriculum


class CarEnv(Unity3DEnv):
    _config_side_channel: ConfigSideChannel
    _metrics_side_channel: MetricsSideChannel

    curriculum: Curriculum

    unity_config: UnityConfig
    wrappers: Dict[Type[Wrapper], Wrapper]
    wrapper_types: Dict[str, Type[Wrapper]]
    last_actions: Dict[AgentID, FloatNDArray]
    _schedulers: List[Scheduler]
    _iters_satisfying_curriculum: int

    def __init__(self,
                 *args,
                 file_name: str = None,
                 wrapper_types: Dict[str, Type[Wrapper]] = {},
                 curriculum: Curriculum = [],
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
            UnityEnvironment.__init__ = new_init
            super().__init__(*args,
                             episode_horizon=float('inf'),
                             file_name=file_name,
                             no_graphics=file_name is None,
                             **kwargs)
        finally:
            UnityEnvironment.__init__ = original_init

        self.curriculum = curriculum

        self.unity_config = {}
        self.wrappers = {}
        self.wrapper_types = wrapper_types
        self.last_actions = {}
        self._schedulers = []
        self.set_curriculum_phase(0)
        self._iters_satisfying_curriculum = 0

    @property
    def unwrapped(self) -> CarEnv:
        return self

    def set_curriculum_phase(self, phase: int) -> None:
        self.phase = phase
        logger = ray.get_actor('param_logger')
        logger.update_param.remote('curriculum_phase', phase)

        counter = ray.get_actor('agent_step_counter')
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
            wrapper_type = self.wrapper_types.get(wrapper_name, None)
            if wrapper_type is None:
                raise ValueError(f'Invalid wrapper type: "{wrapper_name}"')

            if wrapper_type in self.wrappers:
                wrapper = self.wrappers[wrapper_type]
                for param, val in wrapper_params.items():
                    if not hasattr(wrapper, param):
                        raise ValueError(f"Invalid param {param} for {wrapper_type}")
                    setattr(wrapper, param, val)
            else:
                self.wrappers[wrapper_type] = wrapper_type(self, **wrapper_params)

            for param, val in wrapper_params.items():
                logger.update_param.remote(f'{wrapper_type.__name__}/{param}', val)

    def _on_train_result(self, result: dict) -> None:
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
    def step(self, action_dict: Dict[AgentID, FloatNDArray]) \
            -> Tuple[Dict[AgentID, FloatNDArray], Dict[AgentID, float], Dict[AgentID, bool], Dict[AgentID, Info]]:

        self.last_actions.update(action_dict)
        raw_obs, rewards, dones, _ = super().step(action_dict)

        counter = ray.get_actor('agent_step_counter')
        counter.add_agent_steps.remote(len(action_dict))

        obs = self._get_obs(raw_obs)
        infos = self._get_infos(raw_obs, rewards)
        rewards = {agent_id: 0.0 for agent_id in rewards}

        return obs, rewards, dones, infos

    @override(Unity3DEnv)
    def reset(self) -> Dict[AgentID, FloatNDArray]:
        raw_obs = super().reset()
        return self._get_obs(raw_obs)

    @staticmethod
    def get_observation_space(curriculum: Curriculum, wrappers: List[Wrapper] = []) -> gym.Space:
        config = curriculum[0].get('unity_config', {}) if len(curriculum) >= 1 else {}
        rays_per_direction = config.get('AgentRaysPerDirection', 3)
        assert isinstance(rays_per_direction, int)
        rays = 2*rays_per_direction + 1
        space = spaces.Box(0, 1, (2*rays,), dtype=np.float32)
        for w in wrappers:
            space = w.get_observation_space(curriculum, space)
        return space

    @staticmethod
    def get_action_space(curriculum: Curriculum, wrappers: List[Wrapper] = []) -> gym.Space:
        space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        for w in wrappers:
            space = w.get_observation_space(curriculum, space)
        return space

    @staticmethod
    def get_policy(curriculum: Curriculum, wrappers: List[Wrapper] = []) \
            -> Tuple[Optional[Type], gym.Space, gym.Space, Dict]:

        obs_space = CarEnv.get_observation_space(curriculum, wrappers)
        action_space = CarEnv.get_action_space(curriculum, wrappers)
        return (None, obs_space, action_space, {})

    @staticmethod
    def _get_obs(raw_obs: Dict[AgentID, Tuple[FloatNDArray, FloatNDArray]]) -> Dict[AgentID, FloatNDArray]:
        return {agent_id: s[0]
                for agent_id, s in raw_obs.items()}

    def _get_infos(self, raw_obs: Dict[AgentID, Tuple[FloatNDArray, FloatNDArray]], rewards: Dict[AgentID, float]) \
            -> Dict[AgentID, Info]:
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


def create_env(config: EnvConfig) -> CarEnv:
    env = CarEnv(**config)
    for wrapper in config.get('wrappers', []):
        env = wrapper(env)
    return env


tune.register_env("car_env", create_env)