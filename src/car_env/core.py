import numpy as np
from gym import spaces
import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from mlagents_envs.environment import UnityEnvironment

from .config_side_channel import ConfigSideChannel
from .metrics_side_channel import MetricsSideChannel
from .schedulers import Scheduler


def flatten(obj, d=None, prefix=''):
    if d is None:
        d = {}
    for key, val in obj.items():
        full_key = prefix+key
        if isinstance(val, dict):
            flatten(val, d=d, prefix=full_key + '/')
        else:
            d[full_key] = val
    return d


def satisfies_constraint(flat_obj, key, min_value):
    if key not in flat_obj:
        print(f'Warning: missing constraint key "{key}" (in constraint {key} >= {min_value})')
        return False
    return flat_obj[key] >= min_value


def satisfies_constraints(obj, constraints):
    flat_obj = flatten(obj)
    return all(satisfies_constraint(flat_obj, key, min_value)
               for key, min_value in constraints.items())


class CarEnv(Unity3DEnv):

    def __init__(self, *args, file_name=None, unity_config={}, curriculum=[], **kwargs):
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
        self._schedulers = []
        self.set_curriculum_phase(0)
        self._iters_satisfying_curriculum = 0

    def set_curriculum_phase(self, phase):
        self.phase = phase
        logger = ray.get_actor('param_logger')
        logger.update_param.remote('curriculum_phase', phase)

        counter = ray.get_actor('agent_step_counter')
        counter.new_phase.remote()

        for k, v in self.curriculum[phase].get('unity_config', 0).items():
            old = self.unity_config.get(k)
            if isinstance(old, Scheduler):
                self._schedulers.remove(old)
            if isinstance(v, Scheduler):
                self._schedulers.append(v)
            self.unity_config[k] = v
            self._config_side_channel.set(k, v)

    def _on_train_result(self, result):
        next_phase = self.phase+1
        if next_phase < len(self.curriculum):
            if satisfies_constraints(result, self.curriculum[next_phase]['when']):
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
    def step(self, action_dict):
        obs, rewards, dones, infos = super().step(action_dict)

        counter = ray.get_actor('agent_step_counter')
        counter.add_agent_steps.remote(len(action_dict))

        return obs, rewards, dones, infos

    @override(Unity3DEnv)
    def reset(self):
        obs = super().reset()
        return obs

    @staticmethod
    def get_observation_space(curriculum):
        config = curriculum[0].get('unity_config', {}) if len(curriculum) >= 1 else {}
        rays_per_direction = config.get('AgentRaysPerDirection', 3)
        rays = 2*rays_per_direction + 1
        return spaces.Box(0, 1, (2*rays,), dtype=np.float32)

    @staticmethod
    def get_action_space(curriculum):
        return spaces.Box(-1, 1, (3,), dtype=np.float32)

    @staticmethod
    def get_policy(curriculum):
        obs_space = CarEnv.get_observation_space(curriculum)
        action_space = CarEnv.get_action_space(curriculum)
        return (None, obs_space, action_space, {})


tune.register_env(
    "car_env",
    lambda config: CarEnv(**config)
)
