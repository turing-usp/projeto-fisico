import numpy as np
from gym import spaces
import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from config_side_channel import ConfigSideChannel
from metrics_side_channel import MetricsSideChannel

from mlagents_envs.environment import UnityEnvironment

from schedulers import Scheduler


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


class PhysicalEnv(Unity3DEnv):

    observation__space = spaces.Box(float("-inf"), float("inf"),
                                    (14,), dtype=np.float32)

    action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)

    policy = (None, observation__space, action_space, {})

    def __init__(self, *args, unity_config={}, curriculum=[], **kwargs):
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
            super().__init__(*args, **kwargs)
        finally:
            UnityEnvironment.__init__ = original_init

        self.curriculum = curriculum

        self.unity_config = {}
        self._schedulers = []
        self.set_phase(-1, unity_config_updates=unity_config)
        self._iters_satisfying_curriculum = 0

    def set_phase(self, phase, unity_config_updates=None):
        self.phase = phase
        logger = ray.get_actor('config_logger')
        logger.update.remote('phase', phase)

        counter = ray.get_actor('agent_step_counter')
        counter.new_phase.remote()

        if unity_config_updates is None:
            unity_config_updates = self.curriculum[phase]['unity_config']

        for k, v in unity_config_updates.items():
            old = self.unity_config.get(k)
            if isinstance(old, Scheduler):
                self._schedulers.remove(old)
            if isinstance(v, Scheduler):
                self._schedulers.append(v)
            self.unity_config[k] = v
            self._config_side_channel.set(k, v)

    def on_train_result(self, result):
        next_phase = self.phase+1
        if next_phase < len(self.curriculum):
            if satisfies_constraints(result, self.curriculum[next_phase]['when']):
                self._iters_satisfying_curriculum += 1
            else:
                self._iters_satisfying_curriculum = 0

            min_iters = self.curriculum[next_phase].get('for_iterations', 1)
            if self._iters_satisfying_curriculum >= min_iters:
                self.set_phase(next_phase)
                return

        for sch in self._schedulers:
            sch.step_to(result['agent_steps_this_phase'])

    def transform_actions(self, actions):
        return actions

    def transform_observations(self, obs):
        # Even indices indicate whether the ray hit something
        # Odd indices indicate the normalized distance for each ray (or the amx if no object has hit)
        # Keep only the odd indices
        return {agent_id: s[1::2] for agent_id, s in obs.items()}

    def transform_rewards(self, rewards):
        return rewards

    @override(Unity3DEnv)
    def step(self, action_dict):
        action_dict = self.transform_actions(action_dict)
        obs, rewards, dones, infos = super().step(action_dict)
        obs = self.transform_observations(obs)
        rewards = self.transform_rewards(rewards)

        counter = ray.get_actor('agent_step_counter')
        counter.add_agent_steps.remote(len(action_dict))

        return obs, rewards, dones, infos

    @override(Unity3DEnv)
    def reset(self):
        obs = super().reset()
        return self.transform_observations(obs)


tune.register_env(
    "fisico",
    lambda config: PhysicalEnv(
        no_graphics='file_name' in config,
        **config,
    )
)
