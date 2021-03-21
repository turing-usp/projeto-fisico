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


def satisfies_constraints(obj, constraints):
    return all(key in obj and obj[key] >= min_value
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
        if next_phase < len(self.curriculum) and \
                satisfies_constraints(result, self.curriculum[next_phase]['when']):
            self.set_phase(next_phase)
        else:
            for sch in self._schedulers:
                sch.step()

    @override(Unity3DEnv)
    def step(self, action_dict):
        obs, rewards, dones, infos = super().step(action_dict)
        counter = ray.get_actor('agent_step_counter')
        counter.add_agent_steps.remote(len(action_dict))
        return obs, rewards, dones, infos


tune.register_env(
    "fisico",
    lambda config: PhysicalEnv(
        no_graphics='file_name' in config,
        **config,
    )
)
