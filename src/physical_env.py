import numpy as np
from gym import spaces
import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from config_side_channel import ConfigSideChannel
from metrics_side_channel import MetricsSideChannel
import copy

from mlagents_envs.environment import UnityEnvironment

from schedulers import find_schedulers


class PhysicalEnv(Unity3DEnv):

    observation__space = spaces.Box(float("-inf"), float("inf"),
                                    (14,), dtype=np.float32)

    action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)

    policy = (None, observation__space, action_space, {})

    def __init__(self, *args, unity_config={}, **kwargs):
        self._config_side_channel = ConfigSideChannel(**unity_config)
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

        self.schedulers = find_schedulers({'unity_config': unity_config})

    def on_train_result(self, result):
        for sch in self.schedulers.values():
            sch.step_to(result['agent_steps_total'])

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
