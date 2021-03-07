from collections import defaultdict
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.env.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from config_side_channel import ConfigSideChannel
from metrics_side_channel import MetricsSideChannel

from mlagents_envs.environment import UnityEnvironment

from schedulers import Scheduler


def find_schedulers(obj, base=''):
    d = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f'{base}.{k}' if base else k
            if isinstance(v, Scheduler):
                d[full_key] = v
            else:
                d.update(find_schedulers(v, base=full_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            d.update(find_schedulers(v, base=f'{base}[{i}]'))

    return d


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

    def set_config(self, key, value):
        self._config_side_channel.set(key, value)

    def get_metrics(self, reset=False):
        m = self._metrics_side_channel.metrics.copy()
        if reset:
            self._metrics_side_channel.metrics.clear()
        return m

    @override(Unity3DEnv)
    def step(self, action_dict):
        action_dict = {
            agent: np.array([*a0, a1]) for (agent, (a0, a1)) in action_dict.items()
        }

        return super().step(action_dict)

    @override(Unity3DEnv)
    def reset(self):
        for sch in self.schedulers.values():
            sch.step()
        return super().reset()


tune.register_env(
    "fisico",
    lambda config: PhysicalEnv(
        no_graphics='file_name' in config,
        **config,
    )
)
