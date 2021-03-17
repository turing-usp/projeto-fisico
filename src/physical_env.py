from collections import defaultdict
import numpy as np
from gym import spaces
import ray
from ray import tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
from config_side_channel import ConfigSideChannel
from metrics_side_channel import MetricsSideChannel

from mlagents_envs.environment import UnityEnvironment

from schedulers import Scheduler, find_schedulers


class PhysicalEnv(Unity3DEnv):

    observation__space = spaces.Box(float("-inf"), float("inf"),
                                    (14,), dtype=np.float32)

    action_space = spaces.Box(-1, 1, (3,), dtype=np.float32)

    policy = (None, observation__space, action_space, {})

    def __init__(self, *args, scheduler_step_period=1, unity_config={}, **kwargs):
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
        self.agent_count = unity_config['AgentCount']
        self.scheduler_step_period = scheduler_step_period
        self.total_agent_steps = 0
        self._agent_steps_since_scheduler_step = 0

    def set_config(self, key, value):
        self._config_side_channel.set(key, value)

    def get_metrics(self, reset=False):
        m = self._metrics_side_channel.metrics.copy()
        if reset:
            self._metrics_side_channel.metrics.clear()
        return m

    def update_schedulers(self, agent_steps):
        counter = ray.get_actor('step_counter')
        new_total = ray.get(counter.add_agent_steps.remote(agent_steps))
        self._agent_steps_since_scheduler_step += new_total - self.total_agent_steps
        self.total_agent_steps = new_total

        while self._agent_steps_since_scheduler_step >= self.scheduler_step_period:
            self._agent_steps_since_scheduler_step -= self.scheduler_step_period
            for sch in self.schedulers.values():
                sch.step()

    @override(Unity3DEnv)
    def step(self, action_dict):
        obs, rewards, dones, infos = super().step(action_dict)
        self.update_schedulers(len(action_dict))
        return obs, rewards, dones, infos

    @override(Unity3DEnv)
    def reset(self):
        self.update_schedulers(self.agent_count)
        return super().reset()


tune.register_env(
    "fisico",
    lambda config: PhysicalEnv(
        no_graphics='file_name' in config,
        **config,
    )
)
