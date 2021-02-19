from collections import defaultdict
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.env.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override


class PhysicalEnv(Unity3DEnv):

    observation__space = spaces.Box(float("-inf"), float("inf"),
                                    (21,), dtype=np.float32)

    action_space = spaces.Tuple((
        spaces.Box(-1, 1, (2,), dtype=np.float32),
        spaces.Discrete(2),
    ))

    policy = (None, observation__space, action_space, {})

    def __init__(self, *args, bonus_coeff=0, bonus_decay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonus_coeff = bonus_coeff
        self.bonus_decay = bonus_decay
        self.last_actions = defaultdict(lambda: [0, 0, 0])

    def transform_rewards(self, rewards):
        for agent in rewards:
            accel = self.last_actions[agent][0]
            brake = self.last_actions[agent][2] == 0
            assert abs(accel) <= 1
            if accel > 0 and not brake:
                rewards[agent] += self.bonus_coeff * accel
        return rewards

    @override(Unity3DEnv)
    def step(self, action_dict):
        action_dict = {
            agent: np.array([*a0, a1]) for (agent, (a0, a1)) in action_dict.items()
        }
        obs, rewards, dones, infos = super().step(action_dict)

        # Save the last actions for each agent
        self.last_actions.update(action_dict)

        return obs, self.transform_rewards(rewards), dones, infos

    @override(Unity3DEnv)
    def reset(self):
        self.bonus_coeff *= self.bonus_decay
        return super().reset()


tune.register_env(
    "fisico",
    lambda config: PhysicalEnv(
        no_graphics=False,
        episode_horizon=1024,
        **config,
    )
)
