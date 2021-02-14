from collections import defaultdict
import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.env.unity3d_env import Unity3DEnv
from ray.rllib.utils.annotations import override
import torch


class PhysicalEnv(Unity3DEnv):
    def __init__(self, *args, bonus_coeff=0, bonus_decay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonus_coeff = bonus_coeff
        self.bonus_decay = bonus_decay
        self.last_actions = defaultdict(lambda: [0, 0, 0])

    def transform_rewards(self, rewards):
        for agent in rewards:
            assert abs(self.last_actions[agent][0]) <= 1
            rewards[agent] += self.bonus_coeff * self.last_actions[agent][0]
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


def main():
    tune.register_env(
        "fisico",
        lambda _: PhysicalEnv(
            no_graphics=False,
            episode_horizon=1000,
            bonus_coeff=.2,
            bonus_decay=.7,
        )
    )

    obs_space = spaces.Box(float("-inf"), float("inf"),
                           (21,), dtype=np.float32)
    action_space = spaces.Tuple(
        (spaces.Box(-1, 1, (2,), dtype=np.float32), spaces.Discrete(2)))
    policy = (None, obs_space, action_space, {})

    config = {
        "env": "fisico",
        "num_workers": 0,
        "lr": 3e-4,
        "lambda": 0.95,
        "gamma": 0.995,
        "sgd_minibatch_size": 256,
        "train_batch_size": 1000,
        "num_gpus": torch.cuda.device_count(),
        "num_sgd_iter": 20,
        "rollout_fragment_length": 100,
        "clip_param": 0.2,
        "entropy_coeff": 0.02,

        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": {"fisico": policy},
            "policy_mapping_fn": lambda agent_id: "fisico",
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "use_lstm": True,
        },
        "framework": "torch",
        "no_done_at_end": True,
    }

    stop = {
        "timesteps_total": 50_000,
    }

    # Run the experiment.
    results = tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=10,
        local_dir='./results')


if __name__ == '__main__':
    main()
