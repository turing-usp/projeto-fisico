import os
import numpy as np
from gym.spaces import Box
from ray import tune
from ray.rllib.env.unity3d_env import Unity3DEnv
import torch


def main():
    tune.register_env(
        "fisico",
        lambda _: Unity3DEnv(
            no_graphics=False,
            episode_horizon=1000,
        ))

    obs_space = Box(float("-inf"), float("inf"), (21,), dtype=np.float32)
    action_space = Box(-1, 1, (3,), dtype=np.float32)
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
        "entropy_coeff": 0.01,

        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": {"fisico": policy},
            "policy_mapping_fn": lambda agent_id: "fisico",
        },
        "model": {
            "fcnet_hiddens": [256, 256],
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
