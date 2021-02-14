import argparse
import torch
from ray import tune
from physical_env import PhysicalEnv


def main():
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
            "policies": {"fisico": PhysicalEnv.policy},
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
