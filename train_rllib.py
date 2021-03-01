import argparse
import torch
from ray import tune
from physical_env import PhysicalEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", "-f", default=None)
    args = parser.parse_args()

    config = {
        "env": "fisico",
        "env_config": {
            "file_name": args.file_name,
            "bonus_coeff": 0,
            "bonus_decay": .95,
            "unity_config": {
                "AgentCount": 20,
                "AgentCheckpointTTL": 60,
                "ChunkDifficulty": 0,
                "ChunkMinAgentsBeforeDestruction": 0,  # wait for all
                "ChunkTTL": 30,
                "HazardCountPerChunk": 1,
                "TimeScale": 10,
            },
        },
        "num_workers": 0,
        "lr": 3e-4,
        "lambda": 0.95,
        "gamma": 0.995,
        "sgd_minibatch_size": 512,
        "train_batch_size": 2048,
        "num_gpus": torch.cuda.device_count(),
        "num_sgd_iter": 20,
        "rollout_fragment_length": 32,  # for 25 agents
        "clip_param": 0.2,
        "entropy_coeff": 0.005,

        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": {"fisico": PhysicalEnv.policy},
            "policy_mapping_fn": lambda agent_id: "fisico",
        },
        "model": {
            "fcnet_hiddens": [256, 256],
            "use_lstm": True,
            "lstm_cell_size": 64
        },
        "framework": "torch",
        "no_done_at_end": True,
    }

    stop = {
        "timesteps_total": 50*1024,
    }

    # Run the experiment.
    results = tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=5,
        local_dir='./results1')


if __name__ == '__main__':
    main()
