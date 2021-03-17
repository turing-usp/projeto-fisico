#!/usr/bin/env python3

import argparse


def main():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog=prog, width=88))
    parser.add_argument("--file_name", "-f", type=str, default=None,
                        help='Unity build file (default: None, run in editor)')
    parser.add_argument("--log_level", "-l", type=str, default="WARN",
                        help='DEBUG, INFO, WARN or ERROR (default: WARN)')
    parser.add_argument("--agents", type=int, default=256,
                        help='Total number of agents to run '
                        '(default: 256)')
    parser.add_argument("--workers", type=int, default=None,
                        help='Number of workers to use (default: all cpus minus one)')
    parser.add_argument("--gpus", type=int, default=None,
                        help='Number of gpus to use (default: all gpus)')
    parser.add_argument("--batch_size", type=int, default=65_536,
                        help='Batch size, counted in agent steps (default: 65_536)')
    parser.add_argument("--scheduler_step_frequency", type=int, default=None,
                        help='Frequency with which to step the hyperparameter schedulers'
                        '(default: batch_size)')
    parser.add_argument("--train_iters", type=int, default=128,
                        help='Number of training iterations to run (default: 128)')
    parser.add_argument("--time_scale", type=float, default=1000,
                        help='How fast to run the game (default: 1000)')
    parser.add_argument("--framework", type=str, choices=['torch', 'tf'], default='torch',
                        help='Use torch instead of tensorflow (default: torch)')
    args = parser.parse_args()

    import os

    if args.workers is None:
        cpus = os.cpu_count() or 1
        args.workers = cpus - 1

    if args.gpus is None:
        if args.framework == 'torch':
            import torch
            args.gpus = torch.cuda.device_count()
        else:
            import tensorflow as tf
            args.gpus = len(tf.config.list_physical_devices('GPU'))

    if args.file_name is not None:
        # use absolute path because rllib will change the cwd
        args.file_name = os.path.abspath(args.file_name)

    if args.scheduler_step_frequency is None:
        args.scheduler_step_frequency = args.batch_size

    print('Running with:')
    for k, v in vars(args).items():
        print('  ', k, '=', v)

    run_with_args(args)


def run_with_args(args):
    from ray import tune
    from physical_env import PhysicalEnv
    from callbacks import Callbacks
    from schedulers import LinearScheduler, ExponentialScheduler

    num_envs = max(args.workers, 1)

    config = {
        "env": "fisico",
        "env_config": {
            "file_name": args.file_name,
            "episode_horizon": float('inf'),
            "scheduler_step_frequency": args.scheduler_step_frequency,
            "unity_config": {
                "AgentCount": int(round(args.agents / num_envs)),
                "AgentCheckpointTTL": 60,
                "ChunkDifficulty": 0,
                "ChunkMinAgentsBeforeDestruction": 0,  # wait for all
                "ChunkTTL": 30,
                "HazardCountPerChunk": 0,
                "TimeScale": args.time_scale,
                "AgentVelocityBonus_CoeffPerSecond": LinearScheduler(2, 0, num_episodes=20),
            },
        },
        "callbacks": Callbacks,
        "num_workers": args.workers,
        "lr": 3e-4,
        "lambda": 0.95,
        "gamma": 0.995,
        "sgd_minibatch_size": min(args.batch_size, 128),
        "train_batch_size": args.batch_size,
        "num_gpus": args.gpus,
        "num_sgd_iter": 30,
        "rollout_fragment_length": 200,
        "clip_param": 0.2,
        "entropy_coeff": 0.002,
        "multiagent": {
            "policies": {"fisico": PhysicalEnv.policy},
            "policy_mapping_fn": lambda agent_id: "fisico",
            "count_steps_by": "agent_steps",
        },
        "model": {
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
            "use_lstm": True,
            "lstm_cell_size": 16,
        },
        "explore": True,
        "exploration_config": {
            "type": "StochasticSampling",
            "random_timesteps": args.scheduler_step_frequency,
        },
        "framework": args.framework,
        "no_done_at_end": True,
        "log_level": args.log_level,
    }

    stop = {
        "training_iteration": args.train_iters,
    }

    # Run the experiment.
    results = tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        local_dir='./results')


if __name__ == '__main__':
    main()