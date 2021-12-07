#!/usr/bin/env python3

import argparse
from typing import Literal, Optional

from car_env import EnvConfig


class Args(argparse.Namespace):
    file_name: Optional[str]
    log_level: str
    agents: int
    workers: int
    gpus: int
    max_train_iters: int
    time_scale: float
    framework: Literal['torch', 'tf']


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
    parser.add_argument("--max_train_iters", type=int, default=512,
                        help='Number of training iterations to run (default: 512)')
    parser.add_argument("--time_scale", type=float, default=1000,
                        help='How fast to run the game (default: 1000)')
    parser.add_argument("--framework", type=str, choices=['torch', 'tf'], default='torch',
                        help='Use torch instead of tensorflow (default: torch)')
    args = parser.parse_args(namespace=Args())

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

    run_with_args(args)


def run_with_args(args: Args):
    import ray
    from ray import tune

    from car_env import CarEnv, LinearScheduler, CarEnvCallbacks, init as init_car_env
    import wrappers

    ray.init()
    init_car_env()

    num_envs = max(args.workers, 1)
    agent_count_per_env = int(round(args.agents / num_envs))
    if agent_count_per_env * num_envs != args.agents:
        print('Rounding agent count to', agent_count_per_env*num_envs)

    min_steps_per_phase = 250_000
    for_iterations = 10

    config = {
        # === General settings ===
        "env": "car_env",
        "callbacks": CarEnvCallbacks,
        "num_workers": args.workers,
        "num_gpus": args.gpus,
        "framework": args.framework,
        "no_done_at_end": True,
        "log_level": args.log_level,

        # === Environment settings (curriculum) ===
        "env_config": EnvConfig(
            file_name=args.file_name,
            wrappers=[
                wrappers.CheckpointReward,
                wrappers.VelocityReward,
                wrappers.DeathPenalty,
                wrappers.BrakePenalty,
                wrappers.RewardLogger,
            ],
            curriculum=[
                {  # Phase 0 (initial settings)
                    "unity_config": {
                        "AgentCount": agent_count_per_env,
                        "AgentCheckpointTTL": 60,
                        "AgentDecisionPeriod": 10,
                        "ChunkMinAgentsBeforeDestruction": 1,
                        "ChunkTTL": 30,
                        "TimeScale": args.time_scale,
                        "AgentRaysPerDirection": 7,
                        "AgentRayLength": 128,
                        "AgentCheckpointMax": 15,
                        "ChunkDelayBeforeDestruction": 60,
                        "ChunkDifficulty": 0,
                        "HazardCountPerChunk": 0,
                        "HazardMinSpeed": 0,
                        "HazardMaxSpeed": 0,
                    },
                    "wrappers": {
                        "CheckpointReward": {
                            "max_reward": 100,
                            "min_velocity": 0,
                            "max_velocity": 1,
                        },
                        "VelocityReward": {
                            "coeff_per_second": 2,
                            "warmup_time": 10,
                            "min_velocity": -10,
                            "max_velocity": 1,
                        },
                        "DeathPenalty": {
                            "penalty": 100,
                        },
                        "BrakePenalty": {
                            "coeff_per_second": 1,
                        },
                    },
                },
                {  # Phase 1
                    "when": {
                        "custom_metrics/agent_checkpoints_mean": 5,
                        "agent_steps_this_phase": min_steps_per_phase,
                    },
                    "for_iterations": for_iterations,
                    "unity_config": {
                        "HazardCountPerChunk": 1,
                    },
                },
                {  # Phase 2
                    "when": {
                        "custom_metrics/agent_checkpoints_mean": 5,
                        "agent_steps_this_phase": min_steps_per_phase,
                    },
                    "for_iterations": for_iterations,
                    "unity_config": {
                        "AgentVelocityBonus_CoeffPerSecond": 0,
                        "HazardMinSpeed": LinearScheduler(0, 10, 1_500_000),
                        "HazardMaxSpeed": LinearScheduler(0, 10, 1_500_000),
                    },
                    "wrappers": {
                        "VelocityReward": {
                            "coeff_per_second": 0,
                        },
                    }
                },
                {  # Phase 3
                    "when": {
                        "custom_metrics/agent_checkpoints_mean": 5,
                        "agent_steps_this_phase": min_steps_per_phase,
                    },
                    "for_iterations": for_iterations,
                    "unity_config": {
                        "HazardMinSpeed": 0,
                        "HazardMaxSpeed": 10,
                        "ChunkDifficulty": 1,
                    },
                },
            ],
        ),

        # === Model ===
        "model": {
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
            "use_lstm": True,
            "lstm_cell_size": 512,
        },

        # === Training settings ===
        "gamma": 0.999,
        "lr": 1e-4,
        "lambda": 0.98,
        "train_batch_size": 15_000,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 30,
        "rollout_fragment_length": 100,
    }

    # === Multiagent settings ===
    config["multiagent"] = {
        "policies": {"car_agent": CarEnv.get_policy(config["env_config"]["curriculum"])},
        "policy_mapping_fn": lambda agent_id: "car_agent",
        "count_steps_by": "agent_steps",
    }

    # === Stopping criteria (stops when any of the criteria are met) ===
    stop = {
        "training_iteration": args.max_train_iters,
        "custom_metrics/agent_checkpoints_mean": 9,
        "custom_metrics/agent_checkpoints_min": 5.0,
    }

    # === Run the training ===
    tune.run(
        "PPO",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir='./results')


if __name__ == '__main__':
    main()
