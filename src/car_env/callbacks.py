import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.trainer import Trainer


class CarEnvCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        tracker = ray.get_actor('agent_metric_tracker')
        logger = ray.get_actor('param_logger')
        counter = ray.get_actor('agent_step_counter')
        new_metrics, logged_config, (result['agent_steps_total'], result['agent_steps_this_phase']) = \
            ray.get([  # type: ignore
                tracker.get_metrics.remote(reset=True),
                logger.get_params.remote(),
                counter.get_steps.remote(),
            ])

        result['env'] = logged_config.copy()

        new_metrics.setdefault('agent_checkpoints', [])
        new_metrics.setdefault('agent_reward', [])
        hist = result['hist_stats']
        metrics = result['custom_metrics']
        for k, vs in new_metrics.items():
            hist.setdefault(k, []).extend(vs)
            metrics[k + '_min'] = min(vs) if vs else np.nan
            metrics[k + '_max'] = max(vs) if vs else np.nan
            metrics[k + '_mean'] = np.mean(vs) if vs else np.nan

        trainer.workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._on_train_result(result)))
