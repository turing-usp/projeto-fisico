from collections import defaultdict
import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks


class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        tracker = ray.get_actor('agent_metric_tracker')
        counter = ray.get_actor('agent_step_counter')
        new_metrics, (result['agent_steps_total'], result['agent_steps_this_phase']) = ray.get([
            tracker.get_metrics.remote(reset=True),
            counter.get_steps.remote(),
        ])

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
                lambda env: env.on_train_result(result)))
