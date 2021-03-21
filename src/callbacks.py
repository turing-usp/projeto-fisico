from collections import defaultdict
import numpy as np
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks


class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        tracker = ray.get_actor('agent_metric_tracker')
        counter = ray.get_actor('agent_step_counter')
        new_metrics, result['agent_steps_total'] = ray.get([
            tracker.get_metrics.remote(reset=True),
            counter.get_steps.remote(),
        ])

        hist = result['hist_stats']
        metrics = result['custom_metrics']
        for k, vs in new_metrics.items():
            hist.setdefault(k, []).extend(vs)
            metrics[k + '_min'] = min(vs)
            metrics[k + '_max'] = max(vs)
            metrics[k + '_mean'] = np.mean(vs)

        trainer.workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env.on_train_result(result)))
