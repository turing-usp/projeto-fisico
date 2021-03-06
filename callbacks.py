from collections import defaultdict
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune


CALLBACK_MARK = '$from_callback$'


class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs) -> None:
        hist = result['hist_stats']
        metrics = result['custom_metrics']
        for k in list(hist.keys()):
            if k.startswith(CALLBACK_MARK):
                v = hist.pop(k)
                k = k[len(CALLBACK_MARK):]
                hist[k] = v
                metrics[k + '_min'] = min(v)
                metrics[k + '_max'] = max(v)
                metrics[k + '_mean'] = np.mean(v)

    def on_episode_end(self, *, episode, base_env, **kwargs):
        for env in base_env.get_unwrapped():
            for k, v in env.get_metrics(reset=True).items():
                episode.hist_data.setdefault(CALLBACK_MARK + k, []).extend(v)
