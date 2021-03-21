import ray
from collections import defaultdict


_actors = []


def init():
    # Store handles to the actors in order to keep them alive
    _actors.extend([
        AgentStepCounter.options(name='agent_step_counter').remote(),
        UnityConfigLogger.options(name='unity_config_logger').remote(),
        AgentMetricTracker.options(name='agent_metric_tracker').remote(),
    ])


@ray.remote
class AgentStepCounter:
    def __init__(self):
        self.total_agent_steps = 0

    def add_agent_steps(self, n):
        self.total_agent_steps += n

    def get_steps(self):
        return self.total_agent_steps


@ray.remote
class UnityConfigLogger:
    def __init__(self):
        self.configs = {}

    def update(self, key, value):
        if key not in self.configs or self.configs[key] != value:
            print(f'Setting {key}={value}')
            self.configs[key] = value


@ray.remote
class AgentMetricTracker:
    def __init__(self):
        self.metrics = defaultdict(list)

    def add_metric(self, key, value):
        self.metrics[key].append(value)

    def get_metrics(self, reset=True):
        metrics = self.metrics.copy()
        if reset:
            self.metrics.clear()
        return metrics
