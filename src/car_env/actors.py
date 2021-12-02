import ray
from collections import defaultdict


_actors = []


def init():
    # Store handles to the actors in order to keep them alive
    _actors.extend([
        AgentStepCounter.options(name='agent_step_counter').remote(),
        ParamLogger.options(name='param_logger').remote(),
        AgentMetricTracker.options(name='agent_metric_tracker').remote(),
    ])


@ray.remote
class AgentStepCounter:
    def __init__(self):
        self.agent_steps_total = 0
        self.agent_steps_this_phase = 0

    def add_agent_steps(self, n):
        self.agent_steps_total += n
        self.agent_steps_this_phase += n

    def get_steps(self):
        return self.agent_steps_total, self.agent_steps_this_phase

    def new_phase(self):
        self.agent_steps_this_phase = 0


@ray.remote
class ParamLogger:
    def __init__(self):
        self.configs = {}

    def update_param(self, key, value):
        if key not in self.configs or self.configs[key] != value:
            print(f'Setting {key}={value}')
            self.configs[key] = value

    def get_params(self):
        return self.configs


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
