from typing import Any, DefaultDict, Dict, List, Tuple, Union

import ray
import ray.actor
from collections import defaultdict


_actors = []


def init() -> None:
    # Store handles to the actors in order to keep them alive
    _actors.extend([
        AgentStepCounter.options(name='agent_step_counter').remote(),  # type: ignore
        ParamLogger.options(name='param_logger').remote(),  # type: ignore
        AgentMetricTracker.options(name='agent_metric_tracker').remote(),  # type: ignore
    ])


@ray.remote
class AgentStepCounter:
    agent_steps_total: int
    agent_steps_this_phase: int

    def __init__(self) -> None:
        self.agent_steps_total = 0
        self.agent_steps_this_phase = 0

    def add_agent_steps(self, n: int) -> None:
        self.agent_steps_total += n
        self.agent_steps_this_phase += n

    def get_steps(self) -> Tuple[int, int]:
        return self.agent_steps_total, self.agent_steps_this_phase

    def new_phase(self) -> None:
        self.agent_steps_this_phase = 0


@ray.remote
class ParamLogger:
    configs: Dict[str, Any]

    def __init__(self) -> None:
        self.configs = {}

    def update_param(self, key: str, value: Any) -> None:
        if key not in self.configs or self.configs[key] != value:
            print(f'Setting {key}={value}')
            self.configs[key] = value

    def get_params(self) -> Dict[str, Any]:
        return self.configs


@ray.remote
class AgentMetricTracker:
    metrics: DefaultDict[str, List[Union[int, float]]]

    def __init__(self) -> None:
        self.metrics = defaultdict(list)

    def add_metric(self, key: str, value: Union[int, float]) -> None:
        self.metrics[key].append(value)

    def get_metrics(self, reset=True) -> DefaultDict[str, List[Union[int, float]]]:
        metrics = self.metrics.copy()
        if reset:
            self.metrics.clear()
        return metrics
