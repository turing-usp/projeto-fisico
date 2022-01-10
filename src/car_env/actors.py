from typing import Any, DefaultDict, Dict, List, Tuple, Union

import ray
import ray.actor
from collections import defaultdict


_AGENT_STEP_COUNTER = 'agent_step_counter'
_PARAM_LOGGER = 'param_logger'
_AGENT_METRIC_TRACKER = 'agent_metric_tracker'
_actors = []


def init() -> None:
    # Store handles to the actors in order to keep them alive
    _actors.extend([
        ray.remote(AgentStepCounter).options(name=_AGENT_STEP_COUNTER).remote(),
        ray.remote(ParamLogger).options(name=_PARAM_LOGGER).remote(),
        ray.remote(AgentMetricTracker).options(name=_AGENT_METRIC_TRACKER).remote(),
    ])

    # NOTE: we use ray.remote here instead of as a decorator on the actor classes
    # in order to allow for proper documentation to be generated


def _get_actor(name: str) -> ray.actor.ActorHandle:
    try:
        return ray.get_actor(name)
    except ValueError:
        raise RuntimeError(f'No such agent: {name}. Did you forget to call car_env.actors.init() in the main thread?')


def agent_step_counter() -> ray.actor.ActorHandle:
    return _get_actor(_AGENT_STEP_COUNTER)


def param_logger() -> ray.actor.ActorHandle:
    return _get_actor(_PARAM_LOGGER)


def agent_metric_tracker() -> ray.actor.ActorHandle:
    return _get_actor(_AGENT_METRIC_TRACKER)


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
