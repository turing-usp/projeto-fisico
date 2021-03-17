import ray


_actors = []


def init():
    # Store handles to the actors in order to keep them alive
    _actors.extend([
        StepCounter.options(name='step_counter').remote(),
        UnityConfigLogger.options(name='unity_config_logger').remote(),
    ])


@ray.remote
class StepCounter:
    def __init__(self):
        self.total_agent_steps = 0

    def add_agent_steps(self, n):
        self.total_agent_steps += n
        return self.total_agent_steps


@ray.remote
class UnityConfigLogger:
    def __init__(self):
        self.configs = {}

    def update(self, key, value):
        if key not in self.configs or self.configs[key] != value:
            print(f'Setting {key}={value}')
            self.configs[key] = value
