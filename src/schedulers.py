import numpy as np
from abc import abstractmethod, ABC
from event import Event


def identity(x):
    return x


class Scheduler(ABC):
    def __init__(self, f=None):
        self.on_update = Event()
        self.__f_value = 0
        self.__f = f or identity

    def update(self, new_val):
        new_f_val = self.__f(new_val)
        if new_f_val != self.__f_value:
            self.__f_value = new_f_val
            self.on_update()

    @property
    def value(self):
        return self.__f_value

    def __make_repr(self, *args, **kwargs):
        arg_strs = [
            *(repr(v) for v in args),
            *(k + '=' + repr(v) for k, v in kwargs.items()),
        ]
        return self.__class__.__name__ + '(' + ', '.join(arg_strs) + ')'

    def __repr__(self, *args, **kwargs):
        if self.__f != identity:
            kwargs.setdefault('f', self.__f)
        return self.__make_repr(*args, **kwargs)

    @abstractmethod
    def step_to(self, n):
        pass


class PiecewiseScheduler(Scheduler):
    def __init__(self, parts, **kwargs):
        super().__init__(**kwargs)
        assert parts[0][0] == 0
        parts.sort()
        self.parts = parts
        self.current_part = 0

        self.step_to(0)

        self._repr = f'PiecewiseScheduler(' + repr(self.parts) + ')'

    def step_to(self, n):
        while n < self.parts[self.current_part][0]:
            self.current_part -= 1
        while self.current_part + 1 < len(self.parts) and self.parts[self.current_part+1][0] <= n:
            self.current_part += 1
        self.update(self.parts[self.current_part][1])

    def __repr__(self):
        return super().__repr__(self.parts)


class LinearScheduler(Scheduler):
    def __init__(self, start, end, agent_timesteps, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.agent_timesteps = agent_timesteps

        self.step_to(0)

    def step_to(self, n):
        if n >= self.agent_timesteps:
            self.update(self.end)
        else:
            val = ((self.agent_timesteps-n)*self.start + n*self.end) / self.agent_timesteps
            self.update(val)

    def __repr__(self):
        return super().__repr__(self.start, self.end, self.agent_timesteps)


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value, multiplier, min_value=None, max_value=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        self.multiplier = multiplier
        self.min_value = min_value
        self.max_value = max_value

    def step_to(self, n):
        val = self.initial_value * self.multiplier**n
        if self.min_value is not None or self.max_value is not None:
            val = np.clip(val, self.min_value, self.max_value)
        self.update(val)

    def __repr__(self):
        kwargs = {}
        if self.min_value is not None:
            kwargs['min_value'] = self.min_value
        if self.max_value is not None:
            kwargs['max_value'] = self.max_value
        return super().__repr__(self.initial_value, self.multiplier, **kwargs)
