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

    @abstractmethod
    def step(self):
        pass


class PiecewiseScheduler(Scheduler):
    def __init__(self, parts, f=None):
        super().__init__(f)
        assert parts[0][0] == 0
        parts.sort()
        self.steps_remaining = parts

        self.n = -1
        self.step()

    def step(self):
        if not self.steps_remaining:
            return

        self.n += 1
        target_n, new_val = self.steps_remaining
        if self.n == target_n:
            self.update(new_val)
            self.steps_remaining.pop(0)


class LinearScheduler(Scheduler):
    def __init__(self, start, end, num_episodes, f=None):
        super().__init__(f)
        self.start = start
        self.end = end
        self.num_episodes = num_episodes

        self.n = -1
        self.step()

    def step(self):
        if self.n >= self.num_episodes:
            return

        self.n += 1
        a = self.n / self.num_episodes
        val = (1-a)*self.start + a*self.end
        self.update(val)


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value, multiplier, min_value=None, max_value=None, f=None):
        super().__init__(f)
        self._value = initial_value
        self.multiplier = multiplier
        self.min_value = min_value
        self.max_value = max_value

    def step(self):
        self._value = np.clip(self._value * self.multiplier, self.min_value, self.max_value)
        self.update(self._value)
