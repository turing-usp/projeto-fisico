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
    def step(self):
        pass


class PiecewiseScheduler(Scheduler):
    def __init__(self, parts, **kwargs):
        super().__init__(**kwargs)
        assert parts[0][0] == 0
        parts.sort()
        self.parts = parts.copy()
        self.steps_remaining = parts

        self.n = -1
        self.step()

        self._repr = f'PiecewiseScheduler(' + repr(self.parts) + ')'

    def step(self):
        if not self.steps_remaining:
            return

        self.n += 1
        target_n, new_val = self.steps_remaining[0]
        if self.n == target_n:
            self.update(new_val)
            self.steps_remaining.pop(0)

    def __repr__(self):
        return super().__repr__(self.parts)


class LinearScheduler(Scheduler):
    def __init__(self, start, end, num_episodes, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end
        self.num_episodes = num_episodes

        self.n = -1
        self.step()

    def step(self):
        if self.n >= self.num_episodes:
            return

        self.n += 1
        val = ((self.num_episodes-self.n)*self.start +
               self.n*self.end) / self.num_episodes
        self.update(val)

    def __repr__(self):
        return super().__repr__(self.start, self.end, self.num_episodes)


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value, multiplier, min_value=None, max_value=None, **kwargs):
        super().__init__(**kwargs)
        self._value = initial_value
        self.initial_value = initial_value
        self.multiplier = multiplier
        self.min_value = min_value
        self.max_value = max_value

    def step(self):
        self._value = self._value * self.multiplier
        if self.min_value is not None or self.max_value is not None:
            self._value = np.clip(self._value, self.min_value, self.max_value)
        self.update(self._value)

    def __repr__(self):
        kwargs = {}
        if self.min_value is not None:
            kwargs['min_value'] = self.min_value
        if self.max_value is not None:
            kwargs['max_value'] = self.max_value
        return super().__repr__(self.initial_value, self.multiplier, **kwargs)


def find_schedulers(obj, base='', d=None):
    if d is None:
        d = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f'{base}.{k}' if base else k
            if isinstance(v, Scheduler):
                d[full_key] = v
            find_schedulers(v, base=full_key, d=d)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_schedulers(v, base=f'{base}[{i}]', d=d)

    return d
