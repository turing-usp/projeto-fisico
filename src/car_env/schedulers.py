from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from abc import abstractmethod, ABC


def identity(x):
    return x


T = TypeVar('T')


class EventHandler(Generic[T]):
    _registered_on: List[List['EventHandler[T]']]
    fn: Callable[[T], None]

    def __init__(self, fn: Callable[[T], None]) -> None:
        self._registered_on = []
        self.fn = fn

    def register(self, ev: List['EventHandler[T]']) -> None:
        ev.append(self)
        self._registered_on.append(ev)

    def unregister(self) -> None:
        for ev in self._registered_on:
            ev.remove(self)
        self._registered_on.clear()

    def __call__(self, arg: T) -> None:
        self.fn(arg)


class Scheduler(Generic[T], ABC):
    on_update: List[EventHandler[T]]
    __val: T

    def __init__(self, val: T):
        self.on_update = []
        self.__val = val

    def update(self, new_val: Any) -> None:
        if new_val != self.__val:
            self.__val = new_val
            for handler in self.on_update:
                handler(new_val)

    @property
    def value(self) -> T:
        return self.__val

    def __repr__(self, *args, **kwargs) -> str:
        arg_strs = [
            *(repr(v) for v in args),
            *(k + '=' + repr(v) for k, v in kwargs.items()),
        ]
        return self.__class__.__name__ + '(' + ', '.join(arg_strs) + ')'

    @abstractmethod
    def step_to(self, n: int) -> None:
        pass


class PiecewiseScheduler(Scheduler[T]):
    parts: List[Tuple[int, T]]
    current_part: int

    def __init__(self, parts: List[Tuple[int, T]], **kwargs) -> None:
        parts.sort()
        assert parts[0][0] == 0
        super().__init__(parts[0][1], **kwargs)

        self.parts = parts
        self.current_part = 0

        self.step_to(0)

    def step_to(self, n: int) -> None:
        while n < self.parts[self.current_part][0]:
            self.current_part -= 1
        while self.current_part + 1 < len(self.parts) and self.parts[self.current_part+1][0] <= n:
            self.current_part += 1
        self.update(self.parts[self.current_part][1])

    def __repr__(self) -> str:
        return super().__repr__(self.parts)


class LinearScheduler(Scheduler[float]):
    start: float
    end: float
    agent_timesteps: int

    def __init__(self, start: float, end: float, agent_timesteps: int, **kwargs) -> None:
        super().__init__(start, **kwargs)
        self.start = start
        self.end = end
        self.agent_timesteps = agent_timesteps

        self.step_to(0)

    def step_to(self, n: int) -> None:
        if n >= self.agent_timesteps:
            self.update(self.end)
        else:
            val = ((self.agent_timesteps-n)*self.start + n*self.end) / self.agent_timesteps
            self.update(val)

    def __repr__(self):
        return super().__repr__(self.start, self.end, self.agent_timesteps)


class ExponentialScheduler(Scheduler[float]):
    initial_value: float
    multiplier: float
    min_value: Optional[float]
    max_value: Optional[float]

    def __init__(self, initial_value: float, multiplier: float,
                 min_value: float = None, max_value: float = None, **kwargs):
        super().__init__(initial_value, **kwargs)
        self.initial_value = initial_value
        self.multiplier = multiplier
        self.min_value = min_value
        self.max_value = max_value

        self.step_to(0)

    def step_to(self, n: int) -> None:
        val = self.initial_value * self.multiplier**n
        if self.min_value is not None or self.max_value is not None:
            val = np.clip(val, self.min_value, self.max_value)  # type: ignore
        self.update(val)

    def __repr__(self) -> str:
        kwargs = {}
        if self.min_value is not None:
            kwargs['min_value'] = self.min_value
        if self.max_value is not None:
            kwargs['max_value'] = self.max_value
        return super().__repr__(self.initial_value, self.multiplier, **kwargs)
