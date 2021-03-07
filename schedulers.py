from abc import abstractmethod, ABC
from event import Event


class Scheduler(ABC):
    def __init__(self):
        self.on_step = Event()

    def step(self):
        self._step()
        self.on_step()

    @abstractmethod
    def _step(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass


class LinearScheduler(Scheduler):
        super().__init__()
        self.n = 0
        self.start = start
        self.end = end
        self.num_episodes = num_episodes

    def _step(self):
        self.n += 1

    @property
    def value(self):
        if self.n > self.num_episodes:
            return self.end
        a = self.n / self.num_episodes
        return (1-a)*self.start + a*self.end


class ExponentialScheduler(Scheduler):
        super().__init__()
        self._value = initial_value
        self.min_value = min_value
        self.decay = decay

    @property
    def value(self):
        return self._value

    def _step(self):
        super().step()
        if self._value >= 0:
            self._value = max(self._value*self.decay, self.min_value)
        else:
            self._value = min(self._value*self.decay, -self.min_value)
