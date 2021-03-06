from abc import abstractmethod, ABC


class Scheduler(ABC):
    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def __float__(self):
        return float(self.value)


class LinearScheduler(Scheduler):
    def __init__(self, start, end, num_episodes):
        self.n = 0
        self.start = start
        self.end = end
        self.num_episodes = num_episodes

    def step(self):
        self.n += 1

    @property
    def value(self):
        if self.n > self.num_episodes:
            return self.end
        a = self.n / self.num_episodes
        return (1-a)*self.start + a*self.end


class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value, decay, min_value=0):
        self._value = initial_value
        self.min_value = min_value
        self.decay = decay

    @property
    def value(self):
        return self._value

    def step(self):
        if self._value >= 0:
            self._value = max(self._value*self.decay, self.min_value)
        else:
            self._value = min(self._value*self.decay, -self.min_value)
