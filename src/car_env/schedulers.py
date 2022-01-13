"""Define os schedulers, que permitem o uso de parâmetros variáveis."""

from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from abc import abstractmethod, ABC


T = TypeVar('T')


class Scheduler(Generic[T], ABC):
    """Permite que ambientes tenham parâmetros variáveis.

    Um :class:`Scheduler` guarda um valor que varia com o tempo (ou, mais
    precisamente, com o número de steps).

    O valor, accesível em :func:`value` é atualizado sempre que
    :func:`step_to` ou :func:`update` forem chamados.
    Quando o scheduler for atualizado, o evento :attr:`on_update`
    é chamado com o novo valor (c.f. :class:`EventHandler`).

    Args:
        val: O valor inicial.

    Attributes:
        on_update: Lista de callables que são chamadas sempre que o valor
            desse scheduler for atualizado. Essa lista pode ser modificada
            diretamente por usuários com os métodos ``.append()`` e ``.remove()``.
    """
    on_update: List[Callable[[T], None]]
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
        """O valor atual desse scheduler"""
        return self.__val

    def __repr__(self, *args, **kwargs) -> str:
        arg_strs = [
            *(repr(v) for v in args),
            *(k + '=' + repr(v) for k, v in kwargs.items()),
        ]
        return self.__class__.__name__ + '(' + ', '.join(arg_strs) + ')'

    @abstractmethod
    def step_to(self, n: int) -> None:
        """Atualiza o valor do scheduler para o step ``n``.

        Args:
            n: O número atual de steps **de agente**.
        """
        pass


class PiecewiseScheduler(Scheduler[T]):
    """Um scheduler com um conjunto fixo de valores.

    O valor desse scheduler é determinado a partir do parâmetro ``parts``,
    que indica quais valores devem ser assumidos pelo scheduler,
    e em quais momentos esses valores devem ser assumidos.

    Arguments:
        parts: Uma lista de tuplas ``(n, val)`` ordenada por ``n``.
            Cada tupla indica que, no step ``n``, o valor do scheduler
            deve ser atualizado para ``val``, A primeira tupla deve
            **obrigatoriamente** ter ``n=0``.

    Attributes:
        parts: Uma lista de tuplas ``(n, val)``, conforme descrito acima.
        current_parts: O índice da tupla atual em :attr:`parts`.
    """

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
    """Um scheduler cujo valor varia linearmente no tempo.

    O valor desse scheduler é uma interpolação entre o valor inicial
    (:attr:`start`) em `n = 0` e o valor final (:attr:`end`) em `n = agent_timesteps`.
    Para `n >= agent_timesteps`, o valor é constante e igual a :attr:`end`.

    Arguments:
        start: O valor inicial do scheduler.
        end: O valor final do scheduler.
        agent_timesteps: O número de steps após os quais o scheduler atinge
            o valor final.

    Attributes:
        start: O valor inicial do scheduler.
        end: O valor final do scheduler.
        agent_timesteps: O número de steps após os quais o scheduler atinge
            o valor final.
    """
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
    """Um scheduler cujo valor varia exponencialmente no tempo.

    O valor desse scheduler é igual a ``value = initial_value * multiplier**n``.

    O valor é clipado para garantir que ele esteja sempre entre
    :attr:`min_value` e :attr:`max_value`, inclusivo.

    Arguments:
        initial_value: O valor inicial do scheduler.
        multiplier: O valor pelo qual ``scheduler.value`` é multiplicado
            a cada step.
        min_value: O valor mínimo do scheduler.
        max_value: O valor máximo do scheduler.

    Attributes:
        initial_value: O valor inicial do scheduler.
        multiplier: O valor pelo qual ``scheduler.value`` é multiplicado
            a cada step.
        min_value: O valor mínimo do scheduler.
        max_value: O valor máximo do scheduler.
    """
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
