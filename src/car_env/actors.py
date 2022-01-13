"""Define os atores do rllib essenciais para o funcionamento do :class:`~car_env.core.CarEnv`.


Os atores podem ser obtidos por meio das funções :func:`agent_step_counter`,
:func:`param_logger` e :func:`agent_metric_tracker`. As funções suportadas
podem ser vistas na documentação das respectivas classes
(:class:`AgentStepCounter`, :class:`ParamLogger`, ...).

Note:
    Para utilizar os atores, é necessário ter chamado a função
    :func:`car_env.actors.init` (ou, equivalentemente, :func:`car_env.init`).

O mecanismo para chamar um método num actor é um pouco diferente do
mecanismo para métodos normais::

    actor = agent_step_counter()

    # Chamando funções de um actor
    actor.add_agent_steps.remote(10)  # análogo a add_agent_steps(10)

    # Utilizando o resultado de funções de um actor
    steps_total, steps_this_phase = ray.get(actor.get_steps.remote())  # análogo a add_agent_steps(10)

Para mais informações, veja a `documentação do rllib <https://docs.ray.io/en/latest/actors.html>`_.
"""
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import ray
import ray.actor
from collections import defaultdict


_AGENT_STEP_COUNTER = 'agent_step_counter'
_PARAM_LOGGER = 'param_logger'
_AGENT_METRIC_TRACKER = 'agent_metric_tracker'

_actors = []


def init() -> None:
    """Inicializa os atores utilizados pelo :class:`~car_env.core.CarEnv`."""

    # Store handles to the actors in order to keep them alive
    _actors.extend([
        ray.remote(AgentStepCounter).options(name=_AGENT_STEP_COUNTER).remote(),
        ray.remote(ParamLogger).options(name=_PARAM_LOGGER).remote(),
        ray.remote(AgentMetricTracker).options(name=_AGENT_METRIC_TRACKER).remote(),
    ])

    # NOTE: we use ray.remote here instead of as a decorator on the actor classes
    # in order to allow for proper documentation to be generated


def _get_actor(name: str) -> ray.actor.ActorHandle:
    """Equivalente a :func:`ray.get_actor`, mas com uma mensagem de erro especializada.

    Essa função só deve ser chamada os atores definidos em :func:`init`.

    Raises:
        RuntimeError: Se o ator não existir, i.e., se a função
            :func:`init` não tiver sido chamada.
    """
    try:
        return ray.get_actor(name)
    except ValueError:
        raise RuntimeError(f'No such agent: {name}. Did you forget to call car_env.actors.init() in the main thread?')


def agent_step_counter() -> ray.actor.ActorHandle:
    """Retorna o ator correspondente à classe :class:`AgentStepCounter`.

    O objeto retornado por essa função é um :class:`ray.actor.ActorHandle`
    e deve ser utilizado conforme descrito na documentação de :mod:`car_env.actors`.

    Raises:
        RuntimeError: Se o ator não existir, i.e., se a função
            :func:`init` não tiver sido chamada.
    """
    return _get_actor(_AGENT_STEP_COUNTER)


def param_logger() -> ray.actor.ActorHandle:
    """Retorna o ator correspondente à classe :class:`ParamLogger`.

    O objeto retornado por essa função é um :class:`ray.actor.ActorHandle`
    e deve ser utilizado conforme descrito na documentação de :mod:`car_env.actors`.

    Raises:
        RuntimeError: Se o ator não existir, i.e., se a função
            :func:`init` não tiver sido chamada.
    """
    return _get_actor(_PARAM_LOGGER)


def agent_metric_tracker() -> ray.actor.ActorHandle:
    """Retorna o ator correspondente à classe :class:`AgentMetricTracker`.

    O objeto retornado por essa função é um :class:`ray.actor.ActorHandle`
    e deve ser utilizado conforme descrito na documentação de :mod:`car_env.actors`.

    Raises:
        RuntimeError: Se o ator não existir, i.e., se a função
            :func:`init` não tiver sido chamada.
    """
    return _get_actor(_AGENT_METRIC_TRACKER)


class AgentStepCounter:
    """Conta os steps de agente tomados num ambiente multi-agente.

    Ambientes multi-agente têm dois tipos de steps:

        - os steps de ambiente (environment steps), que contam o número de vezes
          que a função ``env.step`` foi chamada; e
        - os steps de agente (agent steps), que contam o número de steps que
          foram tomados individualmente por cada agente.

    Por exemplo, a expressão ``env.step({'agent1': ..., 'agent2': ..., 'agent3:': ...})``
    corresponde a 1 environment step, mas 3 agent steps, já que o step foi tomado
    por 3 agentes.

    Dentre essas métricas, apenas os steps de ambiente são salvos automaticamente pelo rllib.
    Essa classe permite que sejam contados também o número de steps de agente.

    Attributes:
        agent_steps_total: O número total de steps de agente.
        agent_steps_this_phase: O número de steps de agente na fase atual
            do currículo de treinamento.

    Note:
        Essa classe não é, geralmente, utilizada diretamente, mas sim por meio
        do actor retornado por :func:`agent_step_counter`.
        Para mais detalhes, veja a documentação dessa função e do módulo :mod:`car_env.actors`.
    """

    agent_steps_total: int
    agent_steps_this_phase: int

    def __init__(self) -> None:
        self.agent_steps_total = 0
        self.agent_steps_this_phase = 0

    def add_agent_steps(self, n: int) -> None:
        """Aumenta o número de steps de agente por ``n``.

        Args:
            n: O número de novos steps de agente.
        """
        self.agent_steps_total += n
        self.agent_steps_this_phase += n

    def get_steps(self) -> Tuple[int, int]:
        """Retorna ``(self.agent_steps_total, self.agent_steps_this_phase)``."""
        return self.agent_steps_total, self.agent_steps_this_phase

    def new_phase(self) -> None:
        """Inicia uma nova fase.

        Essa função reseta o valor de ``agent_steps_this_phase``
        """
        self.agent_steps_this_phase = 0


class ParamLogger:
    """Guarda e permite acesso aos parâmetros de um ambiente.

    Attributes:
        config: Os parâmetros atuais.

    Note:
        Essa classe não é, geralmente, utilizada diretamente, mas sim por meio
        do actor retornado por :func:`param_logger`.
        Para mais detalhes, veja a documentação dessa função e do módulo :mod:`car_env.actors`.
    """

    configs: Dict[str, Any]

    def __init__(self) -> None:
        self.configs = {}

    def update_param(self, key: str, value: Any) -> None:
        """Salva um parâmetro novo ou atualiza um parâmetro já existente.

        Args:
            key: O nome do parâmetro.
            value: O valor do parâmetro.
        """
        if key not in self.configs or self.configs[key] != value:
            print(f'Setting {key}={value}')
            self.configs[key] = value

    def get_params(self) -> Dict[str, Any]:
        """Retorna os parâmetros atuais."""
        return self.configs


class AgentMetricTracker:
    """Guarda e permite acesso às métricas de um ambiente.

    Attributes:
        metrics: Os valores atuais das métricas.

    Note:
        Essa classe não é, geralmente, utilizada diretamente, mas sim por meio
        do actor retornado por :func:`agent_metric_tracker`.
        Para mais detalhes, veja a documentação dessa função e do módulo :mod:`car_env.actors`.
    """

    metrics: DefaultDict[str, List[Union[int, float]]]

    def __init__(self) -> None:
        self.metrics = defaultdict(list)

    def add_metric(self, key: str, value: Union[int, float]) -> None:
        """Adiciona um valor para a métrica especificada.

        Cada métrica está associada a uma lista de valores, de forma
        que essa função apenas adiciona um novo valor nessa lista
        (em particular, essa função não sobrescreve valores anteriores).

        Args:
            key: O nome da métrica.
            value: O valor da métrica.
        """
        self.metrics[key].append(value)

    def get_metrics(self, reset=True) -> DefaultDict[str, List[Union[int, float]]]:
        """Retorna todas as métricas e valores que foram registrados.

        Args:
            reset: Se ``True``, faz com que todas as métricas armazenadas
                pelo objeto sejam resetadas.
                Valor padrão: True.

        Returns:
            Um dicionário mapeando cada métrica para uma lista de valores dessa métrica.
        """
        metrics = self.metrics.copy()
        if reset:
            self.metrics.clear()
        return metrics
