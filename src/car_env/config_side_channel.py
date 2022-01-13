"""Permite que as configurações do :class:`~.core.CarEnv` sejam enviadas ao Unity."""

from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, Type, Union

from mlagents_envs.side_channel.incoming_message import IncomingMessage
from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
import uuid
import ray

from .schedulers import Scheduler

Value = Union[int, float, str, bool]
"""Ver ``ConfigSideChannel.Value``.

:meta private:
"""

FIELDS: Dict[str, Tuple[Type, Value]] = {
    # Agent configs
    'AgentCount': (int, 1),
    'AgentDecisionPeriod': (int, 20),
    'AgentRaysPerDirection': (int, 3),
    'AgentRayLength': (int, 64),
    'AgentCheckpointTTL': (float, 60.0),
    'AgentCheckpointMax': (int, 0),

    # Chunk configs
    'ChunkDifficulty': (int, 0),
    'ChunkMinAgentsBeforeDestruction': (int, 0),
    'ChunkDelayBeforeDestruction': (float, 5.0),
    'ChunkTTL': (float, 30.0),

    # Hazard configs
    'HazardCountPerChunk': (int, 2),
    'HazardMinSpeed': (float, 1.0),
    'HazardMaxSpeed': (float, 10.0),

    # Car configs
    'CarStrenghtCoefficient': (float, 20_000.0),
    'CarBrakeStrength': (float, 200.0),
    'CarMaxTurn': (float, 20.0),

    # Time configs
    'TimeScale': (float, 1.0),
    'FixedDeltaTime': (float, 0.04),
}
"""Lista as configurações do :class:`~.core.CarEnv` que podem ser passadas ao Unity.

O dicionário ``FIELDS`` associa cada campo ``field_name`` a uma tupla
``(typ, default) == FIELDS[field_name]``, onde ``typ`` é o tipo do campo
de tipo :attr:`~ConfigSideChannel.Value` (``int``, ``float``, ``str`` ou ``bool``)
e ``default`` é o seu valor inicial.

.. rubric:: Configurações dos agentes

Attributes:
    'AgentCount':
        Número de agentes (carros).

        :Tipo, valor padrão: int, padrão=1
        :Momento de efeito: no reset do ambiente

    'AgentDecisionPeriod':
        Número de timesteps antes que o agente solicite uma nova ação.

        :Tipo, valor padrão: int, padrão=20
        :Momento de efeito: no reset do agente

    'AgentRaysPerDirection':
        Número de raios por direção (esquerda/direita) usados
        para criar as observações.

        Obs: ``número total de raios = 2*AgentRaysPerDirection + 1``
        (raios na esquerda, na direita e um raio central).

        :Tipo, valor padrão: int, padrão=3
        :Momento de efeito: no reset do agente

    'AgentCheckpointTTL':
        Tempo máximo (em segundos) entre dois checkpoints.

        Após esse período, o agente é morto.

        :Tipo, valor padrão: float, padrão=60.0
        :Momento de efeito: imediatamente

    'AgentCheckpointMax':
        Número máximo de checkpoints. Quando um agente atinge essenúmero
        de checkpoints, ele é automaticamente resetado (sem penalizações).

        Se zero, nenhum máximo é imposto.

        :Tipo, valor padrão: int, padrão=0
        :Momento de efeito: no próximo checkpoint

.. rubric:: Configurações das chunks

Attributes:
    'ChunkDifficulty':
        Dificuldade das chunks geradas (a partir de 0,
        em ordem crescente de dificuldade).

        :Tipo, valor padrão: int, padrão=0
        :Momento de efeito: no reset do agente

    'ChunkMinAgentsBeforeDestruction':
        Número de agentes que precisam completar uma chunk antes que ela
        possa ser destruída.

        Se zero, espera até que todos os agentes tenham completado a chunk.

        :Tipo, valor padrão: int, padrão=0
        :Momento de efeito: na criação da chunk

    'ChunkTTL':
        O tempo máximo (em segundos) antes que uma chunk seja
        automaticamente destruída, contado a partir do momento em que
        o primeiro carro chega nessa chunk.

        Se zero, espera até que todos os agentes tenham completado a chunk.

        :Tipo, valor padrão: float, padrão=30.0
        :Momento de efeito: quando o primeiro carro chega na chunk

    'ChunkDelayBeforeDestruction':
        Delay (em segundos) antes que uma chunk seja destruída, após ter
        sido completada por ``ChunkMinAgentsBeforeDestruction`` agentes
        ou após o ``ChunkTTL`` ter sido excedido. Essa configuração
        é necessária porque quando um agente completa uma chunk, um pedaço
        do carro ainda se encontra dentro daquela chunk.

        :Tipo, valor padrão: float, padrão=5.0
        :Momento de efeito: imediatamente

.. rubric:: Configurações dos obstáculos

Attributes:
    'HazardCountPerChunk':
        Número de obstáculos por chunk.

        :Tipo, valor padrão: int, padrão=2
        :Momento de efeito: na criação da chunk

    'HazardMinSpeed':
        Velocidade mínima dos obstáculos, em unidades por segundo.

        Cada obstáculo é criado com uma velocidade fixa no intervalo
        ``[HazardMinSpeed, HazardMaxSpeed]``.

        :Tipo, valor padrão: float, padrão=1.0
        :Momento de efeito: na criação da chunk

    'HazardMaxSpeed':
        Velocidade máxima dos obstáculos, em unidades por segundo.

        Cada obstáculo é criado com uma velocidade fixa no intervalo
        ``[HazardMinSpeed, HazardMaxSpeed]``.

        :Tipo, valor padrão: float, padrão=10.0
        :Momento de efeito: na criação da chunk

.. rubric:: Configurações dos carros

Attributes:

    'CarStrenghtCoefficient':
        :Tipo, valor padrão: float, padrão=20_000.0
        :Momento de efeito: no reset do agente

    'CarBrakeStrength':
        :Tipo, valor padrão: float, padrão=200.0
        :Momento de efeito: no reset do agente

    'CarMaxTurn':
        :Tipo, valor padrão: float, padrão=20.0
        :Momento de efeito: no reset do agente

.. rubric:: Configurações de tempo

Attributes:

    'TimeScale':
        Controla a velocidade da simulação.
        O Unity tenta simular ``TimeScale`` segundos a cada segundo,
        i.e., garantir que ``TempoSimulado = TempoReal * TimeScale``.

        Obs: para valores muito elevados desse parâmetro, o Unity pode não ser
        capaz de simular o jogo na velocidade exigida
        (``TempoSimulado < TempoReal * TimeScale``).

        :Tipo, valor padrão: float, padrão=1.0
        :Momento de efeito: imediatamente

    'FixedDeltaTime'
        Tempo entre updates (``FixedUpdates``) do Unity.
        Valores menores permitem uma simulação mais precisa dos fenômenos físicos,
        mas demandam mais poder computacional.

        Obs: esse parâmetro é independente de ``TimeScale``, i.e.,
        o tempo considerado é o tempo simulado, e não o tempo real.

        :Tipo, valor padrão: float, padrão=0.04
        :Momento de efeito: imediatamente

:meta hide-value:

"""

_MessageWriter = Callable[[OutgoingMessage, Any], None]
"""Tipo que aceita qualquer uma da funções ``OutgoingMessage.write_*```."""

_MESSAGE_WRITERS: Dict[Type, _MessageWriter] = {
    int: OutgoingMessage.write_int32,
    float: OutgoingMessage.write_float32,
    str: OutgoingMessage.write_string,
    bool: OutgoingMessage.write_bool,
}

_FIELD_WRITERS: Dict[str, _MessageWriter] = {
    field_name.lower(): _MESSAGE_WRITERS[typ]
    for (field_name, (typ, default)) in FIELDS.items()
}
"""Os :class:`MessageWriter`s associados à cada campo, sempre em lower case."""


class ConfigSideChannel(SideChannel):
    """Permite que as configurações do :class:`~.core.CarEnv` sejam enviadas ao Unity.

    As configurações existentes podem ser vistas em :attr:`FIELDS`.

    Args:
        **kwargs: configurações iniciais.
    """

    Value = Value
    """Todos os tipos possíveis para as configurações utilizadas pelo ambiente."""

    _param_setters: Dict[str, Callable[[Value], None]]
    _schedulers: Dict[str, Scheduler]

    def __init__(self, **kwargs) -> None:
        super().__init__(uuid.UUID("3e7c67af-6e4d-446d-b318-0beb6546e274"))
        self._param_setters = {}
        self._schedulers = {}

        self.set_defaults()
        for k, v in kwargs.items():
            self.set(k, v)

    def _get_or_create_setter(self, key: str) -> Callable[[Value], None]:
        key_lower = key.lower()

        if setter := self._param_setters.get(key_lower):
            return setter

        try:
            writer = _FIELD_WRITERS[key_lower]
        except KeyError:
            raise ValueError(f'Invalid key: {key}')

        def new_setter(val): return self._set(writer, key_lower, val)
        self._param_setters[key_lower] = new_setter
        return new_setter

    def on_message_received(self, msg: IncomingMessage) -> None:
        """Recebe mensagens enviadas pelo Unity.

        Esse método nunca deve ser chamado (a comunicação desse side channel
        é unidirecional).

        Args:
            msg: A mensagem.

        :meta private:
        """
        print('ConfigSideChannel received an unexpected message:', msg)

    def set(self, key: str, value: Union[Value, Scheduler]) -> None:
        """Atualiza o valor de um parâmetro.

        Os parâmetros disponíveis, assim como os seus tipos e valores padrões
        podem ser vistos em :attr:`FIELDS`.

        Args:
            key: O nome do parâmetro.
            value: O valor do parâmetro, ou um :class:`Scheduler` apropriado.
        """
        setter = self._get_or_create_setter(key)
        key_lower = key.lower()

        if old_scheduler := self._schedulers.pop(key_lower, None):
            old_scheduler.on_update.remove(setter)

        if isinstance(value, Scheduler):
            self._schedulers[key_lower] = value
            setter(value.value)
            value.on_update.append(setter)
        else:
            setter(value)

    def _set(self, writer: _MessageWriter, key: str, value: Value) -> None:
        logger = ray.get_actor('param_logger')
        logger.update_param.remote('unity_config/' + key, value)

        msg = OutgoingMessage()
        msg.write_string(key)
        writer(msg, value)
        self.queue_message_to_send(msg)

    def set_defaults(self) -> None:
        """Atribui os valores padrões a todos os parâmetros.

        Os valores padrões são aqueles listados em :attr:`FIELDS`.
        """
        for field_name, (typ, default) in FIELDS.items():
            self.set(field_name, default)
