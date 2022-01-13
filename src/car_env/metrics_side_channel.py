"""Permite que as métricas coletadas no Unity sejam recebidas em python."""

from mlagents_envs.side_channel.incoming_message import IncomingMessage
from mlagents_envs.side_channel.side_channel import SideChannel
import uuid

from . import actors


class MetricsSideChannel(SideChannel):
    """Permite que as métricas coletadas no Unity sejam recebidas em python.

    As métricas recebidas são salvas no :func:`.actors.agent_metric_tracker`.
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID("7a9acce6-3f8f-47cd-adf5-a45956fdbd71"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        """Recebe mensagens enviadas pelo Unity.

        Cada mensagem contém uma string (o nome da métrica), seguida do valor
        da métrica (um float).

        Args:
            msg: A mensagem.
        """
        name = msg.read_string()
        value = msg.read_float32()
        tracker = actors.agent_metric_tracker()
        tracker.add_metric.remote(name, value)
