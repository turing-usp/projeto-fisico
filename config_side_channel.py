from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
import uuid


FIELDS_PER_TYPE = {
    OutgoingMessage.write_int32:   ['NumHazard', 'MinHazardSpeed', 'MaxHazardSpeed', 'AgentCount'],
    OutgoingMessage.write_float32: [],
    OutgoingMessage.write_string:  [],
}

FIELDS = {name.lower(): writer
          for writer, names in FIELDS_PER_TYPE.items()
          for name in names}


class ConfigSideChannel(SideChannel):

    def __init__(self, **kwargs) -> None:
        super().__init__(uuid.UUID("3e7c67af-6e4d-446d-b318-0beb6546e274"))
        for k, v in kwargs.items():
            self.set(k, v)

    def on_message_received(self, msg) -> None:
        print('ConfigSideChannel received an unexpected message:', msg)

    def set(self, key, value) -> None:
        writer = FIELDS.get(key.lower(), None)
        if not writer:
            raise ValueError(f'Invalid key: {key}')

        msg = OutgoingMessage()
        msg.write_string(key)
        writer(msg, value)
        self.queue_message_to_send(msg)
