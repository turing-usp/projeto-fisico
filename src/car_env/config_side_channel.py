from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, Type, Union

from mlagents_envs.side_channel.incoming_message import IncomingMessage
from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
import uuid
import ray

from .schedulers import Scheduler

Value = Union[int, float, str, bool]

FIELDS: Dict[str, Tuple[Type, Value]] = {
    #######################
    #    AGENT CONFIGS    #
    #######################

    # Number of agents (cars).
    # Takes effect: on environment reset
    # Default: 1
    'AgentCount': (int, 1),

    # The agent requests an action every AgentDecisionPeriod timesteps.
    # Takes effect: on agent reset
    # Default: 20
    'AgentDecisionPeriod': (int, 20),

    # Number of rays per direction used to create the observations.
    # Note: total number of rays = 2*AgentRaysPerDirection + 1
    # Takes effect: on agent reset
    # Default: 3
    'AgentRaysPerDirection': (int, 3),

    # Length of the rays.
    # Takes effect: on agent reset
    # Default: 64
    'AgentRayLength': (int, 64),

    # Maximum time (in seconds) between two checkpoints.
    # If an agent stays longer than this without passing a checkpoint, it is killed.
    # Takes effect: immediately
    # Default: 60
    'AgentCheckpointTTL': (float, 60.0),

    # Maximum number of checkpoints.
    # When an agent reaches this number of checkpoints, it is automatically reset.
    # If zero, no maximum is enforced.
    # Takes effect: on agent checkpoint
    # Default: 0
    'AgentCheckpointMax': (int, 0),


    #######################
    #    CHUNK CONFIGS    #
    #######################

    # Difficulty of the chunks used (sequential, starting from zero)
    # Takes effect: on chunk creation
    # Default: 0
    'ChunkDifficulty': (int, 0),

    # The number of agents required to pass a chunk before it's destroyed.
    # If zero, wait until all agents have passed.
    # Takes effect: on chunk creation
    # Default: 0
    'ChunkMinAgentsBeforeDestruction': (int, 0),

    # Delay (in seconds) before a chunk is destroyed when at least ChunkMinAgentsBeforeDestruction
    # have passed it.
    # Necessary because when an agent "passes" a chunk, a part of the car is
    # still in that chunk.
    # Takes effect: immediately
    # Default: 5
    'ChunkDelayBeforeDestruction': (float, 5.0),

    # The maximum time (in seconds) before a chunk is automatically destroyed.
    # If zero, wait until all agents have passed.
    # Takes effect: when a chunk first sees a car
    # Default: 30
    'ChunkTTL': (float, 30.0),


    ########################
    #    HAZARD CONFIGS    #
    ########################

    # Number of hazards spawned per chunk.
    # Takes effect: on chunk creation
    # Default: 2
    'HazardCountPerChunk': (int, 2),

    # Minimum hazard speed
    # Takes effect: on chunk creation
    # Default: 1
    'HazardMinSpeed': (float, 1.0),

    # Maximum hazard speed
    # Takes effect: on chunk creation
    # Default: 10
    'HazardMaxSpeed': (float, 10.0),


    ###############################
    #    CAR CONTROLER CONFIGS    #
    ###############################

    # Takes effect: on agent reset
    # Default: 20_000
    'CarStrenghtCoefficient': (float, 20_000.0),

    # Takes effect: on agent reset
    # Default: 200
    'CarBrakeStrength': (float, 200.0),

    # Takes effect: on agent reset
    # Default: 20
    'CarMaxTurn': (float, 20.0),


    ######################
    #    TIME CONFIGS    #
    ######################

    # Controls the simulation speed.
    # e.g. TimeScale=1 for 1 simulation second  / real second
    #  and TimeScale=2 for 2 simulation seconds / real second
    # Takes effect: immediately
    # Default: 1
    'TimeScale': (float, 1.0),

    # Time between unity FixedUpdate calls.
    # Smaller is more accurate, but more computationally intensive
    # Takes effect: immediately
    # Default: 0.04
    'FixedDeltaTime': (float, 0.04),
}


_MessageWriter = Callable[[OutgoingMessage, Any], None]
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


class ConfigSideChannel(SideChannel):
    Value = Value

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
        print('ConfigSideChannel received an unexpected message:', msg)

    def set(self, key: str, value: Union[Value, Scheduler]) -> None:
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
        for field_name, (typ, default) in FIELDS.items():
            self.set(field_name, default)
