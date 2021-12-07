from typing import Dict
from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
import uuid
import ray

from .schedulers import EventHandler, Scheduler

FIELDS = {
    #######################
    #    AGENT CONFIGS    #
    #######################

    # Number of agents (cars).
    # Takes effect: on environment reset
    # Default: 1
    'AgentCount': int,

    # The agent requests an action every AgentDecisionPeriod timesteps.
    # Takes effect: on agent reset
    # Default: 20
    'AgentDecisionPeriod': int,

    # Number of rays per direction used to create the observations.
    # Note: total number of rays = 2*AgentRaysPerDirection + 1
    # Takes effect: on agent reset
    # Default: 3
    'AgentRaysPerDirection': int,

    # Length of the rays.
    # Takes effect: on agent reset
    # Default: 64
    'AgentRayLength': int,

    # Maximum time (in seconds) between two checkpoints.
    # If an agent stays longer than this without passing a checkpoint, it is killed.
    # Takes effect: immediately
    # Default: 60
    'AgentCheckpointTTL': float,

    # Maximum number of checkpoints.
    # When an agent reaches this number of checkpoints, it is automatically reset.
    # If zero, no maximum is enforced.
    # Takes effect: on agent checkpoint
    # Default: 0
    'AgentCheckpointMax': int,


    #######################
    #    CHUNK CONFIGS    #
    #######################

    # Difficulty of the chunks used (sequential, starting from zero)
    # Takes effect: on chunk creation
    # Default: 0
    'ChunkDifficulty': int,

    # The number of agents required to pass a chunk before it's destroyed.
    # If zero, wait until all agents have passed.
    # Takes effect: on chunk creation
    # Default: 0
    'ChunkMinAgentsBeforeDestruction': int,

    # Delay (in seconds) before a chunk is destroyed when at least ChunkMinAgentsBeforeDestruction
    # have passed it.
    # Necessary because when an agent "passes" a chunk, a part of the car is
    # still in that chunk.
    # Takes effect: immediately
    # Default: 5
    'ChunkDelayBeforeDestruction': float,

    # The maximum time (in seconds) before a chunk is automatically destroyed.
    # If zero, wait until all agents have passed.
    # Takes effect: when a chunk first sees a car
    # Default: 30
    'ChunkTTL': float,


    ########################
    #    HAZARD CONFIGS    #
    ########################

    # Number of hazards spawned per chunk.
    # Takes effect: on chunk creation
    # Default: 2
    'HazardCountPerChunk': int,

    # Minimum hazard speed
    # Takes effect: on chunk creation
    # Default: 1
    'HazardMinSpeed': float,

    # Maximum hazard speed
    # Takes effect: on chunk creation
    # Default: 10
    'HazardMaxSpeed': float,


    ###############################
    #    CAR CONTROLER CONFIGS    #
    ###############################

    # Takes effect: on agent reset
    # Default: 20_000
    'CarStrenghtCoefficient': float,

    # Takes effect: on agent reset
    # Default: 200
    'CarBrakeStrength': float,

    # Takes effect: on agent reset
    # Default: 20
    'CarMaxTurn': float,


    ######################
    #    TIME CONFIGS    #
    ######################

    # Controls the simulation speed.
    # e.g. TimeScale=1 for 1 simulation second  / real second
    #  and TimeScale=2 for 2 simulation seconds / real second
    # Takes effect: immediately
    # Default: 1
    'TimeScale': float,

    # Time between unity FixedUpdate calls.
    # Smaller is more accurate, but more computationally intensive
    # Takes effect: immediately
    # Default: 0.04
    'FixedDeltaTime': float,
}

MESSAGE_WRITERS = {
    int: OutgoingMessage.write_int32,
    float: OutgoingMessage.write_float32,
    str: OutgoingMessage.write_string,
    bool: OutgoingMessage.write_bool,
}

FIELD_WRITERS = {
    field_name.lower(): MESSAGE_WRITERS[typ]
    for (field_name, typ) in FIELDS.items()
}


class ConfigSideChannel(SideChannel):

    def __init__(self, **kwargs) -> None:
        super().__init__(uuid.UUID("3e7c67af-6e4d-446d-b318-0beb6546e274"))
        self._handlers: Dict[str, EventHandler] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def handler(self, key: str) -> EventHandler:
        key = key.lower()

        h = self._handlers.pop(key, None)
        if h is not None:
            return h

        writer = FIELD_WRITERS.get(key, None)
        if not writer:
            raise ValueError(f'Invalid key: {key}')
        self._handlers[key] = EventHandler(lambda val: self._set(writer, key, val))
        return self._handlers[key]

    def on_message_received(self, msg) -> None:
        print('ConfigSideChannel received an unexpected message:', msg)

    def set(self, key, value) -> None:
        h = self.handler(key)

        h.unregister()
        h(value)
        if isinstance(value, Scheduler):
            h.register(value.on_update)

    def _set(self, writer, key, value):
        logger = ray.get_actor('param_logger')
        logger.update_param.remote('unity_config/' + key, value)

        msg = OutgoingMessage()
        msg.write_string(key)
        writer(msg, value)
        self.queue_message_to_send(msg)
