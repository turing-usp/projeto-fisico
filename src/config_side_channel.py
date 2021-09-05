from schedulers import Scheduler
from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
import uuid
import ray

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

    # Configuration for the velocity reward bonus.
    # Given the forward velocity magnitude |v|, the checkpoint reward is
    # MaxReward * Clamp(|v| / VelocityForMaxReward, 0, 1).
    # Takes effect: immediately
    # Default:
    #   AgentCheckpointReward_MaxReward: 100
    #   AgentCheckpointReward_VelocityForMaxReward: 5
    'AgentCheckpointReward_MaxReward': float,
    'AgentCheckpointReward_VelocityForMaxReward': float,

    # Reward given when an agent dies
    # Takes effect: immediately
    # Default: -100
    'AgentDeathReward': float,

    # Configuration for the velocity reward bonus.
    # Given the forward velocity v, clamped to (MinVelocity, MaxVelocity),
    # the cumulative bonus per second is w*v,
    # where w grows linearly from zero to CoeffPerSecond over the WarmupTime
    # (in seconds) and is reset to zero whenever sign(v) changes.
    # Takes effect: immediately
    # Default:
    #   MaxVelocity: 1
    #   MinVelocity: -10
    #   WarmupTime: 10
    #   CoeffPerSecond: 0 (disabled)
    'AgentVelocityBonus_MaxVelocity': float,
    'AgentVelocityBonus_MinVelocity': float,
    'AgentVelocityBonus_WarmupTime': float,
    'AgentVelocityBonus_CoeffPerSecond': float,


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
        self._scheduler_removers = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def on_message_received(self, msg) -> None:
        print('ConfigSideChannel received an unexpected message:', msg)

    def set(self, key, value) -> None:
        key_lower = key.lower()
        writer = FIELD_WRITERS.get(key_lower, None)
        if not writer:
            raise ValueError(f'Invalid key: {key}')

        self._scheduler_removers.pop(key_lower, lambda: None)()

        if isinstance(value, Scheduler):
            sch = value
            self._add_handler(key_lower, sch.on_update,
                              lambda: self._set(writer, key, sch.value))
            value = sch.value
        self._set(writer, key, value)

    def _add_handler(self, key_lower, ev, h):
        ev.add(h)
        self._scheduler_removers[key_lower] = lambda: ev.remove(h)

    def _set(self, writer, key, value):
        logger = ray.get_actor('config_logger')
        logger.update.remote('unity_config/' + key, value)

        msg = OutgoingMessage()
        msg.write_string(key)
        writer(msg, value)
        self.queue_message_to_send(msg)
