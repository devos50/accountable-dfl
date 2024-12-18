from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=11)
class PingPayload:
    identifier: int


@dataclass(msg_id=12)
class PongPayload:
    identifier: int


@dataclass(msg_id=13)
class HasEnoughChunksPayload:
    round: int
