from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=10)
class AdvertiseMembership:
    round: int
    index: int
    change: int


@dataclass(msg_id=11)
class PingPayload:
    round: int
    index: int
    identifier: int


@dataclass(msg_id=12)
class PongPayload:
    round: int
    index: int
    identifier: int


@dataclass(msg_id=13)
class TrainDonePayload:
    round: int
