import asyncio
import copy
import json
import time
from asyncio import ensure_future
from binascii import unhexlify
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch import nn

from accdfl.core.community import LearningCommunity
from accdfl.core.models import serialize_model, unserialize_model
from accdfl.util.eva.result import TransferResult


class DLCommunity(LearningCommunity):
    community_id = unhexlify('e5889074c1e4c60423cdb6e9307ba0ca5695ead7')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round: int = 1
        self.neighbours: List[bytes] = []  # The PKs of the neighbours we will send our model to
        self.incoming_models: Dict[int, List[Tuple[bytes, nn.Module]]] = defaultdict(list)  # Incoming models per round

    def start(self):
        """
        Start to participate in the training process.
        """
        assert self.did_setup, "Process has not been setup - call setup() first"
        assert self.neighbours, "We need some neighbours"
        self.start_next_round()

    def eva_send_model(self, round, model, peer):
        start_time = asyncio.get_event_loop().time() if self.settings.is_simulation else time.time()
        serialized_model = serialize_model(model)
        response = {"round": round}
        serialized_response = json.dumps(response).encode()
        return self.schedule_eva_send_model(peer, serialized_response, serialized_model, start_time)

    def start_next_round(self):
        self.register_task("round_%d" % self.round, self.do_round)

    async def do_round(self):
        """
        Perform a single round. This method is expected to be called by a global coordinator.
        """
        self.logger.info("Peer %s starting round %d", self.peer_manager.get_my_short_id(), self.round)

        # Train
        await self.model_manager.train()

        my_peer_pk = self.my_peer.public_key.key_to_bin()
        self.incoming_models[self.round].append((my_peer_pk, self.model_manager.model))

        # Send the trained model to your neighbours
        for peer_pk in self.neighbours:
            peer = self.get_peer_by_pk(peer_pk)
            if not peer:
                self.logger.warning("Cannot find Peer object for participant %s!",
                                    self.peer_manager.get_short_id(peer_pk))
                continue

            ensure_future(self.eva_send_model(self.round, self.model_manager.model, peer))

        self.check_round_complete()

    def check_round_complete(self):
        """
        Check whether the round is complete.
        """
        if len(self.incoming_models[self.round]) != len(self.neighbours) + 1:
            return

        # The round is complete - wrap it up and proceed
        incoming_models = self.incoming_models.pop(self.round)
        self.model_manager.incoming_trained_models = dict((x, y) for x, y in incoming_models)

        # Transfer these models back to the CPU to prepare for aggregation
        device = torch.device("cpu")
        for peer_pk in self.model_manager.incoming_trained_models.keys():
            model = self.model_manager.incoming_trained_models[peer_pk]
            self.model_manager.incoming_trained_models[peer_pk] = model.to(device)

        self.model_manager.model = self.model_manager.average_trained_models()
        if self.round_complete_callback:
            ensure_future(self.round_complete_callback(self.round))
        if self.aggregate_complete_callback:
            model_cpy = copy.deepcopy(self.model_manager.model)
            ensure_future(self.aggregate_complete_callback(self.round, model_cpy))
        self.logger.info("Peer %s completed round %d", self.peer_manager.get_my_short_id(), self.round)
        self.round += 1

    async def on_receive(self, result: TransferResult):
        """
        We received a model from a neighbouring peer. Store it and check if we received enough models to proceed.
        """
        peer_pk = result.peer.public_key.key_to_bin()
        peer_id = self.peer_manager.get_short_id(peer_pk)
        my_peer_id = self.peer_manager.get_my_short_id()
        self.logger.info(f'Participant {my_peer_id} received data from participant {peer_id}: {result.info.decode()}')

        json_data = json.loads(result.info.decode())
        incoming_model = unserialize_model(result.data, self.settings.dataset)
        self.incoming_models[json_data["round"]].append((peer_pk, incoming_model))
        self.check_round_complete()
