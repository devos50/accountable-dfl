import asyncio
import json
import pickle
from asyncio import ensure_future
from typing import List, Tuple

from accdfl.core.models import serialize_chunk, serialize_model
from accdfl.dfl.community import DFLCommunity
from accdfl.util.eva.result import TransferResult
from simulations.bandwidth_scheduler import BWScheduler


class DFLBypassNetworkCommunity(DFLCommunity):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = None
        self.transfers: List[Tuple[str, str, int, float, float, str, bool]] = []

        self.bw_scheduler: BWScheduler = BWScheduler(self.my_peer.public_key.key_to_bin(),
                                                     self.peer_manager.get_my_short_id())
        
    async def eva_send_chunk(self, round: int, chunk_idx: int, chunk, peer, in_sample: bool = True):
        binary_data = serialize_chunk(chunk)
        response = {"round": round, "idx": chunk_idx, "type": "chunk", "in_sample": in_sample}
        return await self.eva_send_data(binary_data, response, peer)

    async def eva_send_model(self, round, model, type, population_view, peer):
        serialized_model = serialize_model(model)
        serialized_population_view = pickle.dumps(population_view)
        self.bw_out_stats["bytes"]["model"] += len(serialized_model)
        self.bw_out_stats["bytes"]["view"] += len(serialized_population_view)
        self.bw_out_stats["num"]["model"] += 1
        self.bw_out_stats["num"]["view"] += 1
        binary_data = serialized_model + serialized_population_view
        response = {"round": round, "type": type, "model_data_len": len(serialized_model)}

        return await self.eva_send_data(binary_data, response, peer)

    async def eva_send_data(self, binary_data: bytes, response: bytes, peer):
        serialized_response = json.dumps(response).encode()
        found: bool = False
        transfer_success: bool = True
        transfer_time: float = 0
        for node in self.nodes:
            if node.overlays[0].my_peer == peer:
                found = True
                if not node.overlays[0].is_active:
                    break

                transfer_start_time = asyncio.get_event_loop().time()
                if self.bw_scheduler.bw_limit > 0:
                    transfer_size: int = len(binary_data) + len(serialized_response)
                    transfer = self.bw_scheduler.add_transfer(node.overlays[0].bw_scheduler, transfer_size)
                    transfer.metadata = response
                    self.logger.info("Model transfer %s => %s started at t=%f",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     transfer_start_time)
                    try:
                        await transfer.complete_future
                    except RuntimeError:
                        transfer_success = False
                    transfer_time = asyncio.get_event_loop().time() - transfer_start_time

                    transferred_bytes: int = transfer.get_transferred_bytes()
                    self.endpoint.bytes_up += transferred_bytes
                    node.overlays[0].endpoint.bytes_down += transferred_bytes

                    self.logger.info("Model transfer %s => %s %s at t=%f and took %f s.",
                                     self.peer_manager.get_my_short_id(),
                                     node.overlays[0].peer_manager.get_my_short_id(),
                                     "completed" if transfer_success else "failed",
                                     transfer_start_time, transfer_time)
                else:
                    self.endpoint.bytes_up += len(binary_data) + len(serialized_response)
                    node.overlays[0].endpoint.bytes_down += len(binary_data) + len(serialized_response)

                json_data = json.loads(serialized_response.decode())
                self.transfers.append((self.peer_manager.get_my_short_id(),
                                       node.overlays[0].peer_manager.get_my_short_id(), json_data["round"],
                                       transfer_start_time, transfer_time, json_data["type"], transfer_success))

                if transfer_success:
                    res = TransferResult(self.my_peer, serialized_response, binary_data, 0)
                    ensure_future(node.overlays[0].on_receive(res))
                break

        if not found:
            raise RuntimeError("Peer %s not found in node list!" % peer)

        return transfer_success

    def go_offline(self, graceful: bool = True) -> None:
        super().go_offline(graceful=graceful)
        self.bw_scheduler.kill_all_transfers()
