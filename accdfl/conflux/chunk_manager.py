import copy
from math import ceil
from random import Random
from typing import Dict, List

import torch


class ChunkManager:

    def __init__(self, round: int, model, num_peers: int, num_chunks: int, success_fraction: float):
        self.round: int = round
        self.model = model
        self.num_peers: int = num_peers
        self.num_chunks: int = num_chunks
        self.success_fraction: float = success_fraction
        self.chunks: List = []
        for _ in range(num_chunks):
            self.chunks.append([])

        self.received_chunks: List = []
        for _ in range(num_chunks):
            self.received_chunks.append([])

    def prepare(self):
        # Chunk
        flat_params = ChunkManager.get_flat_params(self.model)
        total_elements = flat_params.numel()
        chunk_size = total_elements // self.num_chunks
        self.chunks = [flat_params[i * chunk_size: (i + 1) * chunk_size] for i in range(self.num_chunks)]

        # Handle any remaining elements
        if total_elements % self.num_chunks != 0:
            remaining = flat_params[self.num_chunks * chunk_size:]
            self.chunks[-1] = torch.cat([self.chunks[-1], remaining])

    def get_aggregated_model(self):
        for idx in range(len(self.chunks)):
            assert self.received_chunks[idx], "No chunks received at index %d!" % idx

        self.aggregate_received_chunks()

        # Reconstruct the flat tensor
        flat_params = torch.cat(self.chunks)

        # Copy the flat tensor into the model
        pointer = 0
        model_cpy = copy.deepcopy(self.model)
        for param in model_cpy.parameters():
            numel = param.data.numel()
            param_shape = param.data.shape
            param.data.copy_(flat_params[pointer:pointer + numel].view(param_shape))
            pointer += numel

        return model_cpy

    def process_received_chunk(self, chunk_idx: int, chunk):
        self.received_chunks[chunk_idx].append(chunk)

    def aggregate_received_chunks(self):
        for chunk_idx, chunks in enumerate(self.received_chunks):
            self.chunks[chunk_idx] = torch.mean(torch.stack(chunks), dim=0)
        self.received_chunks = None

    def has_received_enough_chunks(self):
        return all([(len(chunks) / self.num_peers) >= self.success_fraction for chunks in self.received_chunks])

    @staticmethod
    def get_flat_params(model):
        param_tensors = [param.data.view(-1) for param in model.parameters()]
        flat_params = torch.cat(param_tensors)
        return flat_params
