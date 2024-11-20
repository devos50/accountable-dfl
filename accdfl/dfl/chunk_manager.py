import copy
from random import Random
from typing import List

import torch


class ChunkManager:

    def __init__(self, round: int, model, num_chunks: int):
        self.round: int = round
        self.model = model
        self.num_chunks: int = num_chunks
        self.chunks: List = [None] * num_chunks
        self.chunks_received_from_previous_sample: int = 0
        self.step: int = 0

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
        self.chunks[chunk_idx].add_(chunk)
        self.chunks[chunk_idx].div_(2)

    def process_received_chunk_from_previous_sample(self, chunk_idx: int, chunk):
        self.chunks[chunk_idx] = chunk  # TODO doesn't work if we will receive multiple chunks
        self.chunks_received_from_previous_sample += 1

    @staticmethod
    def get_flat_params(model):
        param_tensors = [param.data.view(-1) for param in model.parameters()]
        flat_params = torch.cat(param_tensors)
        return flat_params
    
    def get_random_chunk_to_send(self, rand: Random):
        idx: int = rand.randint(0, len(self.chunks) - 1)
        return idx, self.chunks[idx].clone()
