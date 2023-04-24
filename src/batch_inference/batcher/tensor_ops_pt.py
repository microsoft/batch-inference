# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

try:
    import torch
except:
    pass


class TensorOps:
    @staticmethod
    def concat(tensors):
        return torch.cat(tensors, dim=0)

    @staticmethod
    def slice(tensor, start, end):
        return tensor[start:end]

    @staticmethod
    def size(tensor, dim: int):
        return tensor.size(dim=dim)

    @staticmethod
    def pad(tensor, pad, value):
        return torch.nn.functional.pad(tensor, pad, "constant", value)
