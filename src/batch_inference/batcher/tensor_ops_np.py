# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy


class TensorOps:
    @staticmethod
    def concat(tensors):
        return numpy.concatenate(tensors, axis=0)

    @staticmethod
    def slice(tensor, start, end):
        return tensor[start:end]

    @staticmethod
    def size(tensor, dim: int):
        return tensor.shape[dim]

    @staticmethod
    def pad(tensor, pad, value):
        pad_width = [(0, 0)] * (tensor.ndim - 1) + [pad]
        return numpy.pad(tensor, pad_width, "constant", constant_values=value)
