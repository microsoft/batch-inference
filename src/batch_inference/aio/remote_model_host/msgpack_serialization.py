# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pickle

import msgpack_numpy

# Batch Inference Toolkit doesn't depend on PyTorch to run, The try...catch... is to
# make sure the code works no matter PyTorch is installed or not.
try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


def encode(obj):
    if has_torch and isinstance(obj, torch.Tensor):
        return {"torch.Tensor": True, "data": pickle.dumps(obj)}
    else:
        return msgpack_numpy.encode(obj)


def decode(obj):
    if "torch.Tensor" in obj:
        return pickle.loads(obj["data"])
    else:
        return msgpack_numpy.decode(obj)
