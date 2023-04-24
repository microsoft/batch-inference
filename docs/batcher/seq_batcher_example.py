# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from batch_inference import ModelHost
from batch_inference.batcher import SeqBatcher


class MyModel:
    def __init__(self):
        pass

    # input: [batch_size, n]
    def predict_batch(self, input):
        res = input
        return res


model_host = ModelHost(
    MyModel,
    batcher=SeqBatcher(),
    max_batch_size=32,
)()
