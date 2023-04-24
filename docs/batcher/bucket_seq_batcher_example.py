# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from batch_inference import ModelHost
from batch_inference.batcher import BucketSeqBatcher


class MyModel:
    def __init__(self):
        pass

    # input: [batch_size, n]
    def predict_batch(self, seq):
        res = seq
        return res


model_host = ModelHost(
    MyModel,
    batcher=BucketSeqBatcher(padding_tokens=[0, 0], buckets=[1024, 2048, 4096]),
    max_batch_size=32,
)()
