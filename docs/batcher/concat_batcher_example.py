# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from batch_inference import ModelHost
from batch_inference.batcher import ConcatBatcher


class MyModel:
    def __init__(self):
        self.op = np.matmul

    # x.shape: [batch_size, m, k], y.shape: [batch_size, k, n]
    def predict_batch(self, x, y):
        res = self.op(x, y)
        return res


model_host = ModelHost(
    MyModel,
    batcher=ConcatBatcher(),
    max_batch_size=32,
)()
