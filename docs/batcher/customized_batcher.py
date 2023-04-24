# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple

import numpy as np

from batch_inference import ModelHost
from batch_inference.batcher import Batcher


class MyModel:
    def __init__(self):
        self.op = np.matmul

    # x.shape: [batch_size, m, k], y.shape: [batch_size, k, n]
    def predict_batch(self, x, y):
        res = self.op(x, y)
        return res


class MyBatcher(Batcher):
    def __init__(self):
        super().__init__()

    def batch(self, requests: List[Tuple[np.ndarray]]):
        """Batch n requests into 1 batched request

        Args:
            requests: [(x, y)], each request is a (x, y) from predict method

        Returns:
            batched requests: (x_batched, y_batched) for predict_batch method
            context for unbatch: List[int], the batch sizes of each original (x, y)
        """

        x_batched = np.concatenate([item[0] for item in requests], axis=0)
        y_batched = np.concatenate([item[1] for item in requests], axis=0)
        batch_sizes = [item[0].shape[0] for item in requests]
        return (x_batched, y_batched), batch_sizes

    def unbatch(
        self,
        batched_response: np.ndarray,
        unbatch_ctx: List[int],
    ):
        """Unbatch 1 batched response into n responses

        Args:
            batched_responses: batched_res from predict_batch method,
                               batched_res=batched_x * batched_y
            unbatch_ctx: batch_sizes of n original requests

        Returns:
            responses: [res1, res2, ...], each res will be returned by predict method,
                       res=x * y
        """

        batch_sizes = unbatch_ctx
        responses = []
        start = 0
        for n in batch_sizes:
            responses.append(batched_response[start : start + n])
            start += n
        return responses


model_host = ModelHost(
    MyModel,
    batcher=MyBatcher(),
    max_batch_size=32,
)()
