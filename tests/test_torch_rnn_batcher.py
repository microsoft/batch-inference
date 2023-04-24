# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import threading
import unittest
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from batch_inference import batching
from batch_inference.batcher import Batcher


class RnnModel(torch.nn.Module):
    """
    A sample to show batching with pad_packed_sequence for RNN model
    """

    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(1, 8, 1, batch_first=True)
        self.nn = torch.nn.Linear(1, 8)

    def forward(self, input_ids, langs):
        out, h = self.rnn(input_ids)
        out_pad = pad_packed_sequence(out, batch_first=True)
        embed = self.nn(langs)
        label = torch.matmul(h, embed.t())
        return (
            out_pad[0].detach().numpy(),
            out_pad[1].detach().numpy(),
            label.detach().numpy(),
        )


class MyRnnBatcher(Batcher):
    def batch(self, requests: List[Tuple[Any]]) -> Tuple[Tuple[Any], Any]:
        """Batch multiple N requests into 1 batched requests

        Args:
            requests: a list of N requests, each request is a tuple of args from predict method

        Returns:
            batched requests: 1 batched requests, which is a tuple of args for predict_batch method
            context for unbatch: will be passed to unbatch method
        """
        # sort on input_id length before pack_sequence
        combined = sorted(enumerate(requests), key=lambda i: i[1][0].size, reverse=True)
        # save order to unbatch it later
        order = [i[0] for i in combined]
        langs = [i[1][1] for i in combined]

        # concat langs from multiple requests
        batched_langs = np.concatenate(langs, axis=0)
        batched_langs = torch.from_numpy(batched_langs)

        # Apply pack_sequence on input_ids
        input_ids = [i[1][0][0] for i in combined]
        input_ids = [torch.from_numpy(x).unsqueeze(-1) for x in input_ids]
        packed_input_ids = pack_sequence(input_ids, enforce_sorted=True)
        return (packed_input_ids, batched_langs), order

    def unbatch(self, batched_response: Any, unbatch_ctx: Any) -> List:
        """Unbatch 1 batched response into N responses

        Args:
            batched_response: 1 batched responses from predict_batch method
            unbatch_ctx: context from batch method

        Returns:
            responses: a list of N responses, each response will be returned by predict method
        """
        order = unbatch_ctx
        responses = [
            (
                batched_response[0][i : i + 1],
                batched_response[1][i : i + 1],
                batched_response[2][i : i + 1],
            )
            for i in range(0, len(order))
        ]
        # sort to original order
        responses = [responses[i] for i in order]
        return responses


@batching(batcher=MyRnnBatcher())
class BatchedModel:
    def __init__(self):
        self.model = RnnModel()

    def predict_batch(self, input_ids, langs):
        return self.model.forward(input_ids, langs)


class TestModelHost(unittest.TestCase):
    def setUp(self) -> None:
        self.batched_model = BatchedModel.host()
        self.batched_model.start()

    def tearDown(self) -> None:
        self.batched_model.stop()

    def test_simple(self):
        x = np.random.randn(1, 4).astype("f")
        y = np.random.randn(1, 1).astype("f")
        result_batch = self.batched_model.predict(x, y)
        print(result_batch)
        self.assertTrue(result_batch)

    def test_concurrent(self):
        def send_requests():
            for i in range(1, 10):
                x = np.random.randn(1, i).astype("f")
                y = np.random.randn(1, 1).astype("f")
                result_batch = self.batched_model.predict(x, y)
                self.assertTrue(result_batch)

        threads = [threading.Thread(target=send_requests) for i in range(0, 10)]

        for th in threads:
            th.start()

        for th in threads:
            th.join()


if __name__ == "__main__":
    unittest.main()
