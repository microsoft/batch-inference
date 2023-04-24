# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import threading
import unittest
from typing import List

import torch

from batch_inference import ModelHost, RemoteModelHost
from batch_inference.batcher import ConcatBatcher


class MyTorchModel:
    def __init__(self):
        self.weights = torch.ones(3, 3)

    def predict_batch(self, x):
        res = torch.matmul(x, self.weights)
        return res


class TestModelHost(unittest.TestCase):
    def setUp(self) -> None:
        self.model_host = ModelHost(
            MyTorchModel, batcher=ConcatBatcher(tensor="pt"), max_batch_size=32
        )()
        self.model_host.start()

    def tearDown(self) -> None:
        self.model_host.stop()

    def test_simple(self):
        x = torch.randn(1, 3, 3)
        res = self.model_host.predict(x)
        expected_res = torch.matmul(x, torch.ones(3, 3))
        self.assertTrue(torch.allclose(res, expected_res, rtol=1e-05, atol=1e-05))

    def test_concurrent(self):
        def send_requests():
            for _ in range(0, 10):
                x = torch.randn(1, 3, 3)
                res = self.model_host.predict(x)
                self.assertTrue(
                    torch.allclose(
                        res, torch.matmul(x, torch.ones(3, 3)), rtol=1e-05, atol=1e-05
                    ),
                )

        threads = [threading.Thread(target=send_requests) for i in range(0, 10)]
        [th.start() for th in threads]
        [th.join() for th in threads]


class TestRemoteModelHostConcurrent(unittest.TestCase):
    def setUp(self) -> None:
        self.num_workers = 2
        self.hosts: List[RemoteModelHost] = []
        for _ in range(self.num_workers):
            self.hosts.append(
                RemoteModelHost(
                    MyTorchModel,
                    grpc_port=23333,
                    batcher=ConcatBatcher(tensor="pt"),
                    max_batch_size=32,
                )(),
            )
        [host.start() for host in self.hosts]

    def tearDown(self) -> None:
        [host.stop() for host in self.hosts]

    def test_concurrent(self):
        def send_requests(i):
            for _ in range(0, 100):
                x = torch.randn(1, 3, 3)
                res = self.hosts[i].predict(x)
                expected_res = torch.matmul(x, torch.ones(3, 3))
                self.assertTrue(
                    torch.allclose(res, expected_res, rtol=1e-05, atol=1e-05),
                )

        threads = [
            threading.Thread(target=send_requests, args=(i,))
            for i in range(0, self.num_workers)
        ]
        [th.start() for th in threads]
        [th.join() for th in threads]


if __name__ == "__main__":
    unittest.main()
