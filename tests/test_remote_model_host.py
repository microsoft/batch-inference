# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import threading
import unittest
from typing import List

import numpy as np

from batch_inference import RemoteModelHost
from batch_inference.batcher import ConcatBatcher


class MyModel:
    def __init__(self, weights):
        self.weights = weights

    # x: [batch_size, m, k], self.weights: [k, n]
    def predict_batch(self, x):
        y = np.matmul(x, self.weights)
        return y


class TestRemoteModelHostSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = np.random.randn(3, 3).astype("f")
        self.model_host: RemoteModelHost = RemoteModelHost(
            MyModel,
            grpc_port=23333,
            batcher=ConcatBatcher(),
            max_batch_size=32,
        )(self.weights)
        self.model_host.start()

    def tearDown(self) -> None:
        self.model_host.stop()

    def test_simple(self):
        x = np.random.randn(1, 3, 3).astype("f")
        y = self.model_host.predict(x)
        print(y)
        self.assertTrue(
            np.allclose(y, np.matmul(x, self.weights), rtol=1e-05, atol=1e-05)
        )


class TestRemoteModelHostConcurrent(unittest.TestCase):
    def setUp(self) -> None:
        self.num_workers = 10
        self.hosts: List[RemoteModelHost] = []
        self.weights = np.random.randn(3, 3).astype("f")
        for _ in range(self.num_workers):
            self.hosts.append(
                RemoteModelHost(
                    MyModel,
                    grpc_port=23333,
                    batcher=ConcatBatcher(),
                    max_batch_size=self.num_workers,
                )(self.weights),
            )
        [host.start() for host in self.hosts]

    def tearDown(self) -> None:
        [host.stop() for host in self.hosts]

    def test_concurrent(self):
        def send_requests(i):
            for _ in range(0, 100):
                x = np.random.randn(1, 3, 3).astype("f")
                y = self.hosts[i].predict(x)
                self.assertTrue(
                    np.allclose(y, np.matmul(x, self.weights), rtol=1e-05, atol=1e-05),
                )

        threads = [
            threading.Thread(target=send_requests, args=(i,))
            for i in range(0, self.num_workers)
        ]
        [th.start() for th in threads]
        [th.join() for th in threads]


if __name__ == "__main__":
    unittest.main()
