# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import threading
import unittest

import numpy as np

from batch_inference import ModelHost
from batch_inference.batcher import ConcatBatcher


class MyModel:
    def __init__(self, weights):
        self.weights = weights

    # x: [batch_size, m, k], self.weights: [k, n]
    def predict_batch(self, x):
        y = np.matmul(x, self.weights)
        return y


class TestModelHost(unittest.TestCase):
    def setUp(self) -> None:
        self.num_workers = 10
        self.weights = np.random.randn(3, 3).astype("f")
        self.model_host = ModelHost(
            MyModel,
            batcher=ConcatBatcher(),
            max_batch_size=self.num_workers,
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

    def test_concurrent(self):
        def send_requests():
            for _ in range(0, 10):
                x = np.random.randn(1, 3, 3).astype("f")
                y = self.model_host.predict(x)
                self.assertTrue(
                    np.allclose(y, np.matmul(x, self.weights), rtol=1e-05, atol=1e-05),
                )

        threads = [
            threading.Thread(target=send_requests) for i in range(0, self.num_workers)
        ]
        [th.start() for th in threads]
        [th.join() for th in threads]


class TestModelHostWithAs(unittest.TestCase):
    def test_withas(self):
        weights = np.random.randn(3, 3).astype("f")
        with ModelHost(
            MyModel,
            batcher=ConcatBatcher(),
            max_batch_size=10,
        )(weights) as model_host:
            x = np.random.randn(1, 3, 3).astype("f")
            y = model_host.predict(x)
            self.assertTrue(
                np.allclose(y, np.matmul(x, weights), rtol=1e-05, atol=1e-05)
            )


if __name__ == "__main__":
    unittest.main()
