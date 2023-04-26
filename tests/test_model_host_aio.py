# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import asyncio
import multiprocessing

import numpy as np

from batch_inference import aio
from batch_inference.batcher import ConcatBatcher


class MyModel:
    def __init__(self, weights):
        self.weights = weights

    # x: [batch_size, m, k], self.weights: [k, n]
    def predict_batch(self, x):
        y = np.matmul(x, self.weights)
        return y


class TestModelHost(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.weights = np.random.randn(3, 3).astype("f")
        self.model_host = aio.ModelHost(
            MyModel,
            batcher=ConcatBatcher(),
            max_batch_size=32,
            wait_ms=10,
            wait_n=16,
            num_workers=multiprocessing.cpu_count()
        )(self.weights)
        await self.model_host.start()

    async def asyncTearDown(self) -> None:
        await self.model_host.stop()

    async def test_simple(self):
        x = np.random.randn(1, 3, 3).astype("f")
        y = await self.model_host.predict(x)
        print(y)
        self.assertTrue(
            np.allclose(y, np.matmul(x, self.weights), rtol=1e-05, atol=1e-05)
        )

    async def test_concurrent(self):
        
        async def request(i):
            x = np.random.randn(1, 3, 3).astype("f")
            y = await self.model_host.predict(x)
            self.assertTrue(
                np.allclose(y, np.matmul(x, self.weights), rtol=1e-05, atol=1e-05),
            )

        tasks = [asyncio.create_task(request(i)) for i in range(1000)]
        await asyncio.wait(tasks)


class TestModelHostWithAs(unittest.IsolatedAsyncioTestCase):
    async def test_withas(self):
        weights = np.random.randn(3, 3).astype("f")
        async with aio.ModelHost(
            MyModel,
            batcher=ConcatBatcher(),
            max_batch_size=10,
        )(weights) as model_host:
            x = np.random.randn(1, 3, 3).astype("f")
            y = await model_host.predict(x)
            self.assertTrue(
                np.allclose(y, np.matmul(x, weights), rtol=1e-05, atol=1e-05)
            )


if __name__ == "__main__":
    unittest.main()
