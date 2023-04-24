# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import asyncio
import threading
import time

import torch

from batch_inference.aio import ModelHost
from batch_inference.batcher import ConcatBatcher


class BatchFCModule:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size)
        self.host = ModelHost(
            model_obj=self,
            batcher=ConcatBatcher(tensor="pt"),
            max_batch_size=32,
            wait_ms=-1,
            wait_n=8,
        )
        self.compute_time = {}

    def predict_batch(self, input):
        start_time = time.perf_counter_ns()
        res = torch.matmul(input, self.weights)
        compute_time = time.perf_counter_ns() - start_time
        # print(f"input size: {input.size()}, compute time: {compute_time:.6f} seconds, thread: {threading.get_ident()}")
        self.compute_time[threading.get_ident()] = (
            0
            if threading.get_ident() not in self.compute_time
            else self.compute_time[threading.get_ident()] + compute_time
        )
        return res

    async def predict(self, input):
        await self.host.predict(input)


async def main():
    input_size, output_size = 1024, 10240
    sut = BatchFCModule(input_size, output_size)
    await sut.host.start()

    async def request():
        input = torch.randn(2, input_size)
        await sut.predict(input)

    print("Start Running")
    start_time = time.time()
    tasks = [asyncio.create_task(request()) for i in range(10000)]
    await asyncio.wait(tasks)
    end_time = time.time()
    await sut.host.stop()

    print(f"Total time: {end_time - start_time:.6f} seconds")
    compute_time = {k: v / 1e9 for k, v in sut.compute_time.items()}
    print(f"Compute time ({len(compute_time)}): {compute_time} seconds")


if __name__ == "__main__":
    # cProfile.run('asyncio.run(main())', 'fc_stats.cprofile')
    # p = pstats.Stats('fc_stats.cprofile')
    # p.strip_dirs().sort_stats(SortKey.TIME).print_stats()
    asyncio.run(main())
