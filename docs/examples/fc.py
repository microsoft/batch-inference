# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from benchmark import *

from batch_inference import ModelHost, aio
from batch_inference.batcher import ConcatBatcher


class BatchFCModule:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size)

    # input: [batch_size, input_size]
    def predict_batch(self, input):
        return torch.matmul(input, self.weights)


if __name__ == "__main__":
    input_size, output_size = 1024, 10240
    requests = [(torch.rand(2, input_size),)]
    
    with ModelHost(
            model_cls=BatchFCModule,
            batcher=ConcatBatcher(tensor="pt"),
            max_batch_size=32,
            wait_ms=1,
            wait_n=8,
            num_workers=4
        )(input_size, output_size) as host:
        benchmark_sync(host, requests, num_calls=10000, max_workers=16)
        
    async def run_async():
        async with aio.ModelHost(
            model_cls=BatchFCModule,
            batcher=ConcatBatcher(tensor="pt"),
            max_batch_size=32,
            wait_ms=1,
            wait_n=8,
            num_workers=4)(input_size, output_size) as host:
            await benchmark_async(host, requests, num_calls=10000)
    asyncio.run(run_async())
    
    # no batching
    sut = BatchFCModule(input_size, output_size)  
    benchmark(sut, requests, num_calls=1000, max_workers=4)
