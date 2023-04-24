# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from batch_inference import ModelHost
from batch_inference.batcher import ConcatBatcher


class BatchFCModule:
    def __init__(self, input_size, output_size, batch):
        self.batch = batch
        self.weights = torch.randn(input_size, output_size)
        self.host = ModelHost(
            model_obj=self,
            batcher=ConcatBatcher(tensor="pt"),
            max_batch_size=32,
            wait_ms=-1,
            wait_n=8,
        )
        self.compute_time = {}
        self.lock = threading.Lock()

    def predict_batch(self, input):
        start_time = time.perf_counter_ns()
        res = torch.matmul(input, self.weights)
        compute_time = time.perf_counter_ns() - start_time
        # print(f"input size: {input.size()}, compute time: {compute_time:.6f} seconds, thread: {threading.get_ident()}")
        with self.lock:
            self.compute_time[threading.get_ident()] = (
                0
                if threading.get_ident() not in self.compute_time
                else self.compute_time[threading.get_ident()] + compute_time
            )
        return res

    # input: [batch_size, input_size]
    def predict(self, input):
        if self.batch:
            return self.host.predict(input)
        else:
            return self.predict_batch(input)


def main():
    input_size, output_size = 1024, 10240
    sut = BatchFCModule(input_size, output_size, batch=True)
    sut.host.start()

    def request():
        input = torch.rand(2, input_size)
        return sut.predict(input)

    print("Start Running")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(request) for i in range(10000)]
        result = [f.result() for f in as_completed(futures)]
    end_time = time.time()
    sut.host.stop()

    print(f"Total time: {end_time - start_time:.6f} seconds")
    compute_time = {k: v / 1e9 for k, v in sut.compute_time.items()}
    print(f"Compute time ({len(compute_time)}): {compute_time} seconds")


if __name__ == "__main__":
    main()
