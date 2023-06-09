# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import threading
import time
import functools
from types import MethodType
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def add_timer_to_model(model_obj):
    model_obj.lock = threading.Lock()
    model_obj.compute_times = defaultdict(float)
    model_obj.predit_batch_origin = model_obj.predict_batch
    def predit_batch_with_timer(self, *args, **kwargs):
        compute_start_time = time.perf_counter()      
        res = self.predit_batch_origin(*args, **kwargs)
        compute_end_time = time.perf_counter()
        with self.lock:
            self.compute_times[threading.get_ident()] += (compute_end_time - compute_start_time)
        return res     
    model_obj.predict_batch = MethodType(predit_batch_with_timer, model_obj)


def benchmark_sync(host, queries, num_calls, parallel, warm_up_calls=0):
    def request(i):
        q = queries[i % len(queries)]
        host.predict(*q)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # rum warm up queries
        futures = [executor.submit(request, i) for i in range(warm_up_calls)]
        result = [f.result() for f in as_completed(futures)]
        if hasattr(host.model_obj, "reset_counters"):
            host.model_obj.reset_counters()
        print("Start Running")
        add_timer_to_model(host.model_obj)
        start_time = time.time()
        futures = [executor.submit(request, i) for i in range(num_calls)]
        result = [f.result() for f in as_completed(futures)]
        end_time = time.time()
    print(f"Total time: {end_time - start_time:.6f} seconds")
    compute_times = {k: round(v, 6) for k, v in host.model_obj.compute_times.items()}
    print(f"Compute time ({len(host.model_obj.compute_times)}): {compute_times} seconds")


async def benchmark_async(host, queries, num_calls, warm_up_calls=0):
    async def request(i):
        q = queries[i % len(queries)]
        await host.predict(*q)

    # rum warm up queries
    tasks = [asyncio.create_task(request(i)) for i in range(warm_up_calls)]
    await asyncio.wait(tasks)
    if hasattr(host.model_obj, "reset_counters"):
        host.model_obj.reset_counters()
    print("Start Running")
    add_timer_to_model(host.model_obj)
    start_time = time.time()
    tasks = [asyncio.create_task(request(i)) for i in range(num_calls)]
    await asyncio.wait(tasks)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.6f} seconds")
    compute_times = {k: round(v, 6) for k, v in host.model_obj.compute_times.items()}
    print(f"Compute time ({len(host.model_obj.compute_times)}): {compute_times} seconds")


def benchmark(model_obj, queries, num_calls, parallel, warm_up_calls=0):  
    def request(i):
        q = queries[i % len(queries)]
        model_obj.predict_batch(*q)

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # rum warm up queries
        futures = [executor.submit(request, i) for i in range(warm_up_calls)]
        result = [f.result() for f in as_completed(futures)]
        if hasattr(model_obj, "reset_counters"):
            model_obj.reset_counters()
        print("Start Running")
        add_timer_to_model(model_obj)
        start_time = time.time()
        futures = [executor.submit(request, i) for i in range(num_calls)]
        result = [f.result() for f in as_completed(futures)]
        end_time = time.time()
    print(f"Total time: {end_time - start_time:.6f} seconds")
    compute_times = {k: round(v, 6) for k, v in model_obj.compute_times.items()}
    print(f"Compute time ({len(model_obj.compute_times)}): {compute_times} seconds")