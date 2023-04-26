# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import queue
import threading
import time
from types import TracebackType
from typing import Any, List, Optional, Type

from .batch_context import BatchContext
from .batcher.batcher import Batcher
from .batcher.multi_batcher import MultiBatcher
from .logger import logger


class ModelHost:
    def __init__(
        self,
        model_cls=None,
        model_obj=None,
        batcher: Batcher = None,
        max_batch_size=32,
        wait_n=8,
        wait_ms: int = 5,
        num_workers: int = 1,
    ):
        if model_cls is None and model_obj is None:
            raise RuntimeError(f"Either model_cls or model_obj must be provided")
        self.model_cls = model_cls
        self.model_obj = model_obj
        self.batcher = batcher
        self.max_batch_size = max_batch_size
        self.wait_n = wait_n
        if self.wait_n > self.max_batch_size:
            self.wait_n = self.max_batch_size
        
        self.num_workers = num_workers
        self.wait_ms = wait_ms
        self.cycle_time = 1.0
        self.stopped = True
        # double queue size to avoid blocking
        self.batch_queue = [queue.Queue(maxsize=2 * max_batch_size)] * num_workers
        self.threads = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.model_obj is not None:
            raise RuntimeError(
                f"model_obj has already been set, you can't call __call__() to create it again",
            )
        # pass the original arguments and constructor the original model instance.
        self.model_obj = self.model_cls(*args, **kwargs)
        return self

    def predict(self, *input_list):
        if self.stopped:
            raise RuntimeError(f"model host is stopped, can't predict anymore")
        batch_ctx = BatchContext(input_list)
        idx = batch_ctx.__hash__() % self.num_workers
        self.batch_queue[idx].put(batch_ctx)
        return batch_ctx.get_response()

    def start(self):
        if not self.stopped:
            raise RuntimeError(f"model host is already started")

        self.stopped = False
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._wait_batch_ready_and_process, args=(i,)
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stopped = True
        logger.debug(f"notify worker threads to stop")
        for thread in self.threads:
            thread.join()
        logger.debug(f"all worker threads are stopped")

    def _get_new_batch(self, idx) -> List[BatchContext]:
        # blocking until get at least one request
        try:
            batch_list: List[BatchContext] = [
                self.batch_queue[idx].get(block=True, timeout=self.cycle_time)
            ]
        except queue.Empty:
            return []

        # get more already arrived requests
        while (
            len(batch_list) < self.max_batch_size and not self.batch_queue[idx].empty()
        ):
            try:
                batch_list.append(self.batch_queue[idx].get(block=False))
            except queue.Empty:
                break

        # check if we still need to wait
        if self.wait_ms <= 0 or len(batch_list) >= self.wait_n:
            return batch_list

        # wait for `wait_ms` ms to see if there's more requests to batch
        current_time = time.perf_counter()
        end_time = current_time + self.wait_ms / 1000.0
        while end_time - current_time > 0 and len(batch_list) < self.wait_n:
            try:
                batch_list.append(
                    self.batch_queue[idx].get(timeout=end_time - current_time)
                )
                current_time = time.perf_counter()
            except queue.Empty:
                break
        return batch_list

    def _wait_batch_ready_and_process(self, idx):
        while not self.stopped:
            batch_list = self._get_new_batch(idx)
            if len(batch_list) == 0:
                continue
            # logger.info(f"get batch of size {len(batch_list)}")
            try:
                requests = [batch_ctx.request for batch_ctx in batch_list]
                batched_requests, unbatch_ctx = self.batcher.batch(requests)
                if issubclass(type(self.batcher), MultiBatcher):
                    batched_responses = []
                    for i in batched_requests:
                        batched_responses.append(self.model_obj.predict_batch(*i))
                else:
                    batched_responses = self.model_obj.predict_batch(*batched_requests)
                responses = self.batcher.unbatch(
                    batched_responses,
                    unbatch_ctx,
                )
                for i, ctx in enumerate(batch_list):
                    ctx.set_response(responses[i])
            except Exception as e:
                for ctx in batch_list:
                    ctx.set_error(e)
                logger.error(e, exc_info=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()
