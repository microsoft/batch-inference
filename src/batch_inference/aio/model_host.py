# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import asyncio
import sys
import threading
from types import TracebackType
from typing import Any, Optional, Type

from ..batcher.batcher import Batcher
from ..batcher.multi_batcher import MultiBatcher
from ..logger import logger
from .batch_context import BatchContext


class ModelHost:
    def __init__(
        self,
        model_cls=None,
        model_obj=None,
        batcher: Batcher = None,
        max_batch_size=32,
        wait_ms: int = 5,
        wait_n: int = 16,
        num_workers: int = 1,
        event_loop=None,
    ):
        if model_cls is None and model_obj is None:
            raise RuntimeError(f"Either model_cls or model_obj must be provided")
        self.model_cls = model_cls
        self.model_obj = model_obj
        self.batcher = batcher
        self.max_batch_size = max_batch_size
        # wait for `wait_ms` ms to see if there's more requests to batch
        self.wait_ms = wait_ms
        # during wait for `wait_ms` ms, if there's more than `wait_n` requests,
        # we stop waiting and start to process the batch immediately
        self.wait_n = wait_n
        if self.wait_n > self.max_batch_size:
            self.wait_n = self.max_batch_size

        self.event_loop = event_loop
        self.cv = None
        self.batch_queue = []
        self.num_workers = num_workers
        self.threads = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.model_obj is not None:
            raise RuntimeError(
                f"model_obj has already been set, you can't call __call__() to create it again",
            )
        # pass the originl arguments and construtor the origial model instance.
        self.model_obj = self.model_cls(*args, **kwds)
        return self

    async def predict(self, *input_list):
        async with self.cv:
            is_first_request = len(self.batch_queue) == 0
            if is_first_request or self.batch_queue[-1].size() >= self.max_batch_size:
                self.batch_queue.append(BatchContext())
            #     logger.debug(f'I am the first query of a new batch')
            # else:
            #     logger.debug(f'I am added to an existing batch')
            last_batch: BatchContext = self.batch_queue[-1]
            idx = last_batch.add_request(input_list)
            self.cv.notify(n=1)

        await last_batch.result_ready.wait()
        if last_batch.error is not None:
            raise last_batch.error
        return last_batch.responses[idx]

    async def start(self):
        if self.event_loop is None:
            self.event_loop = asyncio.get_event_loop()

        if sys.version_info >= (3, 10):
            batch_queue_lock = asyncio.Lock()
            self.cv = asyncio.Condition(lock=batch_queue_lock)
        else:
            batch_queue_lock = asyncio.Lock(loop=self.event_loop)
            self.cv = asyncio.Condition(lock=batch_queue_lock, loop=self.event_loop)
        for _ in range(self.num_workers):
            thread = threading.Thread(target=self._wait_batch_ready_and_process)
            thread.start()
            self.threads.append(thread)

    async def stop(self):
        async with self.cv:
            self.batch_queue.append(None)
            self.cv.notify_all()
        logger.debug(f"notify worker threads to stop")
        for thread in self.threads:
            await self.event_loop.run_in_executor(None, thread.join)
        logger.debug(f"all worker threads are stopped")

    async def _get_new_batch(self):
        # WorkerLoop will be notified when there's a new request
        # take first BatchContext out of quueue
        early_ret_n = self.wait_n
        wait_timeout = -1
        if self.wait_ms > 0:
            wait_timeout = self.wait_ms / 1000.0  # in seconds

        async with self.cv:
            await self.cv.wait_for(lambda: len(self.batch_queue) > 0)
            try:
                await asyncio.wait_for(
                    self.cv.wait_for(
                        lambda: self.batch_queue[0]
                        is None  # should exit, no more request
                        or len(self.batch_queue[0].requests) >= early_ret_n,
                    ),
                    wait_timeout,
                )
            except asyncio.TimeoutError:
                # logger.info(f'wait batch size to reach {early_ret_n} timeout, actual batch size={len(self.batch_queue[0].requests)}')
                pass
            
            if self.batch_queue[0] is None:
                return None
            else:
                return self.batch_queue.pop(0)

    def _wait_batch_ready_and_process(self):
        while True:
            f = asyncio.run_coroutine_threadsafe(self._get_new_batch(), self.event_loop)
            batch_ctx = f.result()

            if batch_ctx is None:
                return

            # logger.info(f"get batch of size {len(batch_ctx.requests)}")

            try:
                batched_requests, unbatch_ctx = self.batcher.batch(batch_ctx.requests)
                if issubclass(type(self.batcher), MultiBatcher):
                    batched_responses = []
                    for i in batched_requests:
                        batched_responses.append(self.model_obj.predict_batch(*i))
                else:
                    batched_responses = self.model_obj.predict_batch(*batched_requests)
                batch_ctx.responses = self.batcher.unbatch(
                    batched_responses,
                    unbatch_ctx,
                )
                self.event_loop.call_soon_threadsafe(batch_ctx.set_result_ready)
            except Exception as e:
                self.event_loop.call_soon_threadsafe(
                    lambda: batch_ctx.set_error(error=e),
                )
                logger.error(e, exc_info=True)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.stop()
