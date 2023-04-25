# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import asyncio
import sys
import threading
import time
from types import TracebackType
from typing import Any, Optional, Type, List

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
            batch_ctx = BatchContext(input_list)
            self.batch_queue.append(batch_ctx)
            self.cv.notify(n=1)
        response = await batch_ctx.get_response()
        return response

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
        batch_list = []
        
        # take first BatchContext out of queue
        async with self.cv:          
            await self.cv.wait_for(lambda: len(self.batch_queue) > 0)
                           
            # get already arrived requests
            while (
                len(batch_list) < self.max_batch_size and len(self.batch_queue) > 0
            ):
                if self.batch_queue[0] is None: # end of processing
                    return batch_list
                batch_list.append(self.batch_queue.pop(0))
            
            # check if we still need to wait
            if self.wait_ms <= 0 or len(batch_list) >= self.wait_n:
                return batch_list
            
            # wait for `wait_ms` ms to see if there's more requests to batch
            current_time = time.perf_counter()
            end_time = current_time + self.wait_ms / 1000.0
            while current_time < end_time and len(batch_list) < self.wait_n:
                try:
                    await asyncio.wait_for(
                        self.cv.wait_for(lambda: len(self.batch_queue) > 0),
                        end_time - current_time,
                    )
                    if self.batch_queue[0] is None: # end of processing
                        return batch_list
                    batch_list.append(self.batch_queue.pop(0))
                    current_time = time.perf_counter()
                except asyncio.TimeoutError:
                    # logger.info(f'wait batch size to reach {self.wait_n} timeout, actual batch size={len(batch_list)}')
                    break      
        return batch_list

    def _wait_batch_ready_and_process(self):
        while True:
            batch_list = asyncio.run_coroutine_threadsafe(self._get_new_batch(), self.event_loop).result()
            if len(batch_list) == 0:
                return
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
