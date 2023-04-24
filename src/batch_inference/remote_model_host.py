# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import asyncio
import threading
from types import TracebackType
from typing import Any, Optional, Tuple, Type

from .aio.remote_model_host import remote_model_host
from .batcher.batcher import Batcher
from .logger import logger


class RemoteModelHost(remote_model_host.RemoteModelHost):
    def __init__(
        self,
        model_cls: Type,
        grpc_port: int,
        batcher: Batcher,
        max_batch_size=32,
        wait_ms: int = 5,
        wait_n: int = 32,
    ):
        # Event loop is coupled to thread. You cannot share the loop between different threads,
        # except call_soon_threadsafe()/run_coroutine_threadsafe() calls.
        # With sync APIs, we need another dedicated thread to run the event loop.
        # We can't run event loop in the caller thread that instantiates the class
        # and calls the predict API, it will block the caller thread.
        self.event_loop = asyncio.new_event_loop()
        super().__init__(
            model_cls,
            grpc_port=grpc_port,
            batcher=batcher,
            max_batch_size=max_batch_size,
            wait_ms=wait_ms,
            wait_n=wait_n,
            event_loop=self.event_loop,
        )
        self.event_loop_thread = None

    def start(self):
        if self.event_loop_thread:
            raise RuntimeError("Don't call start method twice in remote model host!")

        def run_event_loop():
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
            logger.info(f"event loop execution thread is terminated")

        self.event_loop_thread = threading.Thread(target=run_event_loop)
        self.event_loop_thread.start()
        asyncio.run_coroutine_threadsafe(super().start(), self.event_loop).result()

    def predict(self, *input_list: Tuple) -> Any:
        f = asyncio.run_coroutine_threadsafe(
            super().predict(*input_list),
            self.event_loop,
        )
        return f.result()

    def stop(self):
        asyncio.run_coroutine_threadsafe(super().stop(), self.event_loop).result()
        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        self.event_loop_thread.join()

    def __enter__(self) -> None:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()
