# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import threading
from types import TracebackType
from typing import Any, Optional, Type

from filelock import FileLock, Timeout

from ...batcher.batcher import Batcher
from ...logger import logger
from .model_host_client import ModelHostClient
from .model_host_server import ModelHostServer


class RemoteModelHost:
    def __init__(
        self,
        model_cls,
        grpc_port,
        batcher: Batcher,
        max_batch_size=32,
        wait_ms: int = 5,
        wait_n: int = 16,
        num_workers: int = 1,
        event_loop=None,
    ):
        file_lock = self._get_server_file_lock(grpc_port)
        self.is_server = file_lock is not None
        if self.is_server:
            self.processor = ModelHostServer(
                file_lock,
                model_cls,
                grpc_port,
                batcher,
                max_batch_size=max_batch_size,
                wait_ms=wait_ms,
                wait_n=wait_n,
                num_workers=num_workers,
                event_loop=event_loop,
            )
        else:
            self.processor = ModelHostClient(grpc_port)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.is_server:
            self.processor(*args, **kwds)
        return self

    async def start(self):
        await self.processor.start()

    async def stop(self):
        await self.processor.stop()

    async def predict(self, *input_list):
        return await self.processor.predict(*input_list)

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

    def _get_server_file_lock(self, grpc_port):
        # need file lock to ensure that only one RemoteModelHost is the server.
        # In case the file does not exist, a+ is needed to ensure that the file lock works. r+ will throw an exception.
        # don't unlock it until:
        # 1. RemoteModelHost.stop is called
        # 2. The proecss exits either normally or exceptionally, then the file is close by OS and lock is released.
        #    This is to make sure that you could re-create the server.
        lock_file = os.path.join(tempfile.gettempdir(), f"filelock_{grpc_port}")
        file_lock = FileLock(lock_file, timeout=1)
        try:
            file_lock.acquire()
            logger.info(
                f"file lock acquired, process: {os.getpid()}, thread: {threading.get_ident()}",
            )
            return file_lock
        except Timeout as e:
            logger.info(f"can't get server file lock: {e}")
            return
