# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import asyncio
from typing import Any

import grpc
import msgpack

from ...batcher.batcher import Batcher
from ...logger import logger
from ..model_host import ModelHost
from . import model_host_pb2, model_host_pb2_grpc, msgpack_serialization


class GrpcServicer(model_host_pb2_grpc.ModelHostServicer):
    def __init__(self, host: ModelHost) -> None:
        self.host = host

    async def predict(self, request, context):
        # logger.debug(f'received request from remote client')
        request_py = msgpack.unpackb(
            request.ndarrays, object_hook=msgpack_serialization.decode
        )
        # request_py is a python list/tuple, unpack it to be positional arguments
        response_py = await self.host.predict(*request_py)
        respone_pb = msgpack.packb(
            response_py, use_bin_type=True, default=msgpack_serialization.encode
        )
        return model_host_pb2.Response(ndarrays=respone_pb)

    async def stop(self):
        await self.host.stop()


class ModelHostServer:
    def __init__(
        self,
        file_lock,
        model_cls,
        grpc_port,
        batcher: Batcher,
        max_batch_size=32,
        wait_ms: int = 5,
        wait_n: int = 16,
        event_loop=None,
    ):
        self.model_cls = model_cls
        self.batcher = batcher
        self.max_batch_size = max_batch_size
        self.wait_ms = wait_ms
        self.wait_n = wait_n

        if event_loop is None:
            self.event_loop = asyncio.get_event_loop()
        else:
            self.event_loop = event_loop

        self.grpc_port = grpc_port
        self.file_lock = file_lock

        self.model_host: ModelHost = None
        self.server = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.model_host = ModelHost(
            model_cls=self.model_cls,
            batcher=self.batcher,
            max_batch_size=self.max_batch_size,
            wait_ms=self.wait_ms,
            wait_n=self.wait_n,
            event_loop=self.event_loop,
        )(*args, **kwds)
        return self

    # read this for details of creating async grpc server:
    # https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_client.py
    async def start(self):
        await self.model_host.start()
        self.server = grpc.aio.server()
        model_host_pb2_grpc.add_ModelHostServicer_to_server(
            GrpcServicer(self.model_host),
            self.server,
        )
        self.server.add_insecure_port(f"[::]:{self.grpc_port}")
        await self.server.start()
        logger.info(f"model host started as server, listening on {self.grpc_port}")

    async def stop(self):
        if self.server is not None:
            await self.server.stop(grace=3)

        if self.model_host is not None:
            await self.model_host.stop()

        self.file_lock.release()

    async def predict(self, *input_list):
        return await self.model_host.predict(*input_list)
