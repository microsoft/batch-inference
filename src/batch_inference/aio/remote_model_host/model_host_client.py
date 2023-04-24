# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import grpc
import msgpack

from ...logger import logger
from . import model_host_pb2, model_host_pb2_grpc, msgpack_serialization


class ModelHostClient:
    def __init__(self, grpc_port):
        self.grpc_port = grpc_port
        self.channel = None
        self.stub = None

    # read this for details of creating async grpc server:
    # https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_client.py
    async def start(self):
        self.channel = grpc.aio.insecure_channel(f"localhost:{self.grpc_port}")
        self.stub = model_host_pb2_grpc.ModelHostStub(self.channel)
        logger.info(f"model host started as client, will talk to {self.grpc_port}")

    async def stop(self):
        pass

    async def predict(self, *input_list):
        request_packed = msgpack.packb(
            input_list,
            use_bin_type=True,
            default=msgpack_serialization.encode,
        )
        response_pb = await self.stub.predict(
            model_host_pb2.Request(ndarrays=request_packed),
        )
        response = msgpack.unpackb(
            response_pb.ndarrays, object_hook=msgpack_serialization.decode
        )
        return response
