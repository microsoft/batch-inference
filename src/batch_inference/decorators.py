# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, Tuple, Type

from .batcher.concat_batcher import Batcher, ConcatBatcher
from .logger import logger
from .model_host import ModelHost
from .remote_model_host import RemoteModelHost


def batching(*arg: Tuple, **kwargs: Dict[str, Any]) -> Callable[[Type], Type]:
    if len(arg) > 0 and _valid_model(arg[0]):
        return _decorator()(arg[0])
    else:
        return _decorator(*arg, **kwargs)


def remote_batching(
    grpc_port: int, batcher: Batcher = None, max_batch_size: int = 32
) -> Callable[[Type], Type]:
    if batcher is None:
        batcher = ConcatBatcher()

    def add_host(model_cls: Type) -> Type:
        def host(*args, **kwargs):
            logger.debug(f"create ModelHost for {str(model_cls)}")
            model_host = RemoteModelHost(
                model_cls=model_cls,
                grpc_port=grpc_port,
                batcher=batcher,
                max_batch_size=max_batch_size,
            )
            model_host(*args, **kwargs)
            return model_host

        setattr(model_cls, "host", host)
        return model_cls

    return add_host


def _valid_model(obj):
    method = getattr(obj, "predict_batch", None)
    return callable(method)


def _decorator(
    batcher: Batcher = None, max_batch_size: int = 32
) -> Callable[[Type], Type]:
    if batcher is None:
        batcher = ConcatBatcher()

    def add_host(model_cls: Type) -> Type:
        def host(*args, **kwargs) -> ModelHost:
            logger.debug(f"create ModelHost for {str(model_cls)}")
            model_host = ModelHost(
                model_cls=model_cls,
                batcher=batcher,
                max_batch_size=max_batch_size,
            )
            model_host(*args, **kwargs)
            return model_host

        setattr(model_cls, "host", host)
        return model_cls

    return add_host
