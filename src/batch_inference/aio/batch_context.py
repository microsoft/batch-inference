# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import asyncio
from typing import Any


class BatchContext:
    def __init__(self, request: Any) -> None:
        self.request = request
        self.response = None
        self.error = None
        self.response_ready = asyncio.Event()

    async def get_response(self) -> Any:
        await self.response_ready.wait()
        if self.error is not None:
            raise self.error
        return self.response

    def set_response(self, response: Any = None):
        self.response = response
        self.response_ready.set()

    def set_error(self, error: Exception):
        self.error = error
        self.response_ready.set()
