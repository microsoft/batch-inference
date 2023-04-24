# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import asyncio
from typing import Any


class BatchContext:
    def __init__(self) -> None:
        self.requests = []
        self.responses = []
        self.result_ready = asyncio.Event()
        self.error = None

    def size(self):
        return len(self.requests)

    def add_request(self, request: Any):
        self.requests.append(request)
        return len(self.requests) - 1

    def set_result_ready(self):
        self.result_ready.set()

    def set_error(self, error: Exception):
        self.error = error
        self.result_ready.set()
