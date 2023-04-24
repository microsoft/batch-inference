# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from threading import Condition
from typing import Any


class BatchContext:
    def __init__(self, request: Any) -> None:
        self.request = request
        self.response = None
        self.error = None
        self.response_ready = Condition()

    def get_response(self) -> Any:
        with self.response_ready:
            self.response_ready.wait()
            if self.error is not None:
                raise self.error
            return self.response

    def set_response(self, response: Any = None):
        with self.response_ready:
            self.response = response
            self.response_ready.notify()

    def set_error(self, error: Exception):
        with self.response_ready:
            self.error = error
            self.response_ready.notify()
