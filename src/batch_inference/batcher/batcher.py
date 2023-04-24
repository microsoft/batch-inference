# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple


class Batcher:
    """A batcher should contain a pair of batch and unbatch method"""

    def batch(self, requests: List[Tuple[Any]]) -> Tuple[Tuple[Any], Any]:
        """Batch multiple N requests into 1 batched requests

        Args:
            requests: a list of N requests, each request is a tuple of args from predict method

        Returns:
            batched requests: 1 batched requests, which is a tuple of args for predict_batch method
            context for unbatch: will be passed to unbatch method
        """
        raise NotImplementedError()

    def unbatch(self, batched_response: Any, unbatch_ctx: Any) -> List:
        """Unbatch 1 batched response into N responses

        Args:
            batched_response: 1 batched responses from predict_batch method
            unbatch_ctx: context from batch method

        Returns:
            responses: a list of N responses, each response will be returned by predict method
        """
        raise NotImplementedError()
