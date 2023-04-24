# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple

from .batcher import Batcher


class MultiBatcher(Batcher):
    """A batcher should contain a pair of batch and unbatch method"""

    def batch(self, requests: List[Tuple[Any]]) -> Tuple[List[Tuple[Any]], Any]:
        """Batch multiple M requests into N batched requests (M>=N)

        Args:
            requests: a list of M requests, each request is a tuple of args from predict method

        Returns:
            batched requests: a list of N batched requests, each batched request is a tuple of args for predict_batch method
            context for unbatch: will be passed to unbatch method
        """
        raise NotImplementedError()

    def unbatch(self, batched_responses: List[Any], unbatch_ctx: Any) -> List:
        """Unbatch N batched responses into M responses

        Args:
            batched_responses: a list of N batched responses from predict_batch method
            unbatch_ctx: context from batch method

        Returns:
            responses: a list of M responses, each response will be returned by predict method
        """
        raise NotImplementedError()
