# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple

from . import tensor_ops_np, tensor_ops_pt
from .batcher import Batcher


class ConcatBatcher(Batcher):
    def __init__(self, tensor="np") -> None:
        super().__init__()
        if tensor == "np":
            self.tensor_concat = tensor_ops_np.TensorOps.concat
            self.tensor_slice = tensor_ops_np.TensorOps.slice
            self.tensor_size = tensor_ops_np.TensorOps.size
        elif tensor == "pt":
            self.tensor_concat = tensor_ops_pt.TensorOps.concat
            self.tensor_slice = tensor_ops_pt.TensorOps.slice
            self.tensor_size = tensor_ops_pt.TensorOps.size
        else:
            raise NotImplementedError(f"unsupported tensor type {tensor}")

    def batch(self, requests: List[Tuple[Any]]) -> Tuple[Tuple[Any], Any]:
        """Batch multiple N requests into 1 batched requests.
        Each requests can contains M args, concat N ndarray requests into 1 ndarray for each arg

        Args:
            requests: a list of N requests, each request is a tuple of M numpy.ndarray from predict method

        Returns:
            batched requests: 1 batched requests, which is a tuple of numpy.ndarray for predict_batch method
            context for unbatch: will be passed to unbatch method
        """
        num_of_inputs = len(requests[0])
        batched_request = []
        batch_sizes = [self.tensor_size(item[0], dim=0) for item in requests]
        for idx in range(0, num_of_inputs):
            batched_request.append(self.tensor_concat([item[idx] for item in requests]))
        return tuple(batched_request), batch_sizes

    def unbatch(self, batched_response: Any, unbatch_ctx: Any) -> List:
        """Unbatch 1 batched response into N responses
        If the batched respone contains K output variables, then each response will also contains K output variables.

        Args:
            batched_response: 1 batched response from predict_batch method
            unbatch_ctx: context from batch method

        Returns:
            responses: a list of N responses, each response will be returned by predict method
        """
        batch_sizes = unbatch_ctx
        return_type = type(batched_response)
        responses = []

        start = 0
        for n in batch_sizes:
            if return_type is tuple:
                response = tuple(
                    [
                        self.tensor_slice(output, start, start + n)
                        for output in batched_response
                    ]
                )
            elif return_type is list:
                response = [
                    self.tensor_slice(output, start, start + n)
                    for output in batched_response
                ]
            else:  # single output
                response = self.tensor_slice(batched_response, start, start + n)
            responses.append(response)
            start += n

        return responses
