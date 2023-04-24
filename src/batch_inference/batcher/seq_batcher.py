# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple

import numpy as np

from . import tensor_ops_np, tensor_ops_pt
from .concat_batcher import ConcatBatcher


class SeqBatcher(ConcatBatcher):
    def __init__(self, padding_tokens=List[Any], tensor: str = "np") -> None:
        """
        Pad and Concat sequence requests

        Args:
            padding_tokens: a List of M padding tokens for corresponding request arg. Use None if an argument doesn't need padding
        """
        super().__init__(tensor=tensor)
        if tensor == "np":
            self.tensor_pad = tensor_ops_np.TensorOps.pad
        elif tensor == "pt":
            self.tensor_pad = tensor_ops_pt.TensorOps.pad
        else:
            raise NotImplementedError(f"unsupported tensor type {tensor}")
        self.padding_tokens = padding_tokens

    def batch(self, requests: List[Tuple[Any]]) -> Tuple[Tuple[Any], Any]:
        """Batch multiple N requests into 1 batched requests.
        Each requests can contains M args, concat N ndarray requests into 1 ndarray for each arg

        Args:
            requests: a list of N requests, each request is a tuple of M numpy.ndarray from predict method

        Returns:
            batched requests: 1 batched requests, which is a tuple of numpy.ndarray for predict_batch method
            context for unbatch: will be passed to unbatch method
        """

        padded_requests = requests
        if self.padding_tokens:
            max_len = -1
            for req in requests:
                seq_len = self.get_seq_length(req)
                if seq_len > max_len:
                    max_len = seq_len
            padded_requests = [self.pad(req, max_len) for req in requests]

        return super().batch(padded_requests)

    def get_seq_length(self, single_request: List[Any]) -> int:
        """
        Return length of first argument, which means other arguments will be padded to same length of first argument
        """
        return self.tensor_size(single_request[0], dim=-1)

    def pad(self, single_request: Tuple[Any], seq_len: int) -> Tuple[Any]:
        # original request: (input1, input2, ...)

        padded_inputs = []
        for idx, input in enumerate(single_request):
            pad_length = seq_len - self.tensor_size(input, dim=-1)
            if pad_length > 0:
                p1d = (0, pad_length)
                padded_inputs.append(
                    self.tensor_pad(input, p1d, self.padding_tokens[idx]),
                )
            else:
                padded_inputs.append(input)
        return padded_inputs
