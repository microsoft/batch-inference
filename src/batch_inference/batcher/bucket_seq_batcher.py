# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, List, Tuple

from .multi_batcher import MultiBatcher
from .seq_batcher import SeqBatcher


class BucketSeqBatcher(MultiBatcher):
    def __init__(
        self, padding_tokens: List[Any], buckets: List[int], tensor: str = "np"
    ) -> None:
        """
        Pad and Concat sequence requests. Put requests with same length into one bucket

        Args:
            padding_tokens: a List of M padding tokens for corresponding request arg
            buckets: a List of request length for each bucket
        """
        super().__init__()
        self.buckets = buckets
        self.seq_batcher = SeqBatcher(padding_tokens=padding_tokens, tensor=tensor)

    def batch(self, requests: List[Tuple[Any]]) -> Tuple[List[Tuple[Any]], Any]:
        """Batch multiple M requests into N batched requests (M>=N)

        Args:
            requests: a list of M requests, each request is a tuple of args from predict method

        Returns:
            batched requests: a list of N batched requests, each batched request is a tuple of args for predict_batch method
            context for unbatch: will be passed to unbatch method
        """

        # Sort requests by sequence length
        requests = [
            (
                item,
                idx,
                self.seq_batcher.get_seq_length(item),
                self.get_batch_size(item),
            )
            for idx, item in enumerate(requests)
        ]
        sorted_requests = sorted(requests, key=lambda x: x[2])
        groups = [[]]

        # Group requests by sequence length
        i, j = 0, 0
        while i < len(self.buckets) and j < len(sorted_requests):
            if sorted_requests[j][2] <= self.buckets[i]:
                groups[-1].append(sorted_requests[j])
                j += 1
            else:
                groups.append([])
                i += 1

        groups.append(sorted_requests[j:])
        groups = [group for group in groups if len(group) != 0]

        # Batch requests
        batched_requests = []
        unbatch_ctx = []
        for group in groups:
            concated, _ = self.seq_batcher.batch([item[0] for item in group])
            batched_requests.append(tuple(concated))
            unbatch_ctx.append([(item[1], item[3]) for item in group])

        return batched_requests, unbatch_ctx

    def unbatch(self, batched_responses: List[Any], unbatch_ctx: Any) -> List:
        """Unbatch responses from the batched_responses. The order of the responses
        should be the same as the order of the inputs.

        Args:
            batched_responses (List): The output from the batched responses.
            unbatch_ctx (Any): The unbatch context.

        Returns:
            List: The unbatched responses.
        """
        responses = []

        for batched_response, group_info in zip(batched_responses, unbatch_ctx):
            start = 0
            for slice_info in group_info:  # slice_info: [(original index, batch_size)]
                response = self.slice(batched_response, start, start + slice_info[1])
                response = (
                    response,
                    slice_info[0],
                )  # append original index for reordering
                responses.append(response)
                start += slice_info[1]

        sorted_responses = sorted(responses, key=lambda x: x[-1])
        return [x[0] for x in sorted_responses]

    def get_batch_size(self, single_request):
        return self.seq_batcher.tensor_size(single_request[0], dim=0)

    def slice(self, batched_response, start, end):
        return_type = type(batched_response)
        if return_type is tuple:
            return tuple(
                [
                    self.seq_batcher.tensor_slice(output, start, end)
                    for output in batched_response
                ]
            )
        elif return_type is list:
            return [
                self.seq_batcher.tensor_slice(output, start, end)
                for output in batched_response
            ]
        else:
            return self.seq_batcher.tensor_slice(batched_response, start, end)
