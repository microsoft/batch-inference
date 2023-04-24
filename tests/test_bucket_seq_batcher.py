# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np

from batch_inference.batcher import BucketSeqBatcher


class TestModelHost(unittest.TestCase):
    def test_batch(self):
        buckets = [3, 5]
        batcher = BucketSeqBatcher(padding_tokens=[0], buckets=buckets)
        requests = [
            (np.random.random([1, 1, 8]),),
            (np.random.random([1, 1, 4]),),
            (np.random.random([1, 1, 9]),),
            (np.random.random([1, 1, 5]),),
            (np.random.random([1, 1, 2]),),
        ]
        batched_requests, unbatch_ctx = batcher.batch(requests)
        self.assertEqual(len(batched_requests), 3)
        for batched_request in batched_requests:
            self.assertEqual(len(batched_request), 1)
        self.assertEqual(batched_requests[0][0].shape, (1, 1, 2))
        self.assertEqual(batched_requests[1][0].shape, (2, 1, 5))
        self.assertEqual(batched_requests[2][0].shape, (2, 1, 9))

    def test_unbatch(self):
        buckets = [3, 5]
        batcher = BucketSeqBatcher(padding_tokens=[0], buckets=buckets)
        requests = [
            (np.random.random([1, 1, 8]),),
            (np.random.random([1, 1, 4]),),
            (np.random.random([1, 1, 9]),),
            (np.random.random([1, 1, 5]),),
            (np.random.random([1, 1, 2]),),
        ]
        batched_requests, unbatch_ctx = batcher.batch(requests)
        responses = batcher.unbatch(batched_requests, unbatch_ctx)
        self.assertEqual(len(responses), len(requests))
        for i in range(len(requests)):
            padded_request = np.zeros(responses[i][0].shape)
            padded_request[:, :, : requests[i][0].shape[2]] = requests[i][0]
            self.assertTrue(np.equal(responses[i][0], padded_request).all())

    def test_prebatched_input(self):
        buckets = [5, 10]
        batcher = BucketSeqBatcher(padding_tokens=[0], buckets=buckets)
        requests = [
            [np.random.random([3, 1, 8])],
            [np.random.random([1, 1, 4])],
            [np.random.random([2, 1, 9])],
            [np.random.random([1, 1, 5])],
            [np.random.random([3, 1, 2])],
        ]
        batched_requests, unbatch_ctx = batcher.batch(requests)
        self.assertEqual(len(batched_requests), 2)
        for batched_request in batched_requests:
            self.assertEqual(len(batched_request), 1)
        self.assertEqual(batched_requests[0][0].shape, (5, 1, 5))
        self.assertEqual(batched_requests[1][0].shape, (5, 1, 9))

        responses = batcher.unbatch(batched_requests, unbatch_ctx)
        self.assertEqual(len(responses), len(requests))
        for i in range(len(requests)):
            padded_request = np.zeros(responses[i][0].shape)
            padded_request[:, :, : requests[i][0].shape[2]] = requests[i][0]
            self.assertTrue(np.equal(responses[i][0], padded_request).all())


if __name__ == "__main__":
    unittest.main()
