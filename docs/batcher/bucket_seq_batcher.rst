==========================
Bucket Sequence Batcher
==========================

Similar to `SeqBatcher`, `BucketSeqBatcher` provides batching support for sequence inputs with variant lengths. The difference is that it will group sequences with similar lengths into the same batch, instead of having all input sequences into a single batch, to reduce the padding cost. This is useful for sequences with significantly different lengths, where some of the sequences are short, but the others are very long.

The following example defines 4 buckets to accommodate sequence of different lenghts: `<=1024`, `(1024, 2048]`, `(2048, 4096]` and `>4096`. The `BucketSeqBatcher` will sort input sequences by lengths, put them in corresponding buckets and then batch sequences within the same bucket. For example, if the sequence lenght is 2000, the `BucketSeqBatcher` will put it into the 2nd bucket. It won't be batched with a sequence of length 500, which is in the 1st bucket.

.. literalinclude:: ./bucket_seq_batcher_example.py
    :language: python
