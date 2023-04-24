==========================
Sequence Batcher
==========================

The `SeqBatcher` provide batching support for sequence inputs with variant lengths, which is common in Natural Language Tasks. It is a wrapper of `ConcatBatcher`. It will first pad the inputs with padding tokens, then concatenate the batched inputs with `ConcatBatcher`.

.. literalinclude:: ./seq_batcher_example.py
    :language: python
