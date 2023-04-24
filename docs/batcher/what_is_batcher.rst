==========================
What is Batcher?
==========================

The `Batcher` class defines how mutiple requests from users are merged into a batch request, and how the batch response is splitted into multiple responses for users. It exposes two interface functions for developers to implement their own batcher that fits a specific scenario.

* `batch`. Merge multiple requests into a batch request, and attach a context object for unbatching.
* `unbatch`. Split a batch response into multiple responses, with the help of the context object.

Built-in Batchers
===================================

The following built-in batchers are provided in `batch_inference.batcher` module. Both `numpy.ndarry` and `torch.Tensor` are supported as input date types.

* :doc:`ConcatBatcher <./concat_batcher>`. Simply concatenate multiple requests into a single batch request.
* :doc:`SeqBatcher <./seq_batcher>`. Pad sequences of different lengths with padding tokens before concatenation.
* :doc:`BucketSeqBatcher <./bucket_seq_batcher>`. Group sequences of similar lengths, pad them with padding tokens and then concatenate.

Implement Customized Batcher
======================================

The following example shows how to implement a customized batcher. The batcher merges multiple requests into a single batch request, and splits the batch response into multiple responses.

.. literalinclude:: ./customized_batcher.py
    :language: python


The MultiBatcher Class
=======================================

In most cases, we merge multiple requests from users into a single batch request. :doc:`ConcatBatcher <./concat_batcher>` and :doc:`SeqBatcher <./seq_batcher>` are examples of this.

Sometimes, we want to group multiple requests first and then merged requests in each group into a batch request. In this case, the Batcher should inherits the `batcher.MultiBatcher` class instead of `batche.Batcher`. :doc:`BucketSeqBatcher <./bucket_seq_batcher>` is an example of this.
