==========================
Bert Embedding
==========================

This tutorial shows the benchmark of applying batch-inference on Bert embedding scenario, and code is provided.

Two batching methods Sequence Batcher and Bulk Sequence Batcher were tested. 
`Sequence Batcher <https://microsoft.github.io/batch-inference/batcher/seq_batcher.html>`__ will pad input sequences to same length within a batch. 
Additionally, `Bulk Sequence Batcher <https://microsoft.github.io/batch-inference/batcher/bucket_seq_batcher.html>`__ divides inputs sequences of different lengths into four buckes [1, 16], [17, 32], [33, 64], [64,) based on given bucket setting. 

As shown in the table, Bulk Sequence Batcher can achieve about **3.7 times throughput** and Sequence Batcher can achieve about **2.4 times throughput** comparing to baseline.

The experiments were run on NVIDIA GeForce RTX 2080 Ti GPU. 

.. list-table:: 
   :widths: 25 25 25 25 25
   :header-rows: 0

   * - Method
     - Query Count
     - Execution Time
     - Max Batch Size Setting
     - Avg Batch Size
   * - Baseline
     - 2000
     - 79.84s
     - /
     - 1
   * - Sequence Batcher
     - 2000
     - 40.64s
     - 4
     - 4
   * - Sequence Batcher
     - 2000
     - 34.40s
     - 8
     - 7.96
   * - Sequence Batcher
     - 2000
     - 32.49s
     - 32
     - 31.25
   * - Bulk Sequence Batcher
     - 2000
     - 21.31
     - 32
     - 15.74

Bert embedding with Bulk Sequence Batcher:

.. code:: python

    from typing import Tuple

    import torch
    from transformers import BertModel

    from batch_inference import batching
    from batch_inference.batcher.bucket_seq_batcher import BucketSeqBatcher


    @batching(batcher=BucketSeqBatcher(padding_tokens=[0, 0], buckets=[16, 32, 64], tensor='pt'), max_batch_size=32)
    class BertEmbeddingModel:
        def __init__(self):
            self.model = BertModel.from_pretrained("bert-base-uncased")

        def predict_batch(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
        ) -> Tuple[torch.Tensor]:

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                embedding = outputs[0]
            return embedding


`Benchmark code <https://github.com/microsoft/batch-inference/blob/main/docs/examples/bert_embedding_benchmark.py>`__