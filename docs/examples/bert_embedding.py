from typing import Tuple

import torch
from transformers import BertModel

from batch_inference import batching
from batch_inference.batcher.seq_batcher import SeqBatcher


@batching(batcher=SeqBatcher(padding_tokens=[0, 0], tensor='pt'), max_batch_size=32)
class BertEmbeddingModel:
    def __init__(self):
        self.model = BertModel.from_pretrained("bert-base-uncased")
        # counters
        self.query_count = 0
        self.batch_count = 0

    def predict_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        self.batch_count += 1
        self.query_count += input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            embedding = outputs[0]
        return embedding
