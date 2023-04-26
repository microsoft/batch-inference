# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import asyncio
import numpy as np
from transformers import BertTokenizer, BertModel

from batch_inference import ModelHost, aio
from batch_inference.batcher import ConcatBatcher
from benchmark import *


class BatchBertModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")

    # input_ids: [batch_size, seq_len], attention_masks: [batch_size, seq_len]
    def predict_batch(self, input_ids, attention_mask, token_type_ids):
        res = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return res
    
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt')


if __name__ == "__main__":
    m = BatchBertModel()
    encoded_dict = m.tokenize("This is the lesson: never give in, never give in nothing")
    queries = [(encoded_dict['input_ids'], encoded_dict['attention_mask'], encoded_dict['token_type_ids'])]
    
    # # batching, sync
    # with ModelHost(
    #         model_cls=BatchBertModel,
    #         batcher=ConcatBatcher(tensor="pt"),
    #         max_batch_size=16,
    #         wait_ms=5,
    #         wait_n=4,
    #         num_workers=4)() as host:   
    #     benchmark_sync(host, queries, num_calls=1000, parallel=16)
    
    # batching, async
    async def run_async():
        async with aio.ModelHost(
                model_cls=BatchBertModel,
                batcher=ConcatBatcher(tensor="pt"),
                max_batch_size=16,
                wait_ms=5,
                wait_n=4,
                num_workers=4)() as host:
            await benchmark_async(host, queries, num_calls=1000)
    asyncio.run(run_async())
    
    # # no batching
    # sut = BatchBertModel()  
    # benchmark(sut, queries, num_calls=1000, parallel=16)
