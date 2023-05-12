# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from benchmark import benchmark, benchmark_sync
from bert_embedding import BertEmbeddingModel
from transformers import BertTokenizer
from batch_inference.model_host import ModelHost
from batch_inference.batcher.bucket_seq_batcher import BucketSeqBatcher

def main():
    texts = [
        "The Manhattan bridge",
        "Python lists are a data structure similar to dynamically",
        "Tuples in Python are a data structure used to store multiple elements in a single variable. Just like list data structure, a tuple is",
        "Even though List and Tuple are different data structures",
        "An operating system (OS) is the program that",
        "An operating system brings powerful benefits to computer software",
        "As long as each application accesses the same resources and services",
        "An operating system provides three essential capabilities: ",
        "The GUI is most frequently used by casual or end users that are primarily",
        "An operating system can",
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    queries = []
    for text in texts:
        encoded_text = tokenizer.encode_plus(
        text, return_attention_mask=True, return_tensors="pt"
        )  
        queries.append((encoded_text["input_ids"], encoded_text["attention_mask"]))

    print("Test with BucketSeqBatcher")
    with ModelHost(
            BertEmbeddingModel,
            batcher=BucketSeqBatcher(padding_tokens=[0, 0], buckets=[16, 32, 64], tensor='pt'),
            max_batch_size=10,
        )() as model_host:
        benchmark_sync(model_host, queries, num_calls=1000, parallel=64, warm_up_calls=10)
        print(f"Query count: {model_host.model_obj.query_count}. Batch count: {model_host.model_obj.batch_count}")

    print("Test with SeqBatcher")
    with BertEmbeddingModel.host() as model_host:
        benchmark_sync(model_host, queries, num_calls=1000, parallel=64, warm_up_calls=10)
        print(f"Query count: {model_host.model_obj.query_count}. Batch count: {model_host.model_obj.batch_count}")
    
    print("Test baseline")
    baseline = BertEmbeddingModel()
    benchmark(baseline, queries, num_calls=1000, parallel=2, warm_up_calls=10)
    print(f"Query count: {baseline.query_count}. Batch count: {baseline.batch_count}")


if __name__ == "__main__":
    main()
