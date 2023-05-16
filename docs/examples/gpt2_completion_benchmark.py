# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from transformers import GPT2Tokenizer

from benchmark import benchmark, benchmark_sync
from gpt2_completion import Gpt2Completion
from gpt2_baseline import Gpt2Baseline


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

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    queries = []
    for text in texts:
        queries.append((tokenizer.encode(text),))

    print("Test with batching")
    with Gpt2Completion.host() as model_host:
        benchmark_sync(model_host, queries, num_calls=100, parallel=64, warm_up_calls=10)
        print(f"Query count: {model_host.model_obj.query_count}. Batch count: {model_host.model_obj.batch_count} Token count: {model_host.model_obj.token_count}. Inference count: {model_host.model_obj.inference_count}")
    
    print("Test baseline")
    baseline = Gpt2Baseline()
    benchmark(baseline, queries, num_calls=100, parallel=2, warm_up_calls=10)
    print(f"Query count: {baseline.query_count}. Token count : {baseline.token_count}")


if __name__ == "__main__":
    main()
