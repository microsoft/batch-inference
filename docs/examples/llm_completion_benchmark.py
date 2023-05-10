
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from benchmark import benchmark, benchmark_sync
from docs.examples.llm_completion import Gpt2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gpt2:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.max_output_length = 64
        self.eos_token = 50256   # Token of <|endoftext|>
        # statistics
        self.token_count = 0
        self.query_count = 0

    def predict_batch(self, input_ids):
        self.query_count += 1
        sequence = input_ids
        context = torch.tensor([sequence]).to(device)
        past_key_values = None

        for i in range(self.max_output_length):
            output = self.model(context, past_key_values=past_key_values, use_cache=True)
            # shape: [layer, k&v, batchsize, head, token length, head dim]
            # for example: [12, 2, 1, 12, n, 64] for GPT2 small; [96, 2, 1, 96, n, 128] for GPT3 davinci
            # 9MB for 1 token in GPT3 davinci
            past_key_values = output.past_key_values
            token = torch.argmax(output.logits[..., -1, :])
            # only generated token is used in next iteration

            context = token.unsqueeze(0)
            token = token.tolist()
            sequence += [token]
            self.token_count += 1
            if token == self.eos_token:
                break
        return sequence


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

    print("Test without batching")
    raw_model = Gpt2()
    benchmark(raw_model, queries, num_calls=10, parallel=1, warm_up_calls=10)
    print(f"Query count: {raw_model.query_count}. Token count : {raw_model.token_count}")

    print("Test with batching")
    with Gpt2Model.host() as model_host:
        benchmark_sync(model_host, queries, num_calls=10, parallel=10, warm_up_calls=10)
        print(f"Query count: {model_host.model_obj.query_count}. Batch count: {model_host.model_obj.batch_count} Token count: {model_host.model_obj.token_count}. Inference count: {model_host.model_obj.inference_count}")


if __name__ == "__main__":
    main()
