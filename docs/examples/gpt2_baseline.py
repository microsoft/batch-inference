# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from transformers import GPT2LMHeadModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Gpt2Baseline:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.max_output_length = 64
        self.eos_token = 50256   # Token of <|endoftext|>
        # counters
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
            past_key_values = output.past_key_values
            token = torch.argmax(output.logits[..., -1, :])

            context = token.unsqueeze(0)
            token = token.tolist()
            sequence += [token]
            self.token_count += 1
            if token == self.eos_token:
                break
        return sequence
    
    def reset_counters(self):
        self.token_count = 0
        self.query_count = 0