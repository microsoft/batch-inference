==========================
GPT Completion
==========================

This tutorial shows the benchmark of applying batch-inference on GPT completion scenario, and code is provided.

Huggingface GPT2 model is used here as it's open source. Greedy search is used to generate next token, other kinds of decoding strategies like beam search can be also applied but they will this tutorial too complex. 

Here's the result running the benchmark code on a NVIDIA V100 GPU, the batching version finished 100 queries with 6400 tokens in 4.86 seconds, while baseline version took 78.04 seconds, 
and that means applying batch-inference can provide **16 times throughput** comparing to baseline. 
With batch-inference it can reduce number of inference calls to GPT2 model, and thus reduce total execution time dramatically. 

.. list-table:: 
   :widths: 30 25 25 25 25 25 25 25
   :header-rows: 0

   * - Method
     - Query Count
     - Execution Time
     - Throughput comparing to Baseline
     - Max Batch Size
     - Avg Batch Size
     - Generated Token Count
     - Inference Count
   * - Baseline
     - 100
     - 78.04s
     - 1x
     - 1
     - 1
     - 6400
     - 6400
   * - With batch-inference
     - 100
     - 4.86s
     - 16x
     - 32
     - 25
     - 6400
     - 256


Baseline GPT2 completion code:

.. code:: python

    import torch
    from transformers import GPT2LMHeadModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Gpt2Baseline:
        def __init__(self):
            self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            self.max_output_length = 64
            self.eos_token = 50256   # Token of <|endoftext|>

        def predict_batch(self, input_ids):
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
                if token == self.eos_token:
                    break
            return sequence

Code links:

`Baseline GPT2 completion <https://github.com/microsoft/batch-inference/blob/main/docs/examples/gpt2_baseline.py>`__

`Applying batching on GPT2 completion <https://github.com/microsoft/batch-inference/blob/main/docs/examples/gpt2_completion.py>`__

`Benchmark code <https://github.com/microsoft/batch-inference/blob/main/docs/examples/gpt2_completion_benchmark.py>`__