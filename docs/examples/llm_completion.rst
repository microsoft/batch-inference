==========================
LLM Completion
==========================

This tutorial shows how to apply batch-inference on LLM completion scenario. 

Huggingface GPT2 model is used here as it's open source and small, you can do the same thing for other LLM models. 
Greedy search is used here to generate next token, other kinds of decoding strategies like beam search can be also applied but they will this tutorial too complex. 

Here's the result running the benchmark code on a NVIDIA GeForce RTX 2080 Ti GPU, 

.. code:: bash

    /batch-inference$ python docs/examples/gpt2_completion_benchmark.py
    Test with batching
    Start Running
    Total time: 12.230200 seconds
    Compute time (1): {140672835569408: 12.219795} seconds
    Query count: 110. Batch count: 15 Token count: 7040. Inference count: 960
    Test baseline
    Start Running
    Total time: 62.871866 seconds
    Compute time (2): {140672835569408: 62.74406, 140672493418240: 62.863644} seconds
    Query count: 110. Token count : 7040


.. literalinclude:: ./gpt2_completion.py
    :language: python