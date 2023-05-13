# Batch Inference Toolkit

Batch Inference Toolkit(batch-inference) is a Python package that batches model input tensors coming from multiple users dynamically, executes the model, un-batches output tensors and then returns them back to each user respectively. This will improve system throughput because of better compute parallelism and better cache locality. The entire process is transparent to developers. 

## When to use

When you want to host Deep Learning model inference on Cloud servers, especially on GPU

## Why to use

It can improve your server throughput up to multiple times

## Advantage of batch-inference

* Platform independent lightweight python library
* Only few lines code change is needed to onboard using built-in [batching algorithms](https://microsoft.github.io/batch-inference/batcher/what_is_batcher.html)
* Flexible APIs to support customized batching algorithms and input types
* Support [multi-process remote mode](https://microsoft.github.io/batch-inference/remote_model_host.html) to avoid python GIL bottleneck
* Tutorials and benchmarks on popular models like [GPT](https://microsoft.github.io/batch-inference/examples/gpt_completion.html) and [Bert](https://microsoft.github.io/batch-inference/examples/bert_embedding.html)


## Installation

**Install from Pip** _(Coming Soon)_

```bash
python -m pip install batch-inference --upgrade
```

**Build and Install from Source** _(for developers)_

```bash
git clone https://github.com/microsoft/batch-inference.git
python -m pip install -e .[docs,testing]

# if you want to format the code before commit
pip install pre-commit
pre-commit install

# run unittests
python -m unittest discover tests
```

## Example

Let's start with a toy model to learn the APIs. Firstly, you need to define a **predict_batch** method in your model class, and then add the **batching** decorator to your model class.

The **batching** decorator adds host() method to create **ModelHost** object. The **predict** method of ModelHost takes a single query as input, and it will merge multiple queries into a batch before calling **predict_batch** method. The predict method also splits outputs from predict_batch method before it returns result.

```python
import numpy as np
from batch_inference import batching
from batch_inference.batcher.concat_batcher import ConcatBatcher

@batching(batcher=ConcatBatcher(), max_batch_size=32)
class MyModel:
    def __init__(self, k, n):
        self.weights = np.random.randn((k, n)).astype("f")

    # shape of x: [batch_size, m, k]
    def predict_batch(self, x):
        y = np.matmul(x, self.weights)
        return y

# initialize MyModel with k=3 and n=3
host = MyModel.host(3, 3)
host.start()

# shape of x: [1, 3, 3]
def process_request(x):
    y = host.predict(x)
    return y

```

**Batcher** is responsible to merge queries and split outputs. In this case ConcatBatcher will concat input tensors into a batched tensors at first dimension. We provide a set of built-in Batchers for common scenarios, and you can also implement your own Batcher. See [What is Batcher](https://microsoft.github.io/batch-inference/batcher/what_is_batcher.html) for more information.
