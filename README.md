# Batch Inference Toolkit

Batch Inference Toolkit(batch-inference) is a Python package that batches model input tensors coming from multiple users dynamically, executes the model, un-batches output tensors and then returns them back to each user respectively. This will improve system throughput because of better compute parallelism and better cache locality. The entire process is transparent to developers. 

The library provides very flexible APIs so it can be used in complex scenarios like LLM and [RNN models](https://github.com/microsoft/batch-inference/blob/main/tests/test_torch_rnn_batcher.py). It achieved **4 times throughput improvement** on **GPT completion** in our [experiment](https://microsoft.github.io/batch-inference/examples/gpt_completion.html).

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

We provide more practical examples including [GPT completion](https://microsoft.github.io/batch-inference/examples/gpt_completion.html) and [Bert embedding](https://microsoft.github.io/batch-inference/examples/bert_mpletion.html) in tutorials, however let's start with a toy model here to learn the APIs.

The **batching** decorator adds host() method to create **ModelHost** object. The **predict** method of ModelHost takes a single query as input, and it will merge multiple queries into a batch before calling **predict_batch** method. The predict method also splits outputs from predict_batch method before it returns result.

```python
import numpy as np
from batch_inference import batching


@batching(max_batch_size=32)
class MyModel:
    def __init__(self, k, n):
        self.weights = np.random.randn((k, n)).astype("f")

    # x: [batch_size, m, k], self.weights: [k, n]
    def predict_batch(self, x):
        y = np.matmul(x, self.weights)
        return y


host = MyModel.host(3, 3)
host.start()

for _ in range(0, 10):
    x = np.random.randn(1, 3, 3).astype("f")
    y = host.predict(x)

host.stop()
```

While ModelHost can gather queries from multiple threads of a process, **RemoteModelHost** can gather queries from multiple processes to avoid GIL's impact on performance. [ModelHost](https://microsoft.github.io/batch-inference/model_host.html) and [RemoteModelHost](https://microsoft.github.io/batch-inference/remote_model_host.html) explain how they work.

**Batcher** is responsible to merge queries and split outputs. We provide a set of built-in Batchers for common scenarios, but you also use your own Batcher. See [What is Batcher](https://microsoft.github.io/batch-inference/batcher/what_is_batcher.html) for more information.

## Build the Docs

Run the following commands and open `docs/_build/html/index.html` in browser.

```bash
pip install sphinx myst-parser sphinx-rtd-theme sphinxemoji
cd docs/

make html         # for linux
.\make.bat html   # for windows
```
