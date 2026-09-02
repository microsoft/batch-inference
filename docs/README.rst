=============================
Batch Inference Toolkit
=============================

Batch Inference Toolkit(batch-inference) is a Python package that batches model input tensors coming from multiple users dynamically, executes the model, un-batches output tensors and then returns them back to each user respectively. This will improve system throughput because of a better cache locality. The entire process is transparent to developers.

.. figure:: figures/batching_overview.png
  :width: 500
  :align: center
  :alt: How Batching Inference Works

Installation
============================

**Install from Pip** *(Coming Soon)*

.. code:: bash

    python -m pip install batch-inference --upgrade

**Build and Install from Source**

.. code:: bash

    git clone https://github.com/microsoft/batch-inference.git
    cd python
    python -m pip install -e .

    # if you want to format the code before commit
    pip install pre-commit
    pre-commit install

    # run unittests
    python -m unittest discover tests

Example
============================

.. code:: python

    import threading
    import numpy as np
    from batch_inference import batching


    @batching(max_batch_size=32)
    class MyModel:
        def __init__(self, k, n):
            self.weights = np.random.randn(k, n).astype("f")

        # x: [batch_size, m, k], self.weights: [k, n]
        def predict_batch(self, x):
            y = np.matmul(x, self.weights)
            return y


    with MyModel.host(3, 3) as host:
        def send_requests():
            for _ in range(0, 10):
                x = np.random.randn(1, 3, 3).astype("f")
                y = host.predict(x)

        threads = [threading.Thread(target=send_requests) for i in range(0, 32)]
        [th.start() for th in threads]
        [th.join() for th in threads]

Build the Docs
=============================

Run the following commands and open ``docs/_build/html/index.html`` in browser.

.. code:: bash

    pip install sphinx myst-parser sphinx-rtd-theme sphinxemoji
    cd docs/

    make html         # for linux
    .\make.bat html   # for windows
