# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest

import numpy as np
import onnxruntime as ort

from batch_inference import batching
from batch_inference.batcher import ConcatBatcher


@batching
class DecoratorModel:
    def __init__(self, model_file):
        self.ort_sess = ort.InferenceSession(model_file)

    def predict_batch(self, x):
        res = self.ort_sess.run(None, {"x": x, "y": x})
        print(f"{type(res)}, {type(res[0])}")
        return res[0]  # 'res'


@batching()
class DecoratorModelDefaultArgs:
    def __init__(self, model_file):
        self.ort_sess = ort.InferenceSession(model_file)

    def predict_batch(self, x):
        res = self.ort_sess.run(None, {"x": x, "y": x})
        return res[0]  # 'res'


@batching(ConcatBatcher())
class DecoratorModelWithArg:
    def __init__(self, model_file):
        self.ort_sess = ort.InferenceSession(model_file)

    def predict_batch(self, x):
        res = self.ort_sess.run(None, {"x": x, "y": x})
        return res[0]  # 'res'


class TestDecorator(unittest.TestCase):
    def setUp(self) -> None:
        model_file = os.path.join(os.path.dirname(__file__), "matmul.onnx")
        self.decorator_model = DecoratorModel.host(model_file)
        self.decorator_model.start()
        self.decorator_model_default_args = DecoratorModelDefaultArgs.host(model_file)
        self.decorator_model_default_args.start()
        self.decorator_model_with_args = DecoratorModelWithArg.host(model_file)
        self.decorator_model_with_args.start()

    def tearDown(self) -> None:
        self.decorator_model.stop()
        self.decorator_model_default_args.stop()
        self.decorator_model_with_args.stop()

    def test_decorator(self):
        x = np.random.randn(1, 3, 3).astype("f")
        res = self.decorator_model.predict(x)
        self.assertTrue(np.allclose(res, np.matmul(x, x), rtol=1e-05, atol=1e-05))

    def test_decorator_default_args(self):
        x = np.random.randn(1, 3, 3).astype("f")
        res = self.decorator_model_default_args.predict(x)
        self.assertTrue(np.allclose(res, np.matmul(x, x), rtol=1e-05, atol=1e-05))

    def test_decorator_with_args(self):
        x = np.random.randn(1, 3, 3).astype("f")
        res = self.decorator_model_with_args.predict(x)
        self.assertTrue(np.allclose(res, np.matmul(x, x), rtol=1e-05, atol=1e-05))


if __name__ == "__main__":
    unittest.main()
