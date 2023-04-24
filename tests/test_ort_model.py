# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import threading
import unittest

import numpy as np
import onnxruntime as ort

from batch_inference import batching


@batching
class MyOrtModel:
    def __init__(self, model_file):
        self.ort_sess = ort.InferenceSession(model_file)

    def predict_batch(self, x, y):
        res = self.ort_sess.run(None, {"x": x, "y": y})
        return res[0]  # 'res'


class TestModelHost(unittest.TestCase):
    def setUp(self) -> None:
        model_file = os.path.join(os.path.dirname(__file__), "matmul.onnx")
        self.model_host = MyOrtModel.host(model_file)
        self.model_host.start()

    def tearDown(self) -> None:
        self.model_host.stop()

    def test_simple(self):
        x = np.random.randn(1, 3, 3).astype("f")
        y = np.random.randn(1, 3, 3).astype("f")
        res = self.model_host.predict(x, y)
        print(res)
        self.assertTrue(np.allclose(res, np.matmul(x, y), rtol=1e-05, atol=1e-05))

    def test_concurrent(self):
        def send_requests():
            for _ in range(0, 10):
                x = np.random.randn(1, 3, 3).astype("f")
                y = np.random.randn(1, 3, 3).astype("f")
                res = self.model_host.predict(x, y)
                self.assertTrue(
                    np.allclose(res, np.matmul(x, y), rtol=1e-05, atol=1e-05),
                )

        threads = []
        for _ in range(0, 10):
            threads.append(threading.Thread(target=send_requests))

        for th in threads:
            th.start()

        for th in threads:
            th.join()


if __name__ == "__main__":
    unittest.main()
