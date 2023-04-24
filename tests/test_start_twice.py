# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import unittest

import onnxruntime as ort

from batch_inference import ModelHost, RemoteModelHost
from batch_inference.batcher import ConcatBatcher


class MyModel:
    def __init__(self, model_file):
        self.ort_sess = ort.InferenceSession(model_file)

    def predict_batch(self, x, y):
        res = self.ort_sess.run(None, {"x": x, "y": y})
        return res[0]  # 'res'


class TestRemoteModelHostStartTwice(unittest.TestCase):
    def test_remote_start_twice(self):
        try:
            model_file = os.path.join(os.path.dirname(__file__), "matmul.onnx")
            self.model_host: RemoteModelHost = RemoteModelHost(
                MyModel,
                grpc_port=23333,
                batcher=ConcatBatcher(),
                max_batch_size=32,
            )(model_file)
            self.model_host.start()
            self.model_host.start()
            raise AssertionError
        except RuntimeError as e:
            print(e)
        finally:
            self.model_host.stop()


class TestModelHostStartTwice(unittest.TestCase):
    def test_local_start_twice(self):
        try:
            model_file = os.path.join(os.path.dirname(__file__), "matmul.onnx")
            self.model_host: ModelHost = ModelHost(
                MyModel,
                batcher=ConcatBatcher(),
                max_batch_size=32,
            )(model_file)
            self.model_host.start()
            self.model_host.start()
            raise AssertionError
        except RuntimeError as e:
            print(e)
        finally:
            self.model_host.stop()
